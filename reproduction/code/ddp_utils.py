"""Multi-node DDP for the PF drivers, launched by srun rather than torchrun.

helma has four GPUs per node, so an eight-GPU run is two nodes. Rather than
nesting torchrun under srun, srun starts one task per GPU directly and the
ranks are read from SLURM's own variables (``SLURM_PROCID`` /
``SLURM_LOCALID`` / ``SLURM_NTASKS``); ``MASTER_ADDR`` and ``MASTER_PORT`` come
from the batch script. ``torchrun`` also works -- it sets RANK/LOCAL_RANK/
WORLD_SIZE itself and those are preferred when present -- so a single-node
``--standalone --nproc_per_node=4`` launch needs no change here.

What the callers have to get right:

* Metrics are summed per rank and must be all-reduced before they mean
  anything. ``reduce_sums`` does that for the running totals and the counts
  together.
* Only rank 0 writes checkpoints, and it writes the *unwrapped* state dict, so
  a DDP checkpoint loads into a single-GPU model through the existing
  ``--init_checkpoint`` / ``--resume_state_dict`` paths.
* ``sampler.set_epoch(epoch)`` every epoch, or every epoch reshuffles the same
  way and the shards never change.
* The final test pass uses an *undistributed* loader, so every rank walks the
  whole test set and the headline number is computed exactly as in the
  single-GPU runs. Reducing identical replicas is harmless: the summed totals
  and the summed counts scale together, so the mean is unchanged.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


@dataclass
class DDPContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.enabled and self.world_size > 1


def _env_int(*names, default=0):
    for n in names:
        v = os.environ.get(n)
        if v not in (None, ""):
            try:
                return int(v)
            except ValueError:
                pass
    return default


def setup_ddp(enabled: bool, *, timeout_hours: float = 3.0) -> DDPContext:
    """Initialise the process group and pick this rank's device."""
    if not enabled:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DDPContext(False, 0, 1, 0, device)

    import torch.distributed as dist

    # torchrun sets RANK/WORLD_SIZE/LOCAL_RANK; srun sets the SLURM_* ones.
    rank = _env_int("RANK", "SLURM_PROCID", default=0)
    world_size = _env_int("WORLD_SIZE", "SLURM_NTASKS", default=1)
    local_rank = _env_int("LOCAL_RANK", "SLURM_LOCALID", default=0)

    if world_size <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DDPContext(False, 0, 1, 0, device)

    # init_process_group reads these; srun-launched jobs get them from the
    # batch script, torchrun sets them itself.
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))

    # Sites differ: some expose all four GPUs to every task on the node (so the
    # task must select its own by local rank), others hand each task a single
    # masked device (where the only valid index is 0). Trust the count.
    visible = torch.cuda.device_count()
    cuda_index = local_rank if visible > 1 else 0
    if visible and local_rank >= visible:
        cuda_index = local_rank % visible
    torch.cuda.set_device(cuda_index)
    os.environ["LOCAL_RANK"] = str(cuda_index)
    local_rank = cuda_index
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=datetime.timedelta(hours=timeout_hours),
    )
    device = torch.device(f"cuda:{local_rank}")
    print(
        f"[ddp] rank {rank}/{world_size} local_rank {local_rank} "
        f"host {os.uname().nodename} device {device}",
        flush=True,
    )
    return DDPContext(True, rank, world_size, local_rank, device)


def wrap_model(model, ctx: DDPContext, *, find_unused_parameters: bool = False,
               static_graph: bool = False):
    """Return the DDP-wrapped model, or the model itself when not distributed.

    ``static_graph=True`` is the fix for a module that reuses a parameter
    within one forward -- GridSFM does, and plain DDP rejects it with "Expected
    to mark a variable ready only once". It requires the set of participating
    parameters and their order to be the same every iteration, which holds here
    because ``share_grid=True`` pins the topology for the whole run. It also
    subsumes ``find_unused_parameters``, so passing both is redundant.
    """
    if not ctx.distributed:
        return model
    from torch.nn.parallel import DistributedDataParallel as DDP

    kwargs = dict(
        device_ids=[ctx.local_rank],
        output_device=ctx.local_rank,
        broadcast_buffers=False,
    )
    if static_graph:
        kwargs["static_graph"] = True
    else:
        kwargs["find_unused_parameters"] = find_unused_parameters
    return DDP(model, **kwargs)


def unwrap(model):
    """The underlying module, so checkpoints never carry a ``module.`` prefix."""
    return getattr(model, "module", model)


def make_loader(dataset, ctx: DDPContext, *, shuffle: bool, distributed: bool = True,
                **loader_kwargs):
    """DataLoader plus its DistributedSampler (None when not sharding).

    ``distributed=False`` gives every rank the whole dataset, which is what the
    rank-0 test pass wants.
    """
    if not (ctx.distributed and distributed):
        return DataLoader(dataset, shuffle=shuffle, **loader_kwargs), None
    sampler = DistributedSampler(
        dataset,
        num_replicas=ctx.world_size,
        rank=ctx.rank,
        shuffle=shuffle,
        drop_last=False,          # pads instead, so every rank runs the same
    )                             # number of batches -- flush() relies on it
    return DataLoader(dataset, sampler=sampler, shuffle=False, **loader_kwargs), sampler


def reduce_sums(values, ctx: DDPContext):
    """All-reduce a list of scalar running totals with SUM.

    Totals and their denominators must both go through this, in the same order
    on every rank, before any mean is taken.
    """
    if not ctx.distributed:
        return [float(v) for v in values]
    import torch.distributed as dist

    t = torch.tensor([float(v) for v in values], dtype=torch.float64, device=ctx.device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return [float(x) for x in t.tolist()]


def broadcast_flag(value: bool, ctx: DDPContext) -> bool:
    """Agree on a rank-0 decision, e.g. whether this epoch is a new best."""
    if not ctx.distributed:
        return bool(value)
    import torch.distributed as dist

    t = torch.tensor([1 if value else 0], dtype=torch.int64, device=ctx.device)
    dist.broadcast(t, src=0)
    return bool(int(t.item()))


def barrier(ctx: DDPContext) -> None:
    if not ctx.distributed:
        return
    import torch.distributed as dist

    dist.barrier()


def cleanup(ctx: DDPContext) -> None:
    if not ctx.distributed:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def add_ddp_args(parser) -> None:
    parser.add_argument(
        "--DDP",
        action="store_true",
        default=False,
        help=(
            "DistributedDataParallel. Ranks come from RANK/WORLD_SIZE/LOCAL_RANK "
            "or the SLURM_* equivalents. --BATCH is per rank, so the effective "
            "batch is BATCH * world_size * grad_accum_steps."
        ),
    )
    parser.add_argument(
        "--ddp_timeout_hours",
        type=float,
        default=3.0,
        help="NCCL collective timeout; raise it when staging is slow.",
    )
    parser.add_argument(
        "--ddp_find_unused_parameters",
        action="store_true",
        default=False,
        help="Needed only if some parameters get no gradient in a forward pass.",
    )
    parser.add_argument(
        "--ddp_static_graph",
        action="store_true",
        default=False,
        help=(
            "For modules that reuse a parameter inside one forward (GridSFM). "
            "Requires the participating parameters to be the same every "
            "iteration; supersedes --ddp_find_unused_parameters."
        ),
    )


class _Devnull:
    """Swallow stdout writes on non-main ranks."""

    def write(self, *_args):
        return 0

    def flush(self):
        pass

    def isatty(self):
        return False


def quiet_non_main(ctx: DDPContext) -> None:
    """Silence stdout everywhere except rank 0.

    stderr is deliberately left alone, so a traceback raised on any rank still
    reaches the job's .err file.
    """
    import sys

    if ctx.distributed and not ctx.is_main:
        sys.stdout = _Devnull()
