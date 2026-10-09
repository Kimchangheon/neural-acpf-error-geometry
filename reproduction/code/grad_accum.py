"""Gradient accumulation, so the effective batch can stay fixed across grids.

The PF dispatcher sizes the micro-batch from a bus-sample budget
(``BATCH ~= BN_BUDGET / n_buses``, capped), so a 9241-bus grid trains at 5-12
graphs per step while a 118-bus grid runs at the 32/64 cap. Gradient noise is
therefore largest exactly where the problem is hardest, and the large-grid runs
are not comparable with the small-grid ones at the optimiser level.

Accumulating the gradient over several micro-batches restores the effective
batch without using more memory. One optimiser step then sees
``BATCH * grad_accum_steps`` graphs, and gradient clipping is applied once per
optimiser step rather than once per micro-batch -- clipping each micro-batch
separately is not the same operation and would keep the noise it is meant to
bound.
"""

from __future__ import annotations

import math

import torch


class GradAccumulator:
    """Accumulate over ``steps`` micro-batches, then clip, step and zero.

    ``steps == 1`` reproduces the original per-batch update exactly, including
    the clip-then-step order, so leaving the flag at its default changes
    nothing.
    """

    def __init__(self, optimizer, model, steps=1, *, max_grad_norm=1.0, scheduler=None,
                 ddp_model=None):
        self.optimizer = optimizer
        self.params = list(model.parameters())
        self.steps = max(1, int(steps))
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        # Kept only so callers can pass it; see the note on backward() for why
        # the no_sync() optimisation is deliberately not attempted here.
        self.ddp_model = ddp_model if hasattr(ddp_model, "no_sync") else None
        self._pending = 0
        self.n_optimizer_steps = 0

    @staticmethod
    def steps_for(batch: int, target_effective_batch: int, world_size: int = 1) -> int:
        """Micro-batches per optimiser step needed to reach the target batch.

        Under DDP every rank runs its own micro-batch simultaneously, so one
        optimiser step already sees ``batch * world_size`` graphs before any
        accumulation.
        """
        if not target_effective_batch or target_effective_batch <= 0:
            return 1
        per_step = max(1, int(batch)) * max(1, int(world_size))
        return max(1, math.ceil(target_effective_batch / per_step))

    def zero(self) -> None:
        """Drop any half-accumulated gradient. Call once at epoch start."""
        self.optimizer.zero_grad(set_to_none=True)
        self._pending = 0

    def backward(self, loss) -> bool:
        """Backward one micro-batch; step when the group is full.

        The loss is divided by ``steps`` so the accumulated gradient is the
        mean over the group, which is what a single batch of that size would
        have produced. Returns True when an optimiser step was taken.
        """
        # Under DDP every micro-batch all-reduces its own gradient. That is
        # correct -- all-reduce and accumulation are both linear, so averaging
        # each piece then summing equals summing then averaging -- but it does
        # communicate once per micro-batch rather than once per optimiser step.
        #
        # The usual fix, ddp_model.no_sync(), cannot be applied from here:
        # DistributedDataParallel decides whether a backward will synchronise
        # during its *forward* pass, and the forward has already run by the
        # time the caller hands us a loss. Wrapping only backward() in
        # no_sync() looks right and does nothing -- verified in
        # tests/test_ddp_grad_accum.py. Suppressing the sync would mean
        # wrapping the driver's whole per-batch body, which is left undone
        # rather than done incorrectly.
        (loss / self.steps).backward()
        self._pending += 1
        if self._pending >= self.steps:
            self._step()
            return True
        return False

    def flush(self) -> bool:
        """Apply a partial group left over at the end of an epoch.

        The leftover micro-batches were each scaled by ``1/steps``, so they are
        rescaled to ``1/pending`` first; without that the last update of every
        epoch would be systematically smaller than the rest.
        """
        if not self._pending:
            return False
        if self._pending < self.steps:
            scale = self.steps / self._pending
            for p in self.params:
                if p.grad is not None:
                    p.grad.mul_(scale)
        # No manual all-reduce is needed: DDP already synchronised each
        # micro-batch's gradient as it was produced. All ranks must still reach
        # flush() the same number of times, which DistributedSampler guarantees
        # by padding shards to equal length.
        self._step()
        return True

    def _step(self) -> None:
        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.params, max_norm=self.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._pending = 0
        self.n_optimizer_steps += 1


def add_grad_accum_args(parser) -> None:
    """Shared CLI surface for the four PF drivers."""
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help=(
            "Micro-batches per optimiser step. 1 (default) keeps the original "
            "per-batch update."
        ),
    )
    parser.add_argument(
        "--target_effective_batch",
        type=int,
        default=0,
        help=(
            "Derive --grad_accum_steps from --BATCH (times the DDP world size) "
            "so one optimiser step sees at least this many graphs. Overrides "
            "--grad_accum_steps when > 0. Use it to hold the effective batch "
            "fixed across grids whose micro-batch is capped by memory, or "
            "across different GPU counts."
        ),
    )


def resolve_accum_steps(args, batch: int, world_size: int = 1) -> int:
    """--target_effective_batch wins when set, else --grad_accum_steps."""
    target = int(getattr(args, "target_effective_batch", 0) or 0)
    if target > 0:
        return GradAccumulator.steps_for(batch, target, world_size)
    return max(1, int(getattr(args, "grad_accum_steps", 1) or 1))
