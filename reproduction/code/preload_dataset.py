"""Decode a split once and keep it in RAM, so epochs stop re-reading parquet.

Measured on the v2 corpus, graphkit at batch 64 takes 630-680 s/epoch on every
grid from 4 buses to 200 -- a fifty-fold range in arithmetic with no change in
wall time. That flatness is the tell: the epoch is not computing, it is
decoding parquet, at roughly 53 ms per sample on a single thread. The GPU sits
idle between batches, which is what ClusterCockpit reports as a sawtooth
``acc_utilization`` with ``flops_any`` near zero.

``share_grid=True`` already caches the twenty grid-only tensors and the Ybus and
hands every row the *same* tensor objects, so the only per-row payload is
``U_start``, ``U_newton``, ``V_start``, ``V_newton``, ``S_start`` and
``S_newton`` -- on the order of a hundred bytes per bus. Holding a whole split
is therefore cheap next to a 750 GB node, and it makes every epoch after the
first a pure indexing operation.

Nothing about the data changes: the cached rows are the very objects the lazy
dataset would have produced, so a preloaded run and a lazy run see identical
tensors in identical order.
"""

from __future__ import annotations

import time

import torch
from torch.utils.data import Dataset


def _unique_bytes(rows) -> int:
    """Bytes actually held, counting each storage once.

    Rows share the grid tensors by reference, so summing every tensor in every
    row would report the Ybus once per sample and overstate the cost wildly.
    """
    seen = set()
    total = 0
    for row in rows:
        for v in row.values():
            if not torch.is_tensor(v):
                continue
            st = v.untyped_storage() if hasattr(v, "untyped_storage") else v.storage()
            key = (st.data_ptr(), st.nbytes())
            if key in seen:
                continue
            seen.add(key)
            total += key[1]
    return total


class PreloadedDataset(Dataset):
    """A split materialised in RAM. Same rows, same order, no parquet."""

    def __init__(self, split, *, label: str = "", log_every: int = 0):
        # Stay a drop-in for the Subset we replace: the pignn driver builds its
        # bucketing sampler from ``split.indices`` and reads signatures off
        # ``split.dataset``, and positional indexing is unchanged either way.
        self.indices = getattr(split, "indices", None)
        self.dataset = getattr(split, "dataset", split)

        n = len(split)
        t0 = time.time()
        self.rows = []
        for i in range(n):
            self.rows.append(split[i])
            if log_every and (i + 1) % log_every == 0:
                el = time.time() - t0
                print(
                    f"[preload] {label} {i + 1}/{n} "
                    f"({el:.0f}s elapsed, {1000 * el / (i + 1):.1f} ms/row, "
                    f"eta {el * (n - i - 1) / (i + 1):.0f}s)",
                    flush=True,
                )
        self.seconds = time.time() - t0
        self.nbytes = _unique_bytes(self.rows)
        print(
            f"[preload] {label}: {n} rows in {self.seconds:.1f}s, "
            f"{self.nbytes / 2**30:.2f} GiB resident "
            f"({self.nbytes / max(n, 1) / 1024:.1f} KiB/row)",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def preload_split(split, label: str = "", log_every: int = 2000):
    """Materialise ``split``; returns it unchanged when it already is."""
    if isinstance(split, PreloadedDataset):
        return split
    return PreloadedDataset(split, label=label, log_every=log_every)


def add_preload_args(parser) -> None:
    # On by default: measured on case300/graphkit it cuts the per-pass time
    # from 233 s to 55 s (4.2x) and the trajectory difference against a lazy
    # run is 0.24% at epoch 3, against an 11% spread between two identical
    # runs -- i.e. far inside the run-to-run noise of the GPU scatter kernels.
    parser.add_argument(
        "--preload_ram",
        action="store_true",
        default=True,
        help=(
            "Decode the train and valid splits once into RAM instead of "
            "re-reading parquet every epoch (default). Results are unchanged; "
            "only the time spent getting the data to the GPU differs."
        ),
    )
    parser.add_argument(
        "--no_preload_ram",
        dest="preload_ram",
        action="store_false",
        help="Stream from parquet every epoch, as before preload was added.",
    )
    parser.add_argument(
        "--preload_test",
        action="store_true",
        default=False,
        help=(
            "Also preload the test split. It is walked once, so this usually "
            "costs memory for nothing."
        ),
    )
