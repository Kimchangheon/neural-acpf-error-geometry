# helper.py
from collections import defaultdict
from typing import Iterable, Tuple, Union, List, Any
import numpy as np
from torch.utils.data import Sampler, DataLoader

ShapeSig = Union[int, Tuple[int, int]]


def _normalize_signature(sig: Any) -> Tuple[int, int]:
    """
    Normalize a sample signature to (N, nl).

    Accepted:
      - int N                -> (N, -1)
      - tuple/list (N, nl)   -> (N, nl)
    """
    if isinstance(sig, (tuple, list)) and len(sig) == 2:
        return (int(sig[0]), int(sig[1]))
    return (int(sig), -1)


def make_size_bucketing_loader(dataset, batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=False):
    """
    Build a DataLoader whose batches are homogeneous in sample shape.

    For the new parquet metadata, the relevant shape is:
        (N, nl)
    where
        N  = number of buses
        nl = number of PPC branch rows

    This is only needed for NON-blockdiag batching.
    """
    signatures = []
    for idx in range(len(dataset)):
        s = dataset[idx]
        signatures.append((int(s["N"]), int(s["nl"])))

    batch_sampler = MultiBucketBatchSampler(
        signatures=signatures,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


class MultiBucketBatchSampler(Sampler):
    """
    Groups dataset indices by homogeneous shape signature and yields
    fixed-size batches.

    New recommended signature:
        (N, nl)

    This ensures all per-bus and per-branch tensors have identical shape
    inside each batch when using the default PyTorch collate.
    """
    def __init__(
        self,
        signatures: Iterable[ShapeSig],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.signatures = [_normalize_signature(s) for s in signatures]
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

        self.buckets = defaultdict(list)   # (N, nl) -> [idx, ...]
        for idx, sig in enumerate(self.signatures):
            self.buckets[sig].append(idx)

    def __iter__(self):
        batches: List[List[int]] = []

        for idxs in self.buckets.values():
            idxs = idxs.copy()

            if self.shuffle:
                np.random.shuffle(idxs)

            if self.drop_last:
                n_full = len(idxs) // self.batch_size
                if n_full == 0:
                    continue
                cut = n_full * self.batch_size
                idxs = idxs[:cut]
            else:
                if len(idxs) == 0:
                    continue

            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)

        if self.shuffle:
            np.random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        total = 0
        for idxs in self.buckets.values():
            if self.drop_last:
                total += len(idxs) // self.batch_size
            else:
                total += (len(idxs) + self.batch_size - 1) // self.batch_size
        return total