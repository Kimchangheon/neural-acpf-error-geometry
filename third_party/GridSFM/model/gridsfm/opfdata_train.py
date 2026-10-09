"""OPFData adapter for GridSFM fine-tuning.

Wraps PyG's `torch_geometric.datasets.OPFDataset` (the Google DeepMind
OPFData benchmark, https://arxiv.org/abs/2406.07234) so each graph yields
a HeteroData ready for `SyntheticMixedDataset` + `compute_loss`. The
public surface is `OPFDataAdapterDataset`.
"""
from __future__ import annotations

import os.path as osp
from typing import Optional

import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.datasets import OPFDataset


class _CachedOPFDataset(OPFDataset):
    """OPFDataset that skips re-download when `<root>/.../processed/*.pt` exists."""

    def _download(self):
        if all(osp.exists(p) for p in self.processed_paths):
            return
        super()._download()

    def _process(self):
        if all(osp.exists(p) for p in self.processed_paths):
            return
        super()._process()


class OPFDataAdapterDataset(Dataset):
    """Yield OPFData samples adapted for FT.

    Args:
      root: storage location for the PyG OPFDataset cache.
      case_name: pglib case name (e.g. ``"pglib_opf_case6470_rte"``).
      variant: ``"fulltop"`` or ``"n1"``.
      split: ``"train"`` (13.5k graphs) / ``"val"`` (750) / ``"test"`` (750).
      n_graphs: cap on `__len__` / `__getitem__`. Note that the cap is
        applied AFTER the underlying PyG ``OPFDataset`` finishes its
        download + process step, so ``n_graphs=1`` does NOT make the
        constructor lightweight: the first call still downloads and
        decodes the full ``num_groups * 15000``-graph cache. Subsequent
        calls hit ``_CachedOPFDataset`` and return instantly. ``None``
        means no cap (use the full split, ~13.5k / 750 / 750).
      num_groups: how many 15k-graph shards to download (default 1).
      transform: callable applied AFTER schema adaptation. The model
        forward requires cycle-basis + Hodge PE features on every graph
        (HodgePE reads `data['cycle']`). Either pass the cycle/PE
        transform here, OR pass `None` when wrapping in
        `SyntheticMixedDataset` (set the transform on the wrapper
        instead; applying it on both sides re-runs the cycle+PE attach
        twice per sample).
    """

    def __init__(
        self,
        root: str,
        case_name: str,
        variant: str = "fulltop",
        split: str = "train",
        n_graphs: Optional[int] = None,
        num_groups: int = 1,
        transform=None,
    ):
        if variant not in ("fulltop", "n1"):
            raise ValueError(f"variant must be 'fulltop' or 'n1', got {variant!r}")
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train' | 'val' | 'test', got {split!r}")
        self.case_name = case_name
        self.variant = variant
        self.split = split
        self.transform = transform
        self._inner = _CachedOPFDataset(
            root=root,
            split=split,
            case_name=case_name,
            num_groups=int(num_groups),
            topological_perturbations=(variant == "n1"),
        )
        self._n = (len(self._inner) if n_graphs is None
                   else min(int(n_graphs), len(self._inner)))

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> HeteroData:
        # OPFData drops infeasibles per the paper, so every yielded graph
        # is feasible by construction; pin the label so the FT feas head
        # sees a uniform schema.
        #
        # Returned graph is NOT a deep copy of the PyG cache. Consumers
        # that mutate (e.g. `SyntheticMixedDataset`) MUST `.clone()` first;
        # the wrapper already does. Dropping a defensive `.clone()` here
        # saves an extra per-sample tensor-clone on large grids.
        g = self._inner[int(idx)]
        g.feasible = torch.tensor(1, dtype=torch.long)
        if self.transform is not None:
            g = self.transform(g)
        return g

    def __repr__(self) -> str:
        return (f"OPFDataAdapterDataset(case={self.case_name}, "
                f"variant={self.variant}, split={self.split}, "
                f"n={self._n}/{len(self._inner)})")
