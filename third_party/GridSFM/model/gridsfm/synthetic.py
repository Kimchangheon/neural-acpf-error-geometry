"""Synthetic infeasibility wrapper for GridSFM fine-tuning.

`SyntheticMixedDataset` wraps a feasible base dataset and (with probability
`infeas_prob`) applies one of the `_MODE_NAMES` perturbation modes that turn a feasible
graph into an infeasible one. Used at training time to teach the
feasibility classifier and the regression heads from infeasible signals.

Modes:
  0. voltage_squeeze       - tighten Vmin/Vmax on 30-60% of buses + raise Qd
  1. thermal_bottleneck    - derate a connected corridor of lines + mild load shift
  2. angle_tighten         - tighten angmin/angmax on 20-40% of edges + raise Pd
  3. capacity_aware_spike  - single-bus load injection scaled to incident
                              transmission capacity (corridor-pocket infeas)

For each index:
  * With probability (1 - infeas_prob): returns the original graph unchanged
    (`.clone()`'d, with `feasible=1` and `perturb_mode=-1`).
  * With probability infeas_prob: `.clone()`'s, applies one mode, sets
    `feasible=0`, zeros per-node-type `.y` labels, and returns.

Deterministic per (seed, epoch, index); call `set_epoch(epoch)` to roll
the perturbation seeding between epochs. The thermal_bottleneck mode
caches an adjacency per unique topology (O(unique_topologies) memory);
on a full n-1 split (~13.5k topologies), keep `num_workers <= 2` or
preprocess the perturbations.
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import (
    AC_LINE_ANGMIN_IDX, AC_LINE_ANGMAX_IDX,
    AC_LINE_KEY, AC_LINE_RATE_A_IDX,
    BUS_VMIN_IDX, BUS_VMAX_IDX,
    LOAD_PD_IDX, LOAD_QD_IDX,
    TRANSFORMER_KEY, TR_ANGMIN_IDX, TR_ANGMAX_IDX, TR_RATE_A_IDX,
)


_ANGMIN_IDX_BY_KEY = {
    AC_LINE_KEY:     AC_LINE_ANGMIN_IDX,
    TRANSFORMER_KEY: TR_ANGMIN_IDX,
}
_ANGMAX_IDX_BY_KEY = {
    AC_LINE_KEY:     AC_LINE_ANGMAX_IDX,
    TRANSFORMER_KEY: TR_ANGMAX_IDX,
}


# Each mode `i` is implemented by `self._perturb_<_MODE_NAMES[i]>`; adding a
# 5th mode means adding one name here + one `_perturb_<name>` method + one
# entry to `_DEFAULT_MODE_WEIGHTS`. The dispatcher resolves the method by
# name, so the (name, function) binding can't drift.
_MODE_NAMES: Tuple[str, ...] = (
    "voltage_squeeze",
    "thermal_bottleneck",
    "angle_tighten",
    "capacity_aware_spike",
)
_DEFAULT_MODE_WEIGHTS: Tuple[float, ...] = (0.25, 0.20, 0.20, 0.35)
assert len(_MODE_NAMES) == len(_DEFAULT_MODE_WEIGHTS)


def _derate_edges(ea: torch.Tensor, col_idx: int, idxs: list,
                  rng: random.Random, lo: float, hi: float) -> None:
    """In-place derating of `ea[:, col_idx]` at the given row indices."""
    idx_t = torch.tensor(idxs, dtype=torch.long, device=ea.device)
    factors = torch.tensor([rng.uniform(lo, hi) for _ in idxs],
                           dtype=ea.dtype, device=ea.device)
    ea[idx_t, col_idx] *= factors


class SyntheticMixedDataset(Dataset):
    """Wrap a feasible dataset; inject one of the `_MODE_NAMES`
    perturbation modes with probability `infeas_prob`.

    Args:
      base_dataset: any PyG Dataset of feasible graphs.
      infeas_prob: probability a sample is perturbed (default 0.5).
      seed: RNG seed for deterministic (seed, epoch, idx) hashing.
      transform: optional callable applied AFTER perturbation
        (typical use: `prepare_for_inference` + cycle/PE attach).
      mode_weights: tuple of len(_MODE_NAMES) summing to 1.0, indexed in
        the same order as `_MODE_NAMES`. Default = `_DEFAULT_MODE_WEIGHTS`.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        infeas_prob: float = 0.5,
        seed: int = 42,
        transform: Optional[Callable] = None,
        mode_weights: Optional[Tuple[float, ...]] = None,
    ):
        self.base = base_dataset
        self.infeas_prob = infeas_prob
        self.seed = seed
        self.transform = transform
        self._epoch = 0

        if mode_weights is None:
            mode_weights = _DEFAULT_MODE_WEIGHTS
        if len(mode_weights) != len(_MODE_NAMES):
            raise ValueError(
                f"mode_weights must have {len(_MODE_NAMES)} elements "
                f"({', '.join(_MODE_NAMES)}); got {len(mode_weights)}."
            )
        total = sum(mode_weights)
        if abs(total - 1.0) >= 1e-6:
            raise ValueError(f"mode_weights must sum to 1.0, got {total}")
        self.mode_weights = tuple(mode_weights)

        self._cum_weights = []
        cum = 0.0
        for w in self.mode_weights:
            cum += w
            self._cum_weights.append(cum)

        # Per-topology BFS-adjacency cache for thermal_bottleneck mode.
        # Keyed by edge_index bytes (collision-free unlike Python hash).
        self._adj_cache: Dict[bytes, tuple] = {}

        self.case_name = getattr(base_dataset, "case_name", "unknown")

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.base)

    def _select_mode(self, rng: random.Random) -> int:
        """Sample a perturbation mode index in [0, len(_MODE_NAMES))."""
        r = rng.random()
        for i, cw in enumerate(self._cum_weights):
            if r < cw:
                return i
        return len(_MODE_NAMES) - 1

    # ── Mode 0: voltage_squeeze ─────────────────────────────────────────

    def _perturb_voltage_squeeze(self, graph, rng: random.Random):
        """Tighten Vmin/Vmax on 30-60% of buses + raise Qd to stress voltage.

        Returns (graph, mutated) - mutated=False signals the dispatcher to
        leave `feasible` and `.y` alone (the graph wasn't actually changed).
        """
        bus_x = graph["bus"].x
        n_bus = bus_x.size(0)
        if n_bus == 0:
            return graph, False

        vmin = bus_x[:, BUS_VMIN_IDX].clone()
        vmax = bus_x[:, BUS_VMAX_IDX].clone()
        vspan = (vmax - vmin).clamp(min=1e-6)

        frac = rng.uniform(0.30, 0.60)
        n_squeeze = max(1, int(n_bus * frac))
        squeeze_idxs = rng.sample(range(n_bus), n_squeeze)
        idx_t = torch.tensor(squeeze_idxs, dtype=torch.long, device=bus_x.device)

        shrink = torch.tensor(
            [rng.uniform(0.10, 0.40) for _ in squeeze_idxs],
            dtype=bus_x.dtype, device=bus_x.device,
        )
        new_span = vspan[idx_t] * shrink
        vmid = (vmin[idx_t] + vmax[idx_t]) * 0.5
        bus_x[idx_t, BUS_VMIN_IDX] = vmid - new_span * 0.5
        bus_x[idx_t, BUS_VMAX_IDX] = vmid + new_span * 0.5

        if "load" in graph.node_types:
            load_x = graph["load"].x
            if load_x.size(0) > 0:
                load_x[:, LOAD_QD_IDX] *= rng.uniform(1.2, 1.8)

        return graph, True

    # ── Mode 1: thermal_bottleneck ──────────────────────────────────────

    def _perturb_thermal_bottleneck(self, graph, rng: random.Random):
        """Derate a BFS-grown corridor of lines + mild heterogeneous load shift.

        Returns (graph, mutated). See `_perturb_voltage_squeeze`.
        """
        if AC_LINE_KEY not in graph.edge_types:
            return graph, False
        ea = graph[AC_LINE_KEY].edge_attr
        ei = graph[AC_LINE_KEY].edge_index
        n_edges = ea.size(0)
        if n_edges < 3:
            return graph, False

        cache_key = ei.detach().cpu().numpy().tobytes()
        cached = self._adj_cache.get(cache_key)
        if cached is not None:
            src, dst, bus_to_edges = cached
        else:
            src, dst = ei[0].tolist(), ei[1].tolist()
            bus_to_edges: Dict[int, List[int]] = {}
            for e_idx in range(n_edges):
                for b in (src[e_idx], dst[e_idx]):
                    bus_to_edges.setdefault(b, []).append(e_idx)
            self._adj_cache[cache_key] = (src, dst, bus_to_edges)

        seed_edge = rng.randint(0, n_edges - 1)
        corridor = {seed_edge}
        target_size = max(2, int(n_edges * rng.uniform(0.10, 0.25)))

        frontier_buses = {src[seed_edge], dst[seed_edge]}
        visited_buses = set(frontier_buses)

        while len(corridor) < target_size and frontier_buses:
            next_frontier = set()
            for bus in frontier_buses:
                for e_idx in bus_to_edges.get(bus, []):
                    if e_idx not in corridor:
                        corridor.add(e_idx)
                        other = dst[e_idx] if src[e_idx] == bus else src[e_idx]
                        if other not in visited_buses:
                            next_frontier.add(other)
                            visited_buses.add(other)
                    if len(corridor) >= target_size:
                        break
                if len(corridor) >= target_size:
                    break
            frontier_buses = next_frontier

        corridor_idxs = list(corridor)
        _derate_edges(ea, AC_LINE_RATE_A_IDX, corridor_idxs, rng, 0.05, 0.20)

        if TRANSFORMER_KEY in graph.edge_types:
            t_ea = graph[TRANSFORMER_KEY].edge_attr
            t_ei = graph[TRANSFORMER_KEY].edge_index
            n_t = t_ea.size(0)
            if n_t > 0:
                t_src, t_dst = t_ei[0].tolist(), t_ei[1].tolist()
                t_corridor = [
                    i for i in range(n_t)
                    if t_src[i] in visited_buses or t_dst[i] in visited_buses
                ]
                if t_corridor:
                    _derate_edges(t_ea, TR_RATE_A_IDX, t_corridor, rng, 0.10, 0.30)

        if "load" in graph.node_types:
            load_x = graph["load"].x
            n_load = load_x.size(0)
            if n_load > 0:
                deltas = torch.tensor(
                    [rng.uniform(-0.1, 0.3) for _ in range(n_load)],
                    dtype=load_x.dtype, device=load_x.device,
                )
                load_x[:, LOAD_PD_IDX] *= (1.0 + deltas)
                load_x[:, LOAD_QD_IDX] *= (1.0 + deltas)

        return graph, True

    # ── Mode 2: angle_tighten ───────────────────────────────────────────

    def _perturb_angle_tighten(self, graph, rng: random.Random):
        """Tighten angmin/angmax on 20-40% of edges + mild active load increase.

        Returns (graph, mutated). See `_perturb_voltage_squeeze`.
        """
        mutated = False
        for et_key in (AC_LINE_KEY, TRANSFORMER_KEY):
            if et_key not in graph.edge_types:
                continue
            ea = graph[et_key].edge_attr
            n_edges = ea.size(0)
            if n_edges == 0:
                continue

            amin_idx = _ANGMIN_IDX_BY_KEY[et_key]
            amax_idx = _ANGMAX_IDX_BY_KEY[et_key]

            frac = rng.uniform(0.20, 0.40)
            n_tighten = max(1, int(n_edges * frac))
            idxs = rng.sample(range(n_edges), n_tighten)
            idx_t = torch.tensor(idxs, dtype=torch.long, device=ea.device)

            angmin = ea[idx_t, amin_idx].clone()
            angmax = ea[idx_t, amax_idx].clone()
            ang_span = (angmax - angmin).clamp(min=1e-6)

            shrink = torch.tensor(
                [rng.uniform(0.05, 0.20) for _ in idxs],
                dtype=ea.dtype, device=ea.device,
            )
            new_span = ang_span * shrink
            ang_mid = (angmin + angmax) * 0.5
            ea[idx_t, amin_idx] = ang_mid - new_span * 0.5
            ea[idx_t, amax_idx] = ang_mid + new_span * 0.5
            mutated = True

        if "load" in graph.node_types:
            load_x = graph["load"].x
            if load_x.size(0) > 0:
                p_scale = rng.uniform(1.05, 1.3)
                load_x[:, LOAD_PD_IDX] *= p_scale
                load_x[:, LOAD_QD_IDX] *= p_scale
                mutated = True

        return graph, mutated

    # ── Mode 3: capacity_aware_spike ────────────────────────────────────

    def _perturb_capacity_aware_spike(self, graph, rng: random.Random):
        """Single-bus injection scaled to incident transmission capacity.

        Targets corridor-pocket / transmission-bottleneck infeasibility. Picks
        1-3 with-load buses (smoothed-inverse-Σrate_a weighted, biasing toward
        weakly-connected buses) and adds a `target_ratio · Σrate_a` spike to
        the largest existing load at each. `target_ratio` is sampled from a
        3-band mixture in [0.75, 5.0].

        Returns (graph, mutated). See `_perturb_voltage_squeeze`.
        """
        if "load" not in graph.node_types:
            return graph, False
        load_x = graph["load"].x
        n_load = load_x.size(0)
        if n_load == 0:
            return graph, False

        bus_dev = graph["bus"].x.device
        n_bus = graph["bus"].x.size(0)
        sum_rate_a = torch.zeros(n_bus, dtype=torch.float32, device=bus_dev)
        for et_key, ra_idx in (
            (AC_LINE_KEY, AC_LINE_RATE_A_IDX),
            (TRANSFORMER_KEY, TR_RATE_A_IDX),
        ):
            if et_key not in graph.edge_types:
                continue
            ei = graph[et_key].edge_index
            ea = graph[et_key].edge_attr
            if ea.size(0) == 0 or ra_idx >= ea.size(1):
                continue
            ra_vals = ea[:, ra_idx].to(torch.float32)
            sum_rate_a.scatter_add_(0, ei[0].long(), ra_vals)
            sum_rate_a.scatter_add_(0, ei[1].long(), ra_vals)

        link_key = ("load", "load_link", "bus")
        if link_key not in graph.edge_types:
            return graph, False
        l2b = graph[link_key].edge_index[1].tolist()
        bus2load_idxs: Dict[int, List[int]] = {}
        for li, b in enumerate(l2b):
            bus2load_idxs.setdefault(int(b), []).append(li)
        load_buses = list(bus2load_idxs.keys())
        if not load_buses:
            return graph, False

        n_spikes = rng.choices([1, 2, 3], weights=[0.7, 0.2, 0.1], k=1)[0]
        n_spikes = min(n_spikes, len(load_buses))
        load_bus_caps = [float(sum_rate_a[b]) for b in load_buses]
        _med = float(np.median(load_bus_caps)) if load_bus_caps else 1.0
        weights = [1.0 / max(c + _med, 1e-3) for c in load_bus_caps]
        avail = list(load_buses)
        avail_w = list(weights)
        chosen = []
        for _ in range(n_spikes):
            b = rng.choices(avail, weights=avail_w, k=1)[0]
            i = avail.index(b)
            chosen.append(b)
            avail.pop(i)
            avail_w.pop(i)

        def _sample_target_ratio():
            r = rng.random()
            if r < 0.30:
                return rng.uniform(0.75, 1.0)
            elif r < 0.80:
                return rng.uniform(1.0, 2.0)
            else:
                return rng.uniform(2.0, 5.0)

        mutated = False
        for bus in chosen:
            ra_b = float(sum_rate_a[bus])
            if ra_b < 1e-3:
                continue
            target_ratio = _sample_target_ratio()
            delta_pu = target_ratio * ra_b
            li = max(bus2load_idxs[bus],
                     key=lambda i: float(load_x[i, LOAD_PD_IDX].abs()))
            p_orig = float(load_x[li, LOAD_PD_IDX])
            q_orig = float(load_x[li, LOAD_QD_IDX])
            if abs(p_orig) > 1e-3:
                spike_qp_ratio = q_orig / p_orig
            else:
                spike_qp_ratio = rng.uniform(0.10, 0.35) * rng.choice([-1.0, 1.0])
            load_x[li, LOAD_PD_IDX] = p_orig + delta_pu
            load_x[li, LOAD_QD_IDX] = q_orig + delta_pu * spike_qp_ratio
            mutated = True

        return graph, mutated

    # ── __getitem__ ─────────────────────────────────────────────────────

    def __getitem__(self, idx: int):
        rng = random.Random(self.seed + self._epoch * 1_000_000 + idx)
        # `.clone()` is ~5000x faster than `copy.deepcopy()` on case6470-scale
        # HeteroData: PyG's clone uses a tuned per-store tensor-clone fast
        # path; Python's deepcopy walks the full object graph through the
        # memo machinery and degrades drastically after the first call.
        graph = self.base[idx].clone()

        if rng.random() < self.infeas_prob:
            mode = self._select_mode(rng)
            graph, mutated = getattr(self, f"_perturb_{_MODE_NAMES[mode]}")(graph, rng)
        else:
            mode, mutated = -1, False

        if mutated:
            graph.feasible = torch.tensor(0, dtype=torch.long)
            graph.perturb_mode = torch.tensor(mode, dtype=torch.long)
            for ntype in graph.node_types:
                if hasattr(graph[ntype], "y"):
                    graph[ntype].y = torch.zeros_like(graph[ntype].y)
        else:
            # Either no perturbation rolled, or the perturbation early-returned
            # mutated=False (e.g. n_edges<3 for thermal_bottleneck).
            graph.feasible = torch.tensor(1, dtype=torch.long)
            graph.perturb_mode = torch.tensor(-1, dtype=torch.long)

        self._ensure_edge_active_mask(graph)
        if self.transform is not None:
            graph = self.transform(graph)
        return graph

    @staticmethod
    def _ensure_edge_active_mask(graph) -> None:
        """Ensure `edge_active_mask` exists on every present edge type so
        PyG `Batch.from_data_list` doesn't raise on heterogeneous attribute
        presence across the batch.
        """
        for et_name in ("ac_line", "transformer"):
            et_key = ("bus", et_name, "bus")
            if et_key not in graph.edge_types:
                continue
            ed = graph[et_key]
            n_e = ed.edge_index.size(1)
            if (not hasattr(ed, "edge_active_mask")
                    or ed.edge_active_mask is None
                    or int(ed.edge_active_mask.numel()) != n_e):
                ed.edge_active_mask = torch.ones(n_e, dtype=torch.bool)

    def __repr__(self) -> str:
        return (f"SyntheticMixedDataset(n={len(self)}, "
                f"case={self.case_name}, infeas_prob={self.infeas_prob}, "
                f"modes={self.mode_weights})")


# Import-time integrity check: every name in `_MODE_NAMES` must have a
# matching `_perturb_<name>` method, and the default weights must be a
# valid distribution. Catches typos and stale entries at module load
# rather than mid-epoch via `getattr` AttributeError.
for _n in _MODE_NAMES:
    assert hasattr(SyntheticMixedDataset, f"_perturb_{_n}"), \
        f"_MODE_NAMES references _perturb_{_n} but no such method exists"
assert abs(sum(_DEFAULT_MODE_WEIGHTS) - 1.0) < 1e-6, \
    f"_DEFAULT_MODE_WEIGHTS must sum to 1.0, got {sum(_DEFAULT_MODE_WEIGHTS)}"
