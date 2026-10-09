#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import copy
import warnings
import numpy as np
from typing import Optional, Dict, Callable, Any, Union, Sequence
from pandapower.pypower.makeSbus import makeSbus

# ============================================================
# Types
# ============================================================

CaseSource = Union[
    str,                    # pandapower case name OR CGMES zip path OR folder path
    Callable[..., Any],     # callable returning a pandapower net
    Any,                    # already-built pandapower net
    Sequence[str],          # list/tuple of CGMES zip paths
    Dict[str, Any],         # CGMES config dict
]


# ============================================================
# Helpers
# ============================================================

def _optional_branch_indices():
    """Return optional pandapower extended branch column indices if available."""
    try:
        from pandapower.pypower.idx_brch import BR_G
    except Exception:
        BR_G = None
    try:
        from pandapower.pypower.idx_brch import BR_R_ASYM
    except Exception:
        BR_R_ASYM = None
    try:
        from pandapower.pypower.idx_brch import BR_X_ASYM
    except Exception:
        BR_X_ASYM = None
    try:
        from pandapower.pypower.idx_brch import BR_B_ASYM
    except Exception:
        BR_B_ASYM = None
    try:
        from pandapower.pypower.idx_brch import BR_G_ASYM
    except Exception:
        BR_G_ASYM = None
    return BR_G, BR_R_ASYM, BR_X_ASYM, BR_B_ASYM, BR_G_ASYM


def _col(row: np.ndarray, idx, default=0.0) -> float:
    if idx is None or idx >= row.shape[0]:
        return default
    return float(row[idx])


def _looks_like_pandapower_net(obj) -> bool:
    """
    Lightweight duck-typing check for a pandapower net.
    """
    return hasattr(obj, "bus") and hasattr(obj, "__getitem__")


def _looks_like_cgmes_source(case_fn: CaseSource) -> bool:
    """
    Heuristic used to decide whether Model-A CGMES cleanup should be applied.
    """
    if isinstance(case_fn, dict):
        return "cgmes_files" in case_fn

    if isinstance(case_fn, (list, tuple)):
        return True

    if isinstance(case_fn, (str, os.PathLike)):
        expanded = os.path.abspath(os.path.expanduser(str(case_fn)))
        return os.path.exists(expanded)

    return False


def _dense_ybus_from_ppc_internal(ppc_int) -> np.ndarray:
    """Safely extract dense Ybus from ppc internal dict."""
    Y = ppc_int["Ybus"]
    if hasattr(Y, "toarray"):
        return Y.toarray().astype(np.complex128)
    return np.asarray(Y, dtype=np.complex128)


def _get_baseMVA_from_ppc(ppc_obj, fallback=None) -> float:
    """
    Read baseMVA from a PPC-like dict.
    Some pandapower versions keep baseMVA only on net._ppc, not on net._ppc["internal"].
    """
    if isinstance(ppc_obj, dict) and "baseMVA" in ppc_obj:
        return float(ppc_obj["baseMVA"])
    if fallback is not None:
        return float(fallback)
    raise KeyError("baseMVA")


def _cgmes_from_files(file_list, converter_kwargs: Optional[Dict[str, Any]] = None):
    """
    Load CGMES zip files into a pandapower net.
    Works with both import styles that exist in different pandapower versions.
    """
    converter_kwargs = converter_kwargs or {}
    files = [str(f) for f in file_list]

    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"CGMES file does not exist: {f}")

    warnings.simplefilter(action="ignore", category=FutureWarning)

    from pandapower.converter.cim import from_cim as cim_import

    if hasattr(cim_import, "from_cim"):
        net = cim_import.from_cim(file_list=files, **converter_kwargs)
    else:
        net = cim_import(file_list=files, **converter_kwargs)

    return net


def _normalize_cgmes_files_from_path(path_str: str):
    """
    Accept either:
      - a single zip file path
      - a directory containing *.zip files
    """
    path_str = os.path.abspath(os.path.expanduser(str(path_str)))

    if os.path.isfile(path_str):
        return [path_str]

    if os.path.isdir(path_str):
        zips = sorted(glob.glob(os.path.join(path_str, "*.zip")))
        if not zips:
            raise FileNotFoundError(f"No .zip files found in directory: {path_str}")
        return zips

    raise FileNotFoundError(f"Path does not exist: {path_str}")


def _default_case_name_from_files(files):
    if len(files) == 1:
        return os.path.splitext(os.path.basename(files[0]))[0]
    parent = os.path.basename(os.path.dirname(files[0]))
    return parent if parent else "cgmes_case"


def _apply_model_a_cgmes_cleanup(net) -> None:
    """
    Apply the same practical cleanup that was validated for the LVN CGMES Model A flow:
      - disable unsupplied buses
      - disable zero-impedance lines
      - clamp suspicious generator voltage setpoints
    """
    from pandapower.topology import unsupplied_buses

    try:
        isolated = unsupplied_buses(net)
    except Exception:
        isolated = set()

    if len(isolated) > 0 and hasattr(net, "bus") and "in_service" in net.bus.columns:
        net.bus.loc[list(isolated), "in_service"] = False

    if (
        hasattr(net, "line")
        and len(net.line) > 0
        and all(c in net.line.columns for c in ["r_ohm_per_km", "x_ohm_per_km", "in_service"])
    ):
        zero_imp_mask = (
            (net.line["r_ohm_per_km"] == 0) &
            (net.line["x_ohm_per_km"] == 0)
        )
        if zero_imp_mask.any():
            net.line.loc[zero_imp_mask, "in_service"] = False

    if (
        hasattr(net, "gen")
        and len(net.gen) > 0
        and "vm_pu" in net.gen.columns
    ):
        bad_vm = (net.gen["vm_pu"] < 0.8) | (net.gen["vm_pu"] > 1.2)
        if bad_vm.any():
            net.gen.loc[bad_vm, "vm_pu"] = 1.0


def _zone_graph(net):
    """Undirected adjacency over in-service lines and transformers."""
    import collections

    n = len(net.bus)
    idx = {b: i for i, b in enumerate(net.bus.index)}
    adj = collections.defaultdict(list)
    for tbl, a, b in (("line", "from_bus", "to_bus"), ("trafo", "hv_bus", "lv_bus")):
        t = getattr(net, tbl, None)
        if t is None or not len(t):
            continue
        live = t["in_service"].to_numpy(bool) if "in_service" in t else np.ones(len(t), bool)
        for f, to, ok in zip(t[a].to_numpy(), t[b].to_numpy(), live):
            if ok and f in idx and to in idx:
                adj[idx[f]].append(idx[to]); adj[idx[to]].append(idx[f])
    return n, idx, adj


def _bus_load_weights(net, idx):
    """Base-case active load per bus -- the weight the partition balances.

    Read from the ORIGINAL network before any scenario perturbation, so the
    partition is a property of the grid rather than of the sample.
    """
    w = np.zeros(len(net.bus))
    if hasattr(net, "load") and len(net.load):
        for b, p in zip(net.load["bus"].to_numpy(), net.load["p_mw"].to_numpy(float)):
            if b in idx:
                w[idx[b]] += max(p, 0.0)
    return w


def _hop_distances(src, adj, n):
    import collections

    d = np.full(n, np.inf)
    d[src] = 0.0
    q = _collections.deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if np.isinf(d[v]):
                d[v] = d[u] + 1.0
                q.append(v)
    return d


def _zone_components(lab, adj, z):
    """Connected-component count of zone z, for the connectivity constraint."""
    import collections

    members = set(np.flatnonzero(lab == z).tolist())
    seen, comps = set(), 0
    for s0 in members:
        if s0 in seen:
            continue
        comps += 1
        q = _collections.deque([s0]); seen.add(s0)
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v in members and v not in seen:
                    seen.add(v); q.append(v)
    return comps


import collections as _collections
import heapq

# Objective weights for balanced-v2 block refinement: balance first,
# spread second, cut last.
L1, L2, L3 = 100.0, 10.0, 1.0


def _v2_connected(nodes, adj):
    nodes = set(int(x) for x in nodes)
    if not nodes:
        return True
    s0 = next(iter(nodes)); seen = {s0}; q = _collections.deque([s0])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v in nodes and v not in seen:
                seen.add(v); q.append(v)
    return len(seen) == len(nodes)


def _v2_objective(lab, w, adj, n_zones, n_edges):
    L = np.array([w[lab == z].sum() for z in range(n_zones)])
    T = w.sum() / n_zones
    cut = sum(1 for u in adj for v in adj[u] if u < v and lab[u] != lab[v])
    cv = L.std(ddof=1) / L.mean() if L.mean() > 0 else np.inf
    return (L1 * np.max(np.abs(L / T - 1.0)) + L2 * cv + L3 * cut / max(n_edges, 1)), L, cut


def _v2_seed_load_aware(n, adj, w, n_zones, rng, alpha=0.5):
    cand = np.flatnonzero(w > 0)
    if len(cand) < n_zones:
        cand = np.arange(n)
    # first seed: sampled with probability proportional to load, so the choice
    # varies with the rng without ever landing in empty territory
    p = w[cand] / w[cand].sum()
    seeds = [int(rng.choice(cand, p=p))]
    dist = np.full(n, np.inf)
    for _ in range(n_zones - 1):
        dist = np.minimum(dist, _hop_distances(seeds[-1], adj, n))
        d = np.where(np.isfinite(dist), dist, 0.0)
        score = (w ** alpha) * d
        score[seeds] = -1.0
        score[w <= 0] = -1.0
        seeds.append(int(np.argmax(score)))
    return seeds


def _v2_grow(n, adj, w, n_zones, seeds):
    lab = np.full(n, -1, dtype=int)
    load = np.zeros(n_zones)
    heaps = [[] for _ in range(n_zones)]
    for z, s0 in enumerate(seeds):
        lab[s0] = z; load[z] += w[s0]
        for v in adj[s0]:
            heapq.heappush(heaps[z], (-w[v], v))
    remaining = int((lab < 0).sum())
    while remaining > 0:
        progressed = False
        for z in sorted(range(n_zones), key=lambda q: (load[q], q)):
            while heaps[z] and lab[heaps[z][0][1]] >= 0:
                heapq.heappop(heaps[z])
            if not heaps[z]:
                continue
            _, u = heapq.heappop(heaps[z])
            lab[u] = z; load[z] += w[u]; remaining -= 1; progressed = True
            for v in adj[u]:
                if lab[v] < 0:
                    heapq.heappush(heaps[z], (-w[v], v))
            break
        if not progressed:
            free = int(np.flatnonzero(lab < 0)[0])
            z = int(np.argmin(load))
            lab[free] = z; load[z] += w[free]; remaining -= 1
            for v in adj[free]:
                if lab[v] < 0:
                    heapq.heappush(heaps[z], (-w[v], v))
    return lab


def _v2_refine_blocks(lab, w, adj, n_zones, n_edges, max_block=60, rounds=400):
    J, L, _ = _v2_objective(lab, w, adj, n_zones, n_edges)
    for _ in range(rounds):
        best = None
        pairs = []
        for u in adj:
            for v in adj[u]:
                if lab[u] != lab[v]:
                    pairs.append((int(lab[u]), int(lab[v]), u))
        # prefer giving to the lightest zone and taking from the heaviest
        pairs.sort(key=lambda t: (L[t[1]] - L[t[0]]))
        seen_pair = set()
        for zf, zt, u0 in pairs:
            if L[zf] <= L[zt] or (zf, zt) in seen_pair:
                continue
            seen_pair.add((zf, zt))
            members = set(np.flatnonzero(lab == zf).tolist())
            # grow a connected block inside zf, starting at the boundary node
            block, q = [], _collections.deque([u0]); inb = {u0}
            while q and len(block) < max_block:
                x = q.popleft(); block.append(x)
                for y in adj[x]:
                    if y in members and y not in inb:
                        inb.add(y); q.append(y)
                if not _v2_connected(members - set(block), adj):
                    continue
                trial = lab.copy(); trial[block] = zt
                Jt, Lt, _ = _v2_objective(trial, w, adj, n_zones, n_edges)
                if Jt < J - 1e-9 and (best is None or Jt < best[0]):
                    best = (Jt, trial, Lt)
            if best is not None:
                break
        if best is None:
            break
        J, lab, L = best[0], best[1], best[2]
    return lab


def _balanced_labels(net, n_zones, rng, alpha=0.7):
    """Connected, load-balanced zones (balanced-v2).

    Region growing from load-aware seeds, then refinement that moves whole
    connected BLOCKS between adjacent zones.  The earlier version seeded on graph
    geometry alone and moved single buses; it spent four zones on one low-demand
    region and left zones holding 0.5% of the load.  v2 reaches 4.9-9.3% across
    all 14 zones on GBnetwork with every zone connected.

    Uses only topology and base-case load -- never a solved quantity -- so the
    partition is fixed before any power flow runs and cannot be tuned on the
    results it is meant to test.
    """
    n, idx, adj = _zone_graph(net)
    adj = {u: sorted(set(v)) for u, v in adj.items()}
    for u in range(n):
        adj.setdefault(u, [])
    w = _bus_load_weights(net, idx)
    n_edges = sum(len(v) for v in adj.values()) // 2
    seeds = _v2_seed_load_aware(n, adj, w, n_zones, rng, alpha)
    lab = _v2_grow(n, adj, w, n_zones, seeds)
    return _v2_refine_blocks(lab, w, adj, n_zones, n_edges)


_ZONE_CACHE = {}

# Set by the tilt block on every scenario; read by the writer.
_LAST_TILT = {"zone_log_tilt": None, "tilt_clr_rms": 0.0}


def _zone_labels(net, n_zones: int, rng, balanced: bool = False) -> np.ndarray:
    """Partition buses into n_zones connected regions of the electrical graph.

    Zones must be connected, not random labels: an independent multiplier on a
    scattered set of buses is spatially white and the power flow smooths it
    away, which would produce diversity in the setpoints without producing
    diversity in the voltage profile.  Region growing from n_zones seeds by
    breadth-first search over in-service branches gives contiguous zones whose
    loading can genuinely tilt one part of the grid against another.

    balanced=True uses the load-balanced builder above instead, which is what
    corpora use: BFS by hop count leaves zones whose demand differs by orders of
    magnitude, and a zone carrying no load cannot tilt anything.
    """
    import collections

    if balanced:
        # A property of the base grid, so computed once and cached; recomputing
        # per scenario would be wasteful and would invite drift.
        key = (n_zones, len(net.bus), len(getattr(net, "line", [])),
               round(float(net.load["p_mw"].sum())
                     if hasattr(net, "load") and len(net.load) else 0.0, 3))
        if key not in _ZONE_CACHE:
            _ZONE_CACHE[key] = _balanced_labels(net, n_zones, rng)
        return _ZONE_CACHE[key]

    n = len(net.bus)
    idx = {b: i for i, b in enumerate(net.bus.index)}
    adj = collections.defaultdict(list)
    for tbl, a, b in (("line", "from_bus", "to_bus"), ("trafo", "hv_bus", "lv_bus")):
        t = getattr(net, tbl, None)
        if t is None or not len(t):
            continue
        live = t["in_service"].to_numpy(bool) if "in_service" in t else np.ones(len(t), bool)
        for f, to, ok in zip(t[a].to_numpy(), t[b].to_numpy(), live):
            if ok and f in idx and to in idx:
                adj[idx[f]].append(idx[to]); adj[idx[to]].append(idx[f])

    lab = np.full(n, -1, dtype=int)
    seeds = rng.choice(n, size=min(n_zones, n), replace=False)
    frontier = _collections.deque()
    for z, s0 in enumerate(seeds):
        lab[s0] = z; frontier.append(s0)
    while frontier:                      # simultaneous BFS -> Voronoi-like regions
        u = frontier.popleft()
        for v in adj[u]:
            if lab[v] < 0:
                lab[v] = lab[u]; frontier.append(v)
    lab[lab < 0] = 0                     # islands, if any, join zone 0
    return lab


def _apply_line_outages(net, prob: float, rng) -> bool:
    """
    Randomly take lines out of service with per-line probability `prob`.
    After applying outages, checks connectivity via unsupplied_buses(); if any
    bus becomes unreachable from the slack, all outages are restored and the
    function returns False.  Returns True when at least one outage sticks.
    """
    if prob <= 0.0 or not hasattr(net, "line") or len(net.line) == 0:
        return False

    active_idx = net.line.index[net.line["in_service"].to_numpy(bool)]
    if len(active_idx) == 0:
        return False

    outage_mask = rng.random(len(active_idx)) < prob
    outaged = active_idx[outage_mask]
    if len(outaged) == 0:
        return False

    net.line.loc[outaged, "in_service"] = False

    try:
        from pandapower.topology import unsupplied_buses
        isolated = unsupplied_buses(net)
        if len(isolated) > 0:
            net.line.loc[outaged, "in_service"] = True
            return False
    except Exception:
        # If the topology check itself fails, be conservative and restore
        net.line.loc[outaged, "in_service"] = True
        return False

    return True


def _has_conflicting_voltage_controllers(net) -> bool:
    """
    Return True if multiple voltage-controlling elements share a bus.
    In such cases, random independent vm_pu jitter can make the case inconsistent.
    """
    ctrl_buses = []

    if hasattr(net, "ext_grid") and len(net.ext_grid) > 0 and "bus" in net.ext_grid.columns:
        ext = net.ext_grid
        if "in_service" in ext.columns:
            ext = ext[ext["in_service"]]
        ctrl_buses.extend(ext["bus"].to_list())

    if hasattr(net, "gen") and len(net.gen) > 0 and "bus" in net.gen.columns:
        gen = net.gen
        if "in_service" in gen.columns:
            gen = gen[gen["in_service"]]
        ctrl_buses.extend(gen["bus"].to_list())

    if not ctrl_buses:
        return False

    buses = np.asarray(ctrl_buses, dtype=np.int64)
    return np.unique(buses).size != buses.size


def _build_net_from_source(case_fn: CaseSource, case_kwargs: Optional[Dict[str, Any]] = None):
    """
    Generalized source resolver.

    Supported inputs:
      1) pandapower test case name, e.g. "case14"
      2) callable returning a net
      3) already-built pandapower net
      4) single CGMES zip path
      5) directory containing CGMES zip files
      6) list/tuple of CGMES zip files
      7) dict:
           {
             "cgmes_files": [...],              # required
             "case_name": "my_case",            # optional
             "converter_kwargs": {...},         # optional
           }
    """
    import pandapower.networks as pn

    case_kwargs = case_kwargs or {}

    # A) already-built pandapower net
    if _looks_like_pandapower_net(case_fn):
        net = copy.deepcopy(case_fn)
        case_name = getattr(net, "name", None) or "pandapower_net"
        return net, str(case_name)

    # B) CGMES config dict
    if isinstance(case_fn, dict):
        if "cgmes_files" not in case_fn:
            raise ValueError(
                "If case_fn is a dict, it must contain key 'cgmes_files'."
            )

        cgmes_files = case_fn["cgmes_files"]
        if isinstance(cgmes_files, (str, os.PathLike)):
            files = _normalize_cgmes_files_from_path(str(cgmes_files))
        else:
            files = [str(f) for f in cgmes_files]

        converter_kwargs = case_fn.get("converter_kwargs", {})
        case_name = case_fn.get("case_name", _default_case_name_from_files(files))
        net = _cgmes_from_files(files, converter_kwargs=converter_kwargs)
        return net, case_name

    # C) list / tuple of CGMES files
    if isinstance(case_fn, (list, tuple)):
        files = [str(f) for f in case_fn]
        net = _cgmes_from_files(files)
        case_name = _default_case_name_from_files(files)
        return net, case_name

    # D) string: either filesystem path or built-in pandapower case name
    if isinstance(case_fn, str):
        expanded = os.path.abspath(os.path.expanduser(case_fn))
        if os.path.exists(expanded):
            files = _normalize_cgmes_files_from_path(expanded)
            net = _cgmes_from_files(files)
            case_name = _default_case_name_from_files(files)
            return net, case_name

        fn = getattr(pn, case_fn, None)
        if fn is None:
            raise ValueError(
                f"Unknown pandapower case name and not a valid path: {case_fn}"
            )
        net = fn(**case_kwargs)
        return net, case_fn

    # E) callable returning a net
    if callable(case_fn):
        net = case_fn(**case_kwargs)
        case_name = getattr(case_fn, "__name__", "pandapower_case")
        return net, case_name

    raise TypeError(f"Unsupported case_fn type: {type(case_fn)}")


def per_unit_to_SI(Y_pu: np.ndarray, ppc_int) -> np.ndarray:
    """Convert Ybus in pu to SI (Siemens) using per-bus Vbase."""
    from pandapower.pypower.idx_bus import BASE_KV

    bus = np.asarray(ppc_int["bus"], dtype=float)
    baseMVA = _get_baseMVA_from_ppc(ppc_int)

    vn_kv = bus[:, BASE_KV].astype(float)
    Vbase = vn_kv * 1e3
    S_base = baseMVA * 1e6

    return Y_pu.astype(np.complex128) * (S_base / np.outer(Vbase, Vbase))


def build_Y_stamped_from_ppc(ppc_int):
    """
    Rebuild Ybus (per-unit) from ppc_int['bus'] and ppc_int['branch'].

    Matches pandapower's extended PPC branch model when available:
      - BR_G
      - BR_R_ASYM
      - BR_X_ASYM
      - BR_B_ASYM
      - BR_G_ASYM
    """
    from pandapower.pypower.idx_bus import GS, BS
    from pandapower.pypower.idx_brch import (
        F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS
    )

    BR_G, BR_R_ASYM, BR_X_ASYM, BR_B_ASYM, BR_G_ASYM = _optional_branch_indices()

    bus = np.asarray(ppc_int["bus"], dtype=float)
    branch = np.asarray(ppc_int["branch"], dtype=float)
    baseMVA = _get_baseMVA_from_ppc(ppc_int)

    nb = bus.shape[0]
    nl = branch.shape[0]
    Ybus = np.zeros((nb, nb), dtype=np.complex128)

    stat = branch[:, BR_STATUS]

    # series admittance. Pandapower allows asymmetric to-side impedance:
    # Ysf = 1 / (R + jX), Yst = 1 / (R + R_ASYM + j(X + X_ASYM)).
    R = branch[:, BR_R]
    X = branch[:, BR_X]
    if BR_R_ASYM is not None and BR_R_ASYM < branch.shape[1]:
        R_asym = branch[:, BR_R_ASYM]
    else:
        R_asym = np.zeros(nl, dtype=float)
    if BR_X_ASYM is not None and BR_X_ASYM < branch.shape[1]:
        X_asym = branch[:, BR_X_ASYM]
    else:
        X_asym = np.zeros(nl, dtype=float)

    Zf = R + 1j * X
    Zt = (R + R_asym) + 1j * (X + X_asym)

    Ysf = np.zeros(nl, dtype=np.complex128)
    nz_f = (stat != 0) & (np.abs(Zf) > 1e-18)
    Ysf[nz_f] = stat[nz_f] / Zf[nz_f]

    Yst = np.zeros(nl, dtype=np.complex128)
    nz_t = (stat != 0) & (np.abs(Zt) > 1e-18)
    Yst[nz_t] = stat[nz_t] / Zt[nz_t]

    # tap + phase shift
    tap = np.ones(nl, dtype=np.complex128)
    tap_raw = branch[:, TAP]
    nonunity = tap_raw != 0.0
    tap[nonunity] = tap_raw[nonunity]
    shift_deg = branch[:, SHIFT]
    tap *= np.exp(1j * np.pi / 180.0 * shift_deg)

    # branch shunts
    B = stat * branch[:, BR_B]

    if BR_G is not None and BR_G < branch.shape[1]:
        G = stat * branch[:, BR_G]
    else:
        G = np.zeros(nl, dtype=float)

    if BR_B_ASYM is not None and BR_B_ASYM < branch.shape[1]:
        B_asym = stat * branch[:, BR_B_ASYM]
    else:
        B_asym = np.zeros(nl, dtype=float)

    if BR_G_ASYM is not None and BR_G_ASYM < branch.shape[1]:
        G_asym = stat * branch[:, BR_G_ASYM]
    else:
        G_asym = np.zeros(nl, dtype=float)

    # end shunts in pu before half-split
    Bcf = (G + 1j * B)
    Bct = ((G + G_asym) + 1j * (B + B_asym))

    # branch stamps in pu
    Yff = (Ysf + Bcf / 2.0) / (tap * np.conj(tap))
    Ytt = Yst + Bct / 2.0
    Yft = -Ysf / np.conj(tap)
    Ytf = -Yst / tap

    f_bus = branch[:, F_BUS].astype(int)
    t_bus = branch[:, T_BUS].astype(int)

    for k in range(nl):
        if stat[k] == 0:
            continue
        if (
            (abs(Ysf[k]) < 1e-18)
            and (abs(Yst[k]) < 1e-18)
            and (abs(Bcf[k]) < 1e-18)
            and (abs(Bct[k]) < 1e-18)
        ):
            continue

        f = f_bus[k]
        t = t_bus[k]

        Ybus[f, f] += Yff[k]
        Ybus[t, t] += Ytt[k]
        Ybus[f, t] += Yft[k]
        Ybus[t, f] += Ytf[k]

    # bus shunts
    Ysh_pu = (bus[:, GS] + 1j * bus[:, BS]) / baseMVA
    Ybus[np.arange(nb), np.arange(nb)] += Ysh_pu

    return Ybus


def _build_manual_flat_start(
    N: int,
    bus_typ: np.ndarray,
    gen_ppc: np.ndarray,
    Vbase: np.ndarray,
    rand_u_start: bool,
    angle_jitter_deg: float,
    mag_jitter_pq: float,
    rng,
) -> np.ndarray:
    """
    Manual flat-start logic:
      - 1.0 pu magnitudes on all buses
      - overwrite PV/slack magnitudes from compiled PPC VG
      - 0 angles, unless rand_u_start=True for non-slack buses
    """
    from pandapower.pypower.idx_gen import GEN_BUS, VG, GEN_STATUS

    mag = Vbase.copy()
    ang = np.zeros(N, dtype=np.float64)

    if gen_ppc.size > 0 and gen_ppc.shape[1] > max(GEN_BUS, VG, GEN_STATUS):
        on = gen_ppc[:, GEN_STATUS] > 0
        gbus = gen_ppc[on, GEN_BUS].astype(int)
        vg = gen_ppc[on, VG].astype(float)
        for b, vm in zip(gbus, vg):
            if 0 <= b < N and np.isfinite(vm) and vm > 0:
                mag[b] = vm * Vbase[b]

    if rand_u_start:
        pq_mask = (bus_typ == 3)
        if pq_mask.any() and mag_jitter_pq > 0:
            mag[pq_mask] *= rng.uniform(
                1.0 - mag_jitter_pq,
                1.0 + mag_jitter_pq,
                size=int(pq_mask.sum())
            )

        non_slack = (bus_typ != 1)
        if non_slack.any() and angle_jitter_deg > 0:
            ang[non_slack] = rng.uniform(
                -angle_jitter_deg,
                angle_jitter_deg,
                size=int(non_slack.sum())
            ) * np.pi / 180.0

    return mag * np.exp(1j * ang)


def _build_dc_compile_start(
    net,
    Vbase: np.ndarray,
    bus_typ: Optional[np.ndarray] = None,
    rand_u_start: bool = False,
    angle_jitter_deg: float = 0.0,
    mag_jitter_pq: float = 0.0,
    rng=None,
) -> np.ndarray:
    """
    Build start voltage from a DC compile on a copy of the same net.
    Returns SI voltage vector aligned to the internal PPC order.

    This does NOT rely on ppc_int['V0'], because some pandapower versions
    do not store V0 after rundcpp().

    Optional start-noise (applied on top of the DC solution):
      rand_u_start    : enable angle and magnitude noise
      angle_jitter_deg: uniform angle noise on non-slack buses  [degrees]
      mag_jitter_pq   : uniform magnitude noise on PQ buses     [fraction]
      bus_typ         : bus type array (1=slack, 2=PV, 3=PQ) — required
                        for noise to be applied
      rng             : numpy Generator; if None, uses np.random.default_rng()
    """
    import pandapower as pp
    from pandapower.pypower.idx_bus import VA
    from pandapower.pypower.idx_gen import GEN_BUS, VG, GEN_STATUS

    net_dc = copy.deepcopy(net)
    pp.rundcpp(net_dc)

    if not hasattr(net_dc, "_ppc") or "internal" not in net_dc._ppc:
        raise RuntimeError("Could not build internal PPC for dc_compile start")

    ppc_dc = net_dc._ppc["internal"]

    # Some cases / versions do not expose baseMVA in internal after rundcpp,
    # but bus/gen are usually still available. We do not require baseMVA here.
    if "bus" not in ppc_dc:
        raise RuntimeError("ppc internal bus table not available after rundcpp()")

    bus_dc = np.asarray(ppc_dc["bus"], dtype=float)
    gen_dc = np.asarray(ppc_dc.get("gen", np.empty((0, 0))), dtype=float)
    if gen_dc.ndim == 1 and gen_dc.size == 0:
        gen_dc = gen_dc.reshape(0, 0)

    N = Vbase.shape[0]
    if bus_dc.shape[0] != N:
        raise RuntimeError(
            f"dc_compile bus length mismatch: len(bus_dc)={bus_dc.shape[0]} vs len(Vbase)={N}"
        )

    # DC angles from bus table (degrees -> radians)
    ang = np.deg2rad(bus_dc[:, VA].astype(float))

    # Default magnitudes = 1.0 pu, overwrite PV/slack buses from VG where available
    mag_pu = np.ones(N, dtype=np.float64)
    if gen_dc.size > 0 and gen_dc.shape[1] > max(GEN_BUS, VG, GEN_STATUS):
        on = gen_dc[:, GEN_STATUS] > 0
        gbus = gen_dc[on, GEN_BUS].astype(int)
        vg = gen_dc[on, VG].astype(float)
        for b, vm in zip(gbus, vg):
            if 0 <= b < N and np.isfinite(vm) and vm > 0:
                mag_pu[b] = vm

    # ----------------------------------------------------------------
    # Optional start-point noise on top of the DC solution
    # ----------------------------------------------------------------
    if rand_u_start and bus_typ is not None:
        if rng is None:
            rng = np.random.default_rng()

        bus_typ_arr = np.asarray(bus_typ)

        # Angle noise: all non-slack buses (PV + PQ)
        if angle_jitter_deg > 0:
            non_slack = (bus_typ_arr != 1)
            if non_slack.any():
                ang[non_slack] += rng.uniform(
                    -angle_jitter_deg,
                    angle_jitter_deg,
                    size=int(non_slack.sum()),
                ) * np.pi / 180.0

        # Magnitude noise: PQ buses only
        if mag_jitter_pq > 0:
            pq_mask = (bus_typ_arr == 3)
            if pq_mask.any():
                mag_pu[pq_mask] *= rng.uniform(
                    1.0 - mag_jitter_pq,
                    1.0 + mag_jitter_pq,
                    size=int(pq_mask.sum()),
                )

    return (mag_pu * np.exp(1j * ang)) * Vbase


# ============================================================
# Generic generator: one metadata row per PPC branch row
# ============================================================

def _empty_decision_space(N: int):
    """Placeholder decision space used when the OPF does not converge."""
    nan = lambda: np.full(N, np.nan, dtype=np.float64)  # noqa: E731
    return (
        nan(), nan(),                      # p_disp_bus, q_disp_bus
        nan(), nan(), nan(), nan(),        # p_min/p_max/q_min/q_max
        np.zeros(N, dtype=np.float64),     # cost_c2
        np.zeros(N, dtype=np.float64),     # cost_c1
        np.zeros(N, dtype=np.float64),     # cost_c0
        np.zeros(N, dtype=np.int8),        # is_controllable
    )


def _bus_decision_space(net, N: int):
    """Aggregate the OPF decision space onto buses, in PPC bus order.

    Covers BOTH net.gen and net.ext_grid: the slack is a dispatchable unit with
    its own cost row and its own limits, and omitting it leaves a large hole in
    the dispatch (case14's slack carries ~194 MW). Several units may sit on one
    bus, so powers and limits are summed per bus and costs are taken from the
    unit with the largest capacity on that bus.

    Bus alignment follows the same assumption as the caller: res_* frames are in
    pandapower bus order, which coincides with the PPC order when the sizes
    agree. Returns all-NaN limits for buses with no dispatchable unit.
    """
    p_disp = np.zeros(N, dtype=np.float64)
    q_disp = np.zeros(N, dtype=np.float64)
    p_min = np.full(N, np.nan, dtype=np.float64)
    p_max = np.full(N, np.nan, dtype=np.float64)
    q_min = np.full(N, np.nan, dtype=np.float64)
    q_max = np.full(N, np.nan, dtype=np.float64)
    c2 = np.zeros(N, dtype=np.float64)
    c1 = np.zeros(N, dtype=np.float64)
    c0 = np.zeros(N, dtype=np.float64)
    ctrl = np.zeros(N, dtype=np.int8)
    cap_seen = np.zeros(N, dtype=np.float64)

    bus_index = net.bus.index
    poly = getattr(net, "poly_cost", None)

    for et, res_name in (("gen", "res_gen"), ("ext_grid", "res_ext_grid")):
        frame = getattr(net, et, None)
        res = getattr(net, res_name, None)
        if frame is None or not len(frame) or res is None or len(res) != len(frame):
            continue

        pos = bus_index.get_indexer(frame["bus"].to_numpy())
        for k in range(len(frame)):
            i = int(pos[k])
            if i < 0 or i >= N:
                # Bus not in the PPC slice (out of service / different order).
                continue
            ctrl[i] = 1
            p_disp[i] += float(res["p_mw"].iloc[k])
            if "q_mvar" in res.columns:
                q_disp[i] += float(res["q_mvar"].iloc[k])

            for col, arr in (("min_p_mw", p_min), ("max_p_mw", p_max),
                             ("min_q_mvar", q_min), ("max_q_mvar", q_max)):
                if col in frame.columns:
                    val = float(frame[col].iloc[k])
                    arr[i] = val if np.isnan(arr[i]) else arr[i] + val

            # Cost curve of the largest unit on the bus.
            if poly is not None and len(poly):
                cap = abs(float(frame["max_p_mw"].iloc[k])) if "max_p_mw" in frame.columns else 0.0
                if cap >= cap_seen[i]:
                    row = poly[(poly["et"] == et) & (poly["element"] == frame.index[k])]
                    if len(row):
                        cap_seen[i] = cap
                        c2[i] = float(row["cp2_eur_per_mw2"].iloc[0])
                        c1[i] = float(row["cp1_eur_per_mw"].iloc[0])
                        c0[i] = float(row["cp0_eur"].iloc[0])

    return p_disp, q_disp, p_min, p_max, q_min, q_max, c2, c1, c0, ctrl


def _solve_opf(
    net,
    *,
    N: int,
    Vbase,
    Y_matrix,
    vm_limits=None,
    line_max_loading=None,
    use_dc: bool = False,
):
    """Solve the OPF for an already-perturbed net and extract labels.

    Returns a tuple appended to the standard case_generation_pandapower result:
        (u_opf_si, S_opf_si, lam_p, lam_q, gen_p_opt, gen_q_opt,
         opf_cost, opf_converged)

    Bus voltage limits and branch loading limits are applied here rather than
    baked into the case, because several PPC benchmarks (case300 in particular)
    ship with limits that make the base case OPF-infeasible.
    """
    # This module imports pandapower function-locally throughout, never at
    # module scope; keep that convention here.
    import pandapower as pp
    from pandapower.optimal_powerflow import OPFNotConverged

    u_opf_si = np.zeros(N, dtype=np.complex128)
    S_opf_si = np.zeros(N, dtype=np.complex128)
    (
        p_disp_bus, q_disp_bus,
        p_min_bus, p_max_bus, q_min_bus, q_max_bus,
        cost_c2, cost_c1, cost_c0, is_controllable,
    ) = _empty_decision_space(N)
    lam_p = np.full(N, np.nan, dtype=np.float64)
    lam_q = np.full(N, np.nan, dtype=np.float64)
    n_gen = int(len(net.gen)) if hasattr(net, "gen") else 0
    gen_p_opt = np.full(n_gen, np.nan, dtype=np.float64)
    gen_q_opt = np.full(n_gen, np.nan, dtype=np.float64)
    opf_cost = float("nan")
    opf_converged = False

    if vm_limits is not None:
        lo, hi = float(vm_limits[0]), float(vm_limits[1])
        net.bus["min_vm_pu"] = lo
        net.bus["max_vm_pu"] = hi

    if line_max_loading is not None:
        lim = float(line_max_loading)
        if hasattr(net, "line") and len(net.line):
            net.line["max_loading_percent"] = lim
        if hasattr(net, "trafo") and len(net.trafo):
            net.trafo["max_loading_percent"] = lim

    try:
        if use_dc:
            pp.rundcopp(net)
        else:
            pp.runopp(net, calculate_voltage_angles=True)

        res_bus = net.res_bus
        vm = np.asarray(res_bus["vm_pu"], dtype=np.float64)
        va = np.deg2rad(np.asarray(res_bus["va_degree"], dtype=np.float64))
        # res_bus is in pandapower bus order; the PPC internal order used for
        # Y_matrix may differ, so only accept it when the sizes line up.
        if vm.shape[0] == N:
            u_opf_si = (vm * np.exp(1j * va)) * Vbase
            S_opf_si = u_opf_si * np.conj(
                np.asarray(Y_matrix, dtype=np.complex128) @ u_opf_si
            )
            if "lam_p" in res_bus.columns:
                lam_p = np.asarray(res_bus["lam_p"], dtype=np.float64)
            if "lam_q" in res_bus.columns:
                lam_q = np.asarray(res_bus["lam_q"], dtype=np.float64)

        if n_gen and hasattr(net, "res_gen") and len(net.res_gen) == n_gen:
            gen_p_opt = np.asarray(net.res_gen["p_mw"], dtype=np.float64)
            if "q_mvar" in net.res_gen.columns:
                gen_q_opt = np.asarray(net.res_gen["q_mvar"], dtype=np.float64)

        # Bus-aligned decision space. res_gen alone omits the slack/ext_grid,
        # which carries a large share of the dispatch (e.g. 194 MW of case14's
        # total) and has its own cost row in poly_cost. Emitting per-bus vectors
        # here means downstream code never has to re-derive the dispatch by
        # differencing injections, nor guess which element owns which limits.
        (
            p_disp_bus, q_disp_bus,
            p_min_bus, p_max_bus, q_min_bus, q_max_bus,
            cost_c2, cost_c1, cost_c0, is_controllable,
        ) = _bus_decision_space(net, N)

        opf_cost = float(getattr(net, "res_cost", float("nan")))
        opf_converged = bool(np.isfinite(opf_cost))
    except OPFNotConverged:
        # Genuine non-convergence: expected for a fraction of samples, and the
        # caller drops these rows. Stays quiet so logs are not flooded.
        opf_converged = False
    except Exception as exc:
        # Anything else is a bug (bad limits, missing cost data, programming
        # error). Never let it masquerade as a convergence failure: that turns
        # a broken run into a plausible-looking 0% convergence statistic.
        opf_converged = False
        warnings.warn(
            f"OPF raised a non-convergence error: {type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    return (
        u_opf_si, S_opf_si, lam_p, lam_q,
        gen_p_opt, gen_q_opt, opf_cost, opf_converged,
        p_disp_bus, q_disp_bus,
        p_min_bus, p_max_bus, q_min_bus, q_max_bus,
        cost_c2, cost_c1, cost_c0, is_controllable,
    )


def case_generation_pandapower(
    case_fn: CaseSource,
    case_kwargs: Optional[Dict[str, Any]] = None,
    *,
    ybus_mode: str = "ppcY",   # "ppcY" or "stamped"
    cgmes_model_a_cleanup: bool = False,
    base_sn_mva: Optional[float] = None,  # override net.sn_mva (per-unit base)
    seed=None,
    jitter_load: float = 0.0,
    jitter_load_q: float = 0.0,
    jitter_gen: float = 0.0,
    pv_vset_range=None,
    rand_u_start: bool = False,
    angle_jitter_deg: float = 5.0,
    mag_jitter_pq: float = 0.02,
    trafo_pfe_kw: Optional[float] = None,
    trafo_i0_percent: Optional[float] = None,
    force_branch_shunt_pu: Optional[Dict[str, float]] = None,
    start_mode: str = "auto",  # "auto", "manual_flat", "ppc_v0", "dc_compile"
    # Operating-point diversity
    load_scale_range=None,        # e.g. (0.7, 1.3) — global correlated load scale
    scale_gen_with_load: bool = True,  # apply same global scale to PV gen p_mw
    # Static generators (net.sgen) are PQ injections: wind, PV plants, CHP.
    # They were historically left untouched by every perturbation, which is
    # harmless where sgen is a rounding error but decisive where it is not:
    # SimBench has net.gen EMPTY and 121 sgen carrying 3,716 MW, so with this
    # off a +/-40% load sweep moves total injection by only 2.2% and all
    # 36,000 scenarios are near-duplicates. Default False so the existing
    # 31-file corpus stays exactly reproducible; opt in for new runs.
    perturb_sgen: bool = False,
    # Recompute the PV-bus reactive entry of s_multi from u_start instead of
    # leaving whatever the PPC compile produced. See the block near the end of
    # this function for why the two differ. Default False to keep the existing
    # corpus reproducible.
    pv_q_from_vstart: bool = False,
    # Spatial load-pattern diversity.  The global scale above moves every load
    # by the SAME factor, so the spatial shape of the loading never changes and
    # the voltage profiles a grid can produce stay very low-dimensional: on
    # GBnetwork, 8 principal directions capture 98.4% of the |V| variation over
    # 12,339 scenarios and 2224 buses.  That is a property of this generator,
    # not of the grid, and it flatters any method that exploits the low
    # dimension -- manifold projection above all.  These two knobs vary the
    # shape as well as the level, so a held-out set can be built that the
    # training-fitted basis has never seen.  Both default off: the existing
    # 31-file corpus must stay reproducible.
    #
    #   load_zone_sigma  : each zone gets an independent lognormal multiplier
    #                      on top of the global scale (0 disables)
    #   load_zone_count  : number of zones; buses are partitioned by a
    #                      k-way split of the electrical graph, so a zone is a
    #                      connected region rather than an arbitrary label
    load_zone_sigma: float = 0.0,
    load_zone_count: int = 0,
    load_zone_correlation: str = "none",   # "none"|"empirical" zone-to-zone structure
    load_zone_structure_seed: int = 0,     # fixes the partition+basis across a corpus
    zone_tilt_targets: str = "load",       # "load" | "load_and_gen"
    zone_tilt_normalization: str = "load_weighted",   # "load_weighted" | "arithmetic"
    zone_tilt_amplitude_mode: str = "sigma",          # "sigma" | "clr_rms"
    zone_tilt_orientation_seed: int = None,  # covariance orientation/SD shuffle only
    line_outage_prob: float = 0.0,     # per-line probability of N-1 outage (legacy bernoulli)
    # Selectable topology / admittance perturbation (see perturbation_options.py)
    contingency_mode: str = "none",    # "none"|"bernoulli"|"ratio"|"poisson"
    contingency_elements=("line",),    # subset of {"line","gen"}
    contingency_ratio_weights=(0.55, 0.27, 0.18),  # P(N-0,N-1,N-2) for "ratio"
    contingency_poisson_q: float = 5e-4,           # per-branch unavailability for "poisson"
    contingency_k_cap: int = 2,        # hard cap on simultaneous outages
    admittance_sigma: float = 0.0,     # R/X jitter magnitude (0 disables)
    return_pp_solution: bool = False,
    # --- Optimal power flow -------------------------------------------------
    # When return_opf_solution is True the perturbed net is solved with
    # pp.runopp() instead of (or in addition to) a plain power flow, and the
    # OPF solution is appended to the returned tuple. Defaults are chosen so
    # that leaving these alone reproduces the power-flow behaviour exactly.
    return_opf_solution: bool = False,
    opf_vm_limits=None,                # (min_vm_pu, max_vm_pu) applied to every bus
    opf_line_max_loading=None,         # percent, applied to lines and trafos
    opf_dc: bool = False,              # True -> rundcopp instead of runopp
):
    """
    Generic pandapower/CGMES case generator with ONE metadata row per PPC branch row.

    Important:
      - bus_typ is derived from compiled PPC bus types
      - s_multi is derived from compiled PPC bus/gen tables
      - u_start is built in PPC order using the selected start_mode

    case_fn may be:
      - built-in pandapower case name, e.g. "case14"
      - callable returning a pandapower net
      - an already-built pandapower net
      - a CGMES zip path
      - a folder containing CGMES zip files
      - a list/tuple of CGMES zip files
      - a dict config: {"cgmes_files": [...], "case_name": ..., "converter_kwargs": ...}

    start_mode:
      - "auto"        : use ppc_int["V0"] if available, else manual flat
      - "manual_flat" : always use manual flat start
      - "ppc_v0"      : always use ppc_int["V0"]
      - "dc_compile"  : run a DC compile on a copy of the same net and build start
                         from DC bus angles + generator VG magnitudes

    Returns:
      (
        gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,

        Branch_f_bus, Branch_t_bus, Branch_status,
        Branch_tau, Branch_shift_deg,

        Branch_y_series_from, Branch_y_series_to, Branch_y_series_ft,
        Branch_y_shunt_from, Branch_y_shunt_to,

        Y_shunt_bus,

        Is_trafo, Branch_hv_is_f, Branch_n,

        Y_Lines, Y_C_Lines,
        U_base, S_base, vn_kv
      )

    Convention for bus_typ (your NR code):
      1 = slack
      2 = PV
      3 = PQ
    """
    import pandapower as pp
    from pandapower.powerflow import LoadflowNotConverged
    from pandapower.pypower.idx_bus import BASE_KV, GS, BS, BUS_TYPE
    from pandapower.pypower.idx_brch import (
        F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS
    )

    BR_G, BR_R_ASYM, BR_X_ASYM, BR_B_ASYM, BR_G_ASYM = _optional_branch_indices()

    valid_start_modes = {"auto", "manual_flat", "ppc_v0", "dc_compile"}
    if start_mode not in valid_start_modes:
        raise ValueError(f"Unknown start_mode={start_mode!r}. Use one of {sorted(valid_start_modes)}")

    case_kwargs = case_kwargs or {}
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # 1) Build network from generalized source
    # ------------------------------------------------------------
    net, case_name = _build_net_from_source(case_fn, case_kwargs=case_kwargs)

    if cgmes_model_a_cleanup and _looks_like_cgmes_source(case_fn):
        _apply_model_a_cgmes_cleanup(net)

    # Per-unit base override. Some CGMES exports carry a base (e.g. sn_mva=1)
    # that leaves injections at thousands of per-unit, which wrecks the
    # conditioning of pandapower's per-unit Newton-Raphson: it cannot reach
    # tolerance_mva=1e-8 at any iteration count. Rebasing to 100 MVA brings the
    # per-unit magnitudes into the range of the other benchmark cases and the
    # solve converges normally. Physics is unchanged; only the normalisation is.
    if base_sn_mva is not None and float(base_sn_mva) > 0:
        net.sn_mva = float(base_sn_mva)

    # optional transformer magnetizing params
    if hasattr(net, "trafo") and net.trafo is not None and len(net.trafo):
        if trafo_pfe_kw is not None:
            net.trafo.loc[:, "pfe_kw"] = float(trafo_pfe_kw)
        if trafo_i0_percent is not None:
            net.trafo.loc[:, "i0_percent"] = float(trafo_i0_percent)

    # ------------------------------------------------------------
    # Global correlated load scale  (moves the operating point)
    # Applied BEFORE per-bus jitter so the two effects are additive.
    # Gens are scaled by the same factor to keep generation roughly
    # balanced with load; the slack only has to absorb the small
    # per-bus jitter mismatch, not a large global imbalance.
    # ------------------------------------------------------------
    if load_scale_range is not None:
        lo, hi = load_scale_range
        global_scale = float(rng.uniform(lo, hi))
        global_scale = max(global_scale, 0.05)   # safety clamp — never negative/zero

        if hasattr(net, "load") and len(net.load):
            net.load["p_mw"]   = net.load["p_mw"].to_numpy(float)   * global_scale
            net.load["q_mvar"] = net.load["q_mvar"].to_numpy(float) * global_scale

        if scale_gen_with_load and hasattr(net, "gen") and len(net.gen):
            net.gen["p_mw"] = np.maximum(
                net.gen["p_mw"].to_numpy(float) * global_scale, 0.0
            )

        # Per-zone multiplier on top of the global scale.  Lognormal so the
        # factor is positive and its log is symmetric; mean-corrected so the
        # zones do not systematically raise or lower total load, which would
        # confound a shape change with a level change.
        # Entered even at sigma == 0 so the control consumes the same latent
        # draws; the multipliers are then all 1 and applying them is a no-op.
        if load_zone_sigma >= 0.0 and load_zone_count > 1 and hasattr(net, "load") and len(net.load):
            from perturbation_options import sample_zone_multipliers
            # "empirical" carries the measured zone-to-zone structure, which is a
            # standing property of a grid: the partition and the correlation
            # basis must then be identical in every scenario, so both come from
            # load_zone_structure_seed rather than from the per-scenario rng.
            # Redrawing either per scenario averages the structure away.
            # One partition per corpus, for BOTH correlation modes.  Until now
            # "none" re-grew the zones from the per-scenario rng while
            # "empirical" held them fixed, so an independent-versus-correlated
            # comparison differed in the partition as well as in the covariance.
            # Now the only difference is the covariance of the multipliers.
            lab = _zone_labels(net, load_zone_count,
                               np.random.default_rng(load_zone_structure_seed),
                               balanced=True)
            zmul = sample_zone_multipliers(
                load_zone_count, load_zone_sigma, rng,
                correlation=load_zone_correlation,
                # The partition seed and the covariance-orientation seed are
                # separate: which zone gets which marginal spread, and how the
                # correlation basis is oriented, are arbitrary (our zones are not
                # real GSP groups), so they must be varied to test robustness
                # WITHOUT disturbing the frozen partition.
                structure_seed=(load_zone_structure_seed
                                if zone_tilt_orientation_seed is None
                                else zone_tilt_orientation_seed),
                normalize=("none" if zone_tilt_normalization == "load_weighted"
                           else "arithmetic"),
                amplitude_mode=zone_tilt_amplitude_mode)
            bus_of = {b: i for i, b in enumerate(net.bus.index)}
            zone_of_load = np.array([lab[bus_of[b]]
                                     for b in net.load["bus"].to_numpy()])
            p_pre = net.load["p_mw"].to_numpy(float)
            q_pre = net.load["q_mvar"].to_numpy(float)

            if zone_tilt_normalization == "load_weighted":
                # Normalise on the load present at tilt time -- after the global
                # scale and the per-load jitter -- so that sum_z L_z m_z equals
                # sum_z L_z exactly and the tilt cannot move the system load
                # level.  The correction is one scalar applied to every zone, so
                # log m_i - log m_j = u_i - u_j: the covariance structure and the
                # centred-log sigma are untouched.
                Lz = np.array([p_pre[zone_of_load == z].sum()
                               for z in range(load_zone_count)])
                denom = float((Lz * zmul).sum())
                zmul = zmul * (float(Lz.sum()) / denom) if denom != 0.0 else zmul

            m = zmul[zone_of_load]
            # Same multiplier on P and Q, so each load keeps its power factor.
            # The weights are active power, so total Q is not exactly preserved;
            # that is measured rather than corrected.
            net.load["p_mw"]   = p_pre * m
            net.load["q_mvar"] = q_pre * m

            # Generation-time record of what the tilt actually was.  Recovering
            # it from the written loads is not reliable -- the per-load jitter is
            # applied afterwards and cannot be divided back out -- so the truth
            # is stored here instead.
            _u = np.log(np.clip(zmul, 1e-300, None))
            globals()["_LAST_TILT"] = {
                "zone_log_tilt": (_u - _u.mean()).astype(np.float64),
                "tilt_clr_rms": float(np.sqrt(((_u - _u.mean()) ** 2).mean())),
            }
            # The tilt is calibrated on GSP Group Take -- metered demand entering
            # a distribution system -- which says how regional DEMAND
            # redistributes and nothing about the output of transmission-
            # connected plant in that region, which follows dispatch rather than
            # local load.  Applying m_z to generators as well also partly
            # cancels the perturbation: where a zone's load and generation rise
            # together its net injection barely moves, and the cancellation is
            # uneven because generation share and load share do not match zone
            # by zone.  Generation still follows the GLOBAL load scale above --
            # that keeps the slack from absorbing a system-wide imbalance -- but
            # the spatial tilt is load-only by default.
            if (zone_tilt_targets == "load_and_gen"
                    and scale_gen_with_load and hasattr(net, "gen") and len(net.gen)):
                mg = np.array([zmul[lab[bus_of[b]]] for b in net.gen["bus"].to_numpy()])
                net.gen["p_mw"] = np.maximum(net.gen["p_mw"].to_numpy(float) * mg, 0.0)

        # Static generation follows the same global scale as net.gen. Q is
        # scaled with P so the unit's power factor is preserved; sgen q_mvar
        # is signed (it may absorb), so it is NOT clamped to be non-negative
        # the way p_mw is.
        if (perturb_sgen and scale_gen_with_load
                and hasattr(net, "sgen") and len(net.sgen)):
            net.sgen["p_mw"] = np.maximum(
                net.sgen["p_mw"].to_numpy(float) * global_scale, 0.0
            )
            net.sgen["q_mvar"] = net.sgen["q_mvar"].to_numpy(float) * global_scale

    # ------------------------------------------------------------
    # Topology + admittance perturbation (Y_bus VARIES — share_grid=False)
    # Applied BEFORE compilation so the compiled Y_matrix and Branch_status
    # reflect the actual in-service topology and branch impedances.
    #
    # Backward compatible: if contingency_mode == "none" but the legacy
    # line_outage_prob > 0 is set, fall back to the old per-line Bernoulli.
    # Otherwise use the selectable sampler (ratio / poisson / bernoulli).
    # ------------------------------------------------------------
    if contingency_mode == "none" and line_outage_prob > 0.0:
        _apply_line_outages(net, prob=line_outage_prob, rng=rng)
    elif contingency_mode != "none":
        from perturbation_options import sample_contingency
        sample_contingency(
            net, rng,
            mode=contingency_mode,
            elements=tuple(contingency_elements),
            bernoulli_p=line_outage_prob,
            ratio_weights=tuple(contingency_ratio_weights),
            poisson_q=contingency_poisson_q,
            k_cap=contingency_k_cap,
        )

    if admittance_sigma > 0.0:
        from perturbation_options import apply_admittance_jitter
        apply_admittance_jitter(net, rng, sigma=admittance_sigma)

    # optional load jitter — P and Q are drawn independently so power factor varies
    if jitter_load > 0 and hasattr(net, "load") and len(net.load):
        s_p = rng.normal(1.0, jitter_load, size=len(net.load))
        net.load["p_mw"] = net.load["p_mw"].to_numpy(float) * s_p
    if jitter_load_q > 0 and hasattr(net, "load") and len(net.load):
        s_q = rng.normal(1.0, jitter_load_q, size=len(net.load))
        net.load["q_mvar"] = net.load["q_mvar"].to_numpy(float) * s_q

    # optional generator active-power jitter
    if jitter_gen > 0 and hasattr(net, "gen") and len(net.gen):
        s = rng.normal(1.0, jitter_gen, size=len(net.gen))
        net.gen["p_mw"] = net.gen["p_mw"].to_numpy(float) * s

    # Static generators get the same per-unit dispatch jitter. P and Q are
    # drawn independently, matching how loads are treated, so the power factor
    # of the static generation varies rather than staying pinned.
    if perturb_sgen and jitter_gen > 0 and hasattr(net, "sgen") and len(net.sgen):
        net.sgen["p_mw"] = (net.sgen["p_mw"].to_numpy(float)
                            * rng.normal(1.0, jitter_gen, size=len(net.sgen)))
        net.sgen["q_mvar"] = (net.sgen["q_mvar"].to_numpy(float)
                              * rng.normal(1.0, jitter_gen, size=len(net.sgen)))

    # optional PV voltage-setpoint jitter
    # Disable it for cases with conflicting voltage controllers, because multiple
    # voltage-controlling elements on the same bus with different random vm_pu
    # values can make the compiled case internally inconsistent.
    DISABLE_PV_JITTER_CASES = {"GBnetwork", "GBreducednetwork"}
    disable_pv_jitter = (
        case_name in DISABLE_PV_JITTER_CASES
        or _has_conflicting_voltage_controllers(net)
    )

    if (
        pv_vset_range is not None
        and not disable_pv_jitter
        and hasattr(net, "gen")
        and len(net.gen)
    ):
        lo, hi = pv_vset_range
        net.gen["vm_pu"] = rng.uniform(lo, hi, size=len(net.gen))

    # ------------------------------------------------------------
    # 2) Compile PPC / Ybus
    # ------------------------------------------------------------
    runpp_exception = None
    try:
        pp.runpp(
            net,
            init="flat",
            calculate_voltage_angles=True,
            max_iteration=1,
            enforce_q_lims=False,
            tolerance_mva=1e9,
        )
    except LoadflowNotConverged as e:
        runpp_exception = e
    except Exception as e:
        runpp_exception = e

    if not hasattr(net, "_ppc") or "internal" not in net._ppc:
        raise RuntimeError(
            f"Could not build internal PPC for case {case_name}. "
            f"Original runpp exception: {repr(runpp_exception)}"
        )

    ppc_int = net._ppc["internal"]
    baseMVA = _get_baseMVA_from_ppc(ppc_int, fallback=net._ppc.get("baseMVA"))

    if "baseMVA" not in ppc_int:
        ppc_int = dict(ppc_int)
        ppc_int["baseMVA"] = baseMVA

    bus_ppc = np.asarray(ppc_int["bus"], dtype=float)
    branch_ppc = np.asarray(ppc_int["branch"], dtype=float)

    gen_ppc = np.asarray(ppc_int.get("gen", np.empty((0, 0))), dtype=float)
    if gen_ppc.ndim == 1 and gen_ppc.size == 0:
        gen_ppc = gen_ppc.reshape(0, 0)

    N = bus_ppc.shape[0]
    nl = branch_ppc.shape[0]

    vn_kv = bus_ppc[:, BASE_KV].astype(float)
    Vbase = vn_kv * 1e3
    S_base = baseMVA * 1e6

    # ------------------------------------------------------------
    # 3) Optional PPC branch-shunt forcing
    # ------------------------------------------------------------
    if force_branch_shunt_pu is not None:
        g_add = float(force_branch_shunt_pu.get("g", 0.0))
        b_add = float(force_branch_shunt_pu.get("b", 0.0))
        g_asym_add = float(force_branch_shunt_pu.get("g_asym", 0.0))
        b_asym_add = float(force_branch_shunt_pu.get("b_asym", 0.0))

        br2 = branch_ppc.copy()

        tap_raw = br2[:, TAP]
        shift_raw = br2[:, SHIFT]
        fb = br2[:, F_BUS].astype(int)
        tb = br2[:, T_BUS].astype(int)
        Vf0 = Vbase[fb]
        Vt0 = Vbase[tb]

        is_tr_like = (np.abs(tap_raw) > 0) & (np.abs(tap_raw - 1.0) > 1e-12)
        is_tr_like |= (np.abs(shift_raw) > 1e-12)
        is_tr_like |= (np.abs(Vf0 - Vt0) > 1e-6)

        br2[is_tr_like, BR_B] += b_add
        if BR_G is not None and BR_G < br2.shape[1]:
            br2[is_tr_like, BR_G] += g_add
        if BR_B_ASYM is not None and BR_B_ASYM < br2.shape[1]:
            br2[is_tr_like, BR_B_ASYM] += b_asym_add
        if BR_G_ASYM is not None and BR_G_ASYM < br2.shape[1]:
            br2[is_tr_like, BR_G_ASYM] += g_asym_add

        ppc_used = dict(ppc_int)
        ppc_used["branch"] = br2
        branch_used = br2
    else:
        ppc_used = ppc_int
        branch_used = branch_ppc

    # ------------------------------------------------------------
    # 4) Y_matrix in SI
    # ------------------------------------------------------------
    if (ybus_mode.lower() == "ppcy") and (force_branch_shunt_pu is None):
        Ypu = _dense_ybus_from_ppc_internal(ppc_int)
        Y_matrix = Ypu * (S_base / np.outer(Vbase, Vbase))
    else:
        Ypu = build_Y_stamped_from_ppc(ppc_used)
        Y_matrix = per_unit_to_SI(Ypu, ppc_used)

    # ------------------------------------------------------------
    # 5) Bus shunts in SI
    # ------------------------------------------------------------
    Ysh_pu = (bus_ppc[:, GS] + 1j * bus_ppc[:, BS]) / baseMVA
    Y_shunt_bus = Ysh_pu * (S_base / (Vbase * Vbase))

    # ------------------------------------------------------------
    # 6) bus_typ from compiled PPC bus table
    # ------------------------------------------------------------
    ppc_bus_type = bus_ppc[:, BUS_TYPE].astype(int)

    bus_typ = np.full(N, 3, dtype=np.int8)
    bus_typ[ppc_bus_type == 3] = 1
    bus_typ[ppc_bus_type == 2] = 2
    bus_typ[ppc_bus_type == 1] = 3

    # ------------------------------------------------------------
    # 7) s_multi from compiled PPC bus/gen tables
    # ------------------------------------------------------------
    try:
        Sbus_pu = makeSbus(baseMVA, bus_ppc, gen_ppc)
        s_multi = np.asarray(Sbus_pu).reshape(-1).astype(np.complex128) * S_base
    except Exception:
        from pandapower.pypower.idx_bus import PD, QD
        from pandapower.pypower.idx_gen import GEN_BUS, PG, GEN_STATUS

        s_multi = -(bus_ppc[:, PD] + 1j * bus_ppc[:, QD]) * 1e6

        if gen_ppc.size > 0 and gen_ppc.shape[1] > max(GEN_BUS, PG, GEN_STATUS):
            on = gen_ppc[:, GEN_STATUS] > 0
            gbus = gen_ppc[on, GEN_BUS].astype(int)
            pg = gen_ppc[on, PG].astype(float) * 1e6
            np.add.at(s_multi, gbus, pg)

    # ------------------------------------------------------------
    # 8) u_start in PPC order with start_mode
    # ------------------------------------------------------------
    if start_mode == "auto":
        if "V0" in ppc_int:
            u_start = np.asarray(ppc_int["V0"], dtype=np.complex128) * Vbase
        else:
            u_start = _build_manual_flat_start(
                N=N,
                bus_typ=bus_typ,
                gen_ppc=gen_ppc,
                Vbase=Vbase,
                rand_u_start=rand_u_start,
                angle_jitter_deg=angle_jitter_deg,
                mag_jitter_pq=mag_jitter_pq,
                rng=rng,
            )

    elif start_mode == "manual_flat":
        u_start = _build_manual_flat_start(
            N=N,
            bus_typ=bus_typ,
            gen_ppc=gen_ppc,
            Vbase=Vbase,
            rand_u_start=rand_u_start,
            angle_jitter_deg=angle_jitter_deg,
            mag_jitter_pq=mag_jitter_pq,
            rng=rng,
        )

    elif start_mode == "ppc_v0":
        if "V0" not in ppc_int:
            raise RuntimeError("start_mode='ppc_v0' requested but ppc_int['V0'] is not available")
        u_start = np.asarray(ppc_int["V0"], dtype=np.complex128) * Vbase

    elif start_mode == "dc_compile":
        u_start = _build_dc_compile_start(
            net=net,
            Vbase=Vbase,
            bus_typ=bus_typ,
            rand_u_start=rand_u_start,
            angle_jitter_deg=angle_jitter_deg,
            mag_jitter_pq=mag_jitter_pq,
            rng=rng,
        )

    # ------------------------------------------------------------
    # 8b) Optionally re-derive the PV-bus reactive entry from u_start
    #
    # What s_multi carries at a PV bus today is NOT a specified quantity. The
    # compile call above runs with tolerance_mva=1e9, so it converges
    # immediately and pfsoln back-computes each generator's QG from whatever
    # voltage the PPC holds at that moment (V0: flat magnitudes with the gen
    # setpoints applied, zero angles). makeSbus then folds that into s_multi.
    # Verified to machine precision on case118 / case300 / case1354pegase:
    #
    #     s_multi.imag[PV] == Im[V0 * conj(Y V0)]     (1e-16 .. 1e-13 pu)
    #
    # The catch is that V0 is not u_start. With start_mode="dc_compile",
    # u_start carries DC angles plus the start jitter, so the stored reactive
    # entry describes a DIFFERENT operating point from the voltage shipped
    # alongside it -- corr(s_multi.imag[PV], Q(u_start)) is 0.911 on case118
    # and only 0.568 on case300, and the gap varies per grid for no reason a
    # consumer of the data can see.
    #
    # With this flag the PV reactive entry is recomputed from u_start, so the
    # column means the same thing on every grid: "the reactive injection this
    # network implies at the initial voltage you were given". It uses only
    # u_start and Y, never u_newton, so it is not label leakage. Active power
    # at PV buses is untouched -- P there IS specified.
    #
    # Note this does nothing for grids with no PV buses. SimBench is exactly
    # that case (net.gen empty, 121 sgen, so 3 slack + 91 PQ and no PV at
    # all); its fix is perturb_sgen, not this.
    # ------------------------------------------------------------
    if pv_q_from_vstart:
        pv_mask = (bus_typ == 2)
        if np.any(pv_mask):
            S_of_ustart = u_start * np.conj(Y_matrix @ u_start)
            s_multi = s_multi.copy()
            s_multi[pv_mask] = (s_multi[pv_mask].real
                                + 1j * S_of_ustart[pv_mask].imag)

    # ------------------------------------------------------------
    # 9) One metadata row per PPC branch row
    # ------------------------------------------------------------
    Branch_f_bus = branch_used[:, F_BUS].astype(np.int32)
    Branch_t_bus = branch_used[:, T_BUS].astype(np.int32)
    Branch_status = branch_used[:, BR_STATUS].astype(np.int8)

    Branch_tau = np.ones(nl, dtype=np.float64)
    Branch_shift_deg = np.zeros(nl, dtype=np.float64)

    Branch_y_series_from = np.zeros(nl, dtype=np.complex128)
    Branch_y_series_to = np.zeros(nl, dtype=np.complex128)
    Branch_y_series_ft = np.zeros(nl, dtype=np.complex128)

    Branch_y_shunt_from = np.zeros(nl, dtype=np.complex128)
    Branch_y_shunt_to = np.zeros(nl, dtype=np.complex128)

    Is_trafo = np.zeros(nl, dtype=np.int8)
    Branch_hv_is_f = np.zeros(nl, dtype=np.int8)
    Branch_n = np.ones(nl, dtype=np.float64)

    Y_Lines = np.zeros(nl, dtype=np.complex128)
    Y_C_Lines = np.zeros(nl, dtype=np.float64)

    for k, row in enumerate(branch_used):
        fb = int(row[F_BUS])
        tb = int(row[T_BUS])

        stat = float(row[BR_STATUS])
        r = float(row[BR_R])
        x = float(row[BR_X])
        r_asym = _col(row, BR_R_ASYM, 0.0)
        x_asym = _col(row, BR_X_ASYM, 0.0)

        b = stat * float(row[BR_B])
        g = stat * _col(row, BR_G, 0.0)
        b_asym = stat * _col(row, BR_B_ASYM, 0.0)
        g_asym = stat * _col(row, BR_G_ASYM, 0.0)

        tau = float(row[TAP])
        if tau == 0.0:
            tau = 1.0
        shift = float(row[SHIFT])

        Branch_tau[k] = tau
        Branch_shift_deg[k] = shift

        Vf = Vbase[fb]
        Vt = Vbase[tb]
        Vh = max(Vf, Vt)
        Vl = min(Vf, Vt)

        Branch_hv_is_f[k] = 1 if Vf >= Vt else 0
        Branch_n[k] = (Vh / Vl) if Vl > 0 else 1.0

        is_tr = (abs(tau - 1.0) > 1e-12) or (abs(shift) > 1e-12) or (abs(Vf - Vt) > 1e-6)
        Is_trafo[k] = 1 if is_tr else 0

        z_from = complex(r, x)
        z_to = complex(r + r_asym, x + x_asym)
        Ysf_pu = 0j if (stat == 0.0 or abs(z_from) < 1e-12) else (stat / z_from)
        Yst_pu = 0j if (stat == 0.0 or abs(z_to) < 1e-12) else (stat / z_to)

        Branch_y_series_from[k] = Ysf_pu * (S_base / (Vf ** 2))
        Branch_y_series_to[k] = Yst_pu * (S_base / (Vt ** 2))
        Branch_y_series_ft[k] = Ysf_pu * (S_base / (Vf * Vt))

        Bcf_pu = (g + 1j * b)
        Bct_pu = ((g + g_asym) + 1j * (b + b_asym))
        Branch_y_shunt_from[k] = Bcf_pu * (S_base / (Vf ** 2))
        Branch_y_shunt_to[k] = Bct_pu * (S_base / (Vt ** 2))

        if not is_tr:
            Y_Lines[k] = Branch_y_series_ft[k]
            Y_C_Lines[k] = 0.5 * b * (S_base / (Vf ** 2))

    is_connected = bool(N > 0)
    U_base = float(Vbase[0])
    gridtype = f"{case_name}_pandapower_{'ppcY' if ybus_mode.lower() == 'ppcy' else 'stamped'}"

    result = (
        gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,

        Branch_f_bus, Branch_t_bus, Branch_status,
        Branch_tau, Branch_shift_deg,

        Branch_y_series_from, Branch_y_series_to, Branch_y_series_ft,
        Branch_y_shunt_from, Branch_y_shunt_to,

        Y_shunt_bus.astype(np.complex128),

        Is_trafo, Branch_hv_is_f, Branch_n,

        Y_Lines, Y_C_Lines,
        U_base, S_base, vn_kv.astype(np.float64),
    )

    if return_opf_solution:
        return result + _solve_opf(
            net,
            N=N,
            Vbase=Vbase,
            Y_matrix=Y_matrix,
            vm_limits=opf_vm_limits,
            line_max_loading=opf_line_max_loading,
            use_dc=opf_dc,
        )

    if not return_pp_solution:
        return result

    pp_converged = False
    u_pp_si = np.zeros(N, dtype=np.complex128)
    S_pp_si = np.zeros(N, dtype=np.complex128)

    try:
        pp.runpp(
            net,
            algorithm="nr",
            init="auto",
            calculate_voltage_angles=True,
            enforce_q_lims=False,
            max_iteration=30,
            tolerance_mva=1e-8,
        )
        ppci_label = net._ppc["internal"]
        Vpu_label = np.asarray(ppci_label["V"], dtype=np.complex128)
        if Vpu_label.shape[0] != N:
            raise RuntimeError(
                f"pandapower label PPC size changed from {N} to {Vpu_label.shape[0]}"
            )
        u_pp_si = Vpu_label * Vbase
        S_pp_si = u_pp_si * np.conj(np.asarray(Y_matrix, dtype=np.complex128) @ u_pp_si)
        pp_converged = bool(getattr(net, "converged", False))
    except Exception:
        pp_converged = False

    return result + (u_pp_si, S_pp_si, pp_converged)


def case_generation_pandapower_stamped(
    case_fn: CaseSource,
    case_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """Wrapper that always builds Y_matrix by stamping PPC branch+bus data."""
    return case_generation_pandapower(
        case_fn=case_fn,
        case_kwargs=case_kwargs,
        ybus_mode="stamped",
        **kwargs,
    )


# ============================================================
# Reconstruction: direct SI, one metadata row per PPC branch
# ============================================================

def reconstruct_Y_pandapower_branchrows_direct_SI(
    N: int,
    Branch_f_bus: np.ndarray,
    Branch_t_bus: np.ndarray,
    Branch_status: np.ndarray,
    Branch_tau: np.ndarray,
    Branch_shift_deg: np.ndarray,
    Branch_y_series_from: np.ndarray,
    Branch_y_series_to: np.ndarray,
    Branch_y_series_ft: np.ndarray,
    Branch_y_shunt_from: np.ndarray,
    Branch_y_shunt_to: np.ndarray,
    Y_shunt_bus: np.ndarray,
    Vbase_bus: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Reconstruct Ybus directly in SI, using ONE metadata row per PPC branch row.

    Direct SI stamps:
      Yff = (y_from + ysh_from/2) / |a|^2
      Ytt =  y_to   + ysh_to/2
      Yft = -y_ft / conj(a)
      Ytf = -y_tf / a

    where:
      a = tau * exp(j*theta)

    If Vbase_bus is supplied, asymmetric series impedances are represented exactly
    by deriving the SI cross-base terms from y_from and y_to. Without Vbase_bus,
    Ytf falls back to the legacy symmetric Branch_y_series_ft approximation.
    """
    N = int(N)
    nl = len(Branch_f_bus)

    Y_SI = np.zeros((N, N), dtype=np.complex128)
    Y_SI[np.diag_indices(N)] += Y_shunt_bus.astype(np.complex128)

    for k in range(nl):
        if int(Branch_status[k]) == 0:
            continue

        f = int(Branch_f_bus[k])
        t = int(Branch_t_bus[k])

        tau = float(Branch_tau[k])
        theta = np.deg2rad(float(Branch_shift_deg[k]))
        a = tau * np.exp(1j * theta)

        y_from = complex(Branch_y_series_from[k])
        y_to = complex(Branch_y_series_to[k])
        y_ft = complex(Branch_y_series_ft[k])
        y_tf = y_ft
        ysh_f = complex(Branch_y_shunt_from[k])
        ysh_t = complex(Branch_y_shunt_to[k])

        if Vbase_bus is not None:
            Vf = float(Vbase_bus[f])
            Vt = float(Vbase_bus[t])
            if Vf > 0.0 and Vt > 0.0:
                y_ft = y_from * (Vf / Vt)
                y_tf = y_to * (Vt / Vf)

        Yff = (y_from + ysh_f / 2.0) / (a * np.conj(a))
        Ytt = (y_to + ysh_t / 2.0)
        Yft = -y_ft / np.conj(a)
        Ytf = -y_tf / a

        Y_SI[f, f] += Yff
        Y_SI[t, t] += Ytt
        Y_SI[f, t] += Yft
        Y_SI[t, f] += Ytf

    return Y_SI


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    import pandapower.networks as pn

    CASES = [
        "case4gs",
        "case5",
        "case6ww",
        "case9",
        "case14",
        "case24_ieee_rts",
        "case30",
        "case_ieee30",
        "case33bw",
        "case39",
        "case57",
        "case89pegase",
        "case118",
        "case145",
        "case_illinois200",
        "case300",
        "case1354pegase",
        "case1888rte",
        "case2848rte",
        "case2869pegase",
        "case3120sp",
        "case6470rte",
        "case6495rte",
        "case6515rte",
        "case9241pegase",
        "GBnetwork",
        "GBreducednetwork",
        "iceland",
    ]

    FORCE_SHUNT_CASES = {
        "case4gs", "case5", "case6ww", "case9", "case30", "case33bw"
    }

    CASE_KWARGS: Dict[str, Dict[str, Any]] = {
        # "case1888rte": {"ref_bus_idx": 1246},
        # "case2848rte": {"ref_bus_idx": 271},
        # "case6470rte": {"ref_bus_idx": 5988},
        # "case6515rte": {"ref_bus_idx": 6171},
    }

    print("=== branch-row direct-SI reconstruction check ===")

    for name in CASES:
        case_kwargs = CASE_KWARGS.get(name, {})

        force = None
        if name in FORCE_SHUNT_CASES:
            force = {"g": 0.0, "b": 0.2, "g_asym": 0.0, "b_asym": 0.0}

        try:
            (
                gridtype,
                bus_typ,
                s_multi,
                u_start,
                Y_matrix,
                is_connected,

                Branch_f_bus,
                Branch_t_bus,
                Branch_status,
                Branch_tau,
                Branch_shift_deg,

                Branch_y_series_from,
                Branch_y_series_to,
                Branch_y_series_ft,
                Branch_y_shunt_from,
                Branch_y_shunt_to,

                Y_shunt_bus,

                Is_trafo,
                Branch_hv_is_f,
                Branch_n,

                Y_Lines,
                Y_C_Lines,
                U_base,
                S_base,
                vn_kv,
            ) = case_generation_pandapower(
                case_fn=name,
                case_kwargs=case_kwargs,
                ybus_mode="ppcY",
                seed=0,
                trafo_pfe_kw=50.0,
                trafo_i0_percent=2.0,
                force_branch_shunt_pu=force,
                start_mode="auto",
            )

            N = len(bus_typ)
            nl = len(Branch_f_bus)

            Y_rec = reconstruct_Y_pandapower_branchrows_direct_SI(
                N,
                Branch_f_bus,
                Branch_t_bus,
                Branch_status,
                Branch_tau,
                Branch_shift_deg,
                Branch_y_series_from,
                Branch_y_series_to,
                Branch_y_series_ft,
                Branch_y_shunt_from,
                Branch_y_shunt_to,
                Y_shunt_bus,
            )

            diff = np.max(np.abs(Y_rec - Y_matrix))
            n_tr = int(np.sum(Is_trafo))
            n_on = int(np.sum(Branch_status))

            print(
                f"{name:16s} | "
                f"gridtype={gridtype:30s} | "
                f"N={N:5d} nl={nl:6d} on={n_on:6d} trafo_like={n_tr:6d} | "
                f"forced_shunt={'yes' if force is not None else 'no ':3s} | "
                f"max|diff|={diff:.12g}"
            )

        except Exception as e:
            print(f"{name:16s} | FAILED | {repr(e)}")

    print("=== done ===")

    # ------------------------------------------------------------
    # Example: CGMES usage
    # ------------------------------------------------------------
    # cgmes_case = {
    #     "cgmes_files": [
    #         "/Users/changhunkim/PycharmProjects/PIGNN-Attn-LS/CGMES_to_PandaPower/IEEE14Bus.zip"
    #     ],
    #     "case_name": "IEEE14Bus_CGMES",
    #     "converter_kwargs": {},
    # }
    #
    # res = case_generation_pandapower(
    #     case_fn=cgmes_case,
    #     ybus_mode="ppcY",
    #     seed=0,
    #     start_mode="auto",
    # )

    # net = cim2pp.from_cim(file_list=cgmes_files)
    #
    # res = case_generation_pandapower(
    #     case_fn=net,
    #     ybus_mode="ppcY",
    #     seed=0,
    #     start_mode="auto",
    # )
