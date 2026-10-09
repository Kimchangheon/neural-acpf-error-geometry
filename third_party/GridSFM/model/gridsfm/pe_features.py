"""Operating-point and topology positional-encoding features."""
from __future__ import annotations

import hashlib
import math
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from .schema import (
    BUS_TYPE_IDX,
    BUS_VMIN_IDX,
    BUS_VMAX_IDX,
    GEN_VG_IDX,
    GEN_PMIN_IDX,
    GEN_PMAX_IDX,
    GEN_QMIN_IDX,
    GEN_QMAX_IDX,
    LOAD_PD_IDX,
    LOAD_QD_IDX,
    AC_LINE_R_IDX,
    AC_LINE_X_IDX,
    AC_LINE_RATE_A_IDX,
    AC_LINE_KEY,
    TR_R_IDX,
    TR_X_IDX,
    TR_RATE_A_IDX,
    TR_SHIFT_IDX,
    TRANSFORMER_KEY,
    REACTANCE_EPS,
    REACTANCE_FLOOR,
)


def hash_topology(
    data: HeteroData,
    *,
    include_bus_type: bool = False,
    include_gen_link: bool = False,
    prefix: bytes = b"",
) -> str:
    h = hashlib.sha1()
    if prefix:
        h.update(prefix)
    h.update(f"{int(data['bus'].x.size(0))}|".encode())
    if include_bus_type:
        bx = data['bus'].x
        if bx.size(1) > BUS_TYPE_IDX:
            h.update(b"bus_type|")
            h.update(bx[:, BUS_TYPE_IDX].detach().cpu().contiguous()
                      .numpy().astype('int64').tobytes())
    if include_gen_link:
        gen_key = ('generator', 'generator_link', 'bus')
        if gen_key in data.edge_types:
            h.update(b"gen_link|")
            h.update(data[gen_key].edge_index.detach().cpu().numpy().tobytes())
        else:
            h.update(b"gen_link:none|")
    for et in (AC_LINE_KEY, TRANSFORMER_KEY):
        if et in data.edge_types:
            ei = data[et].edge_index.cpu().numpy()
            h.update(f"{et[1]}|".encode())
            h.update(ei.tobytes())
            ea = data[et].get('edge_attr', None)
            if ea is not None and ea.numel() > 0:
                h.update(ea.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(f"{et[1]}:none|".encode())
    return h.hexdigest()


def _topology_fingerprint(data: HeteroData) -> str:
    return hash_topology(data, include_bus_type=True, include_gen_link=True)


class LaplacianFactorizationCache:
    """Per-topology Laplacian-factor cache (keyed by topology hash, LRU)."""

    def __init__(self, n_landmarks: int = 64, max_cache: int = 16):
        import threading
        from collections import OrderedDict
        self._cache: "OrderedDict[str, Dict]" = OrderedDict()
        self._max = int(max_cache)
        self.n_landmarks = int(n_landmarks)
        self._lock = threading.Lock()

    def get_or_build(self, data: HeteroData) -> Dict:
        key = _topology_fingerprint(data)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        state = _build_topology_state(data, self.n_landmarks)
        with self._lock:
            self._cache[key] = state
            self._cache.move_to_end(key)
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)
        return state


DEFAULT_LAPLACIAN_CACHE = LaplacianFactorizationCache()


def _build_topology_state(data: HeteroData, K: int) -> Dict:
    n_bus = int(data['bus'].x.size(0))

    edges_i, edges_j, edges_b, edges_phi = [], [], [], []
    edge_offsets = {'ac_line': (0, 0), 'transformer': (0, 0)}

    if AC_LINE_KEY in data.edge_types:
        ei = data[AC_LINE_KEY].edge_index.cpu().numpy()
        ea = data[AC_LINE_KEY].edge_attr.cpu().numpy().astype(np.float64)
        x = ea[:, AC_LINE_X_IDX]
        x = np.where(np.abs(x) < REACTANCE_EPS, REACTANCE_FLOOR, x)
        b = 1.0 / x
        start = len(edges_i)
        edges_i.append(ei[0]); edges_j.append(ei[1])
        edges_b.append(b); edges_phi.append(np.zeros(ei.shape[1], dtype=np.float64))
        edge_offsets['ac_line'] = (start, ei.shape[1])

    if TRANSFORMER_KEY in data.edge_types:
        ei = data[TRANSFORMER_KEY].edge_index.cpu().numpy()
        ea = data[TRANSFORMER_KEY].edge_attr.cpu().numpy().astype(np.float64)
        x = ea[:, TR_X_IDX]
        x = np.where(np.abs(x) < REACTANCE_EPS, REACTANCE_FLOOR, x)
        b = 1.0 / x
        phi = ea[:, TR_SHIFT_IDX]
        start = len(edges_i)
        edges_i.append(ei[0]); edges_j.append(ei[1])
        edges_b.append(b); edges_phi.append(phi)
        edge_offsets['transformer'] = (start, ei.shape[1])

    if not edges_i:
        return _empty_state(n_bus, K)

    all_i = np.concatenate(edges_i)
    all_j = np.concatenate(edges_j)
    all_b = np.concatenate(edges_b)
    all_phi = np.concatenate(edges_phi)

    rows = np.concatenate([all_i, all_j, all_i, all_j])
    cols = np.concatenate([all_j, all_i, all_i, all_j])
    vals = np.concatenate([-all_b, -all_b, all_b, all_b])
    L_full = sp.csr_matrix((vals, (rows, cols)), shape=(n_bus, n_bus)).tocsc()

    diag = np.array(L_full.diagonal())
    eps = max(1e-8, float(np.median(diag)) * 1e-6)
    L_full_reg = (L_full + eps * sp.eye(n_bus)).tocsc()
    L_full_solve = factorized(L_full_reg)

    bus_type = data['bus'].x[:, BUS_TYPE_IDX].cpu().numpy().astype(int)
    slack_buses = np.where(bus_type == 3)[0]

    gen_buses_set = set()
    if ('generator', 'generator_link', 'bus') in data.edge_types:
        g2b = data[('generator', 'generator_link', 'bus')].edge_index[1].cpu().numpy()
        gen_buses_set.update(g2b.tolist())

    deg_order = np.argsort(-diag)

    chosen = []
    seen = set()
    def _push(idx_arr):
        for i in idx_arr:
            i = int(i)
            if i not in seen:
                chosen.append(i); seen.add(i)
                if len(chosen) >= K:
                    return True
        return False
    if _push(slack_buses): pass
    elif _push(np.array(sorted(gen_buses_set), dtype=int)): pass
    elif _push(deg_order): pass
    landmarks = np.array(chosen[:K], dtype=int)
    if len(landmarks) < K:
        landmarks = np.concatenate([landmarks,
                                     np.full(K - len(landmarks), landmarks[0])])

    if len(slack_buses) > 0:
        slack_idx = int(slack_buses[0])
    else:
        slack_idx = 0
    mask = np.ones(n_bus, dtype=bool); mask[slack_idx] = False
    L_reduced = L_full[mask][:, mask].tocsc()
    try:
        L_reduced_solve = factorized(L_reduced)
    except RuntimeError as e:
        if "singular" in str(e).lower():
            lam = max(eps, 1e-6)
            L_reduced_reg = (L_reduced + lam * sp.eye(L_reduced.shape[0],
                                                      dtype=L_reduced.dtype,
                                                      format='csc')).tocsc()
            L_reduced_solve = factorized(L_reduced_reg)
        else:
            raise

    return {
        'n_bus':           n_bus,
        'edges_i':         all_i,
        'edges_j':         all_j,
        'edges_b':         all_b,
        'edges_phi':       all_phi,
        'edge_offsets':    edge_offsets,
        'L_full_solve':    L_full_solve,
        'L_reduced_solve': L_reduced_solve,
        'reduced_mask':    mask,
        'slack_idx':       slack_idx,
        'landmarks':       landmarks,
        'K':               K,
        'eps':             eps,
    }


def _empty_state(n_bus: int, K: int) -> Dict:
    return {
        'n_bus': n_bus, 'edges_i': np.zeros(0, dtype=int),
        'edges_j': np.zeros(0, dtype=int), 'edges_b': np.zeros(0),
        'edges_phi': np.zeros(0),
        'edge_offsets': {'ac_line': (0, 0), 'transformer': (0, 0)},
        'L_full_solve': None, 'L_reduced_solve': None,
        'reduced_mask': np.ones(n_bus, dtype=bool), 'slack_idx': 0,
        'landmarks': np.zeros(K, dtype=int), 'K': K, 'eps': 1e-8,
    }


def compute_er_landmark_moments(state: Dict) -> np.ndarray:
    K = state['K']
    n_bus = state['n_bus']
    if state['L_full_solve'] is None:
        return np.zeros((n_bus, 5), dtype=np.float32)

    landmarks = state['landmarks']
    X = np.zeros((n_bus, K), dtype=np.float64)
    for col, lm in enumerate(landmarks):
        e = np.zeros(n_bus, dtype=np.float64)
        e[lm] = 1.0
        X[:, col] = state['L_full_solve'](e)

    M = np.zeros((n_bus, 5), dtype=np.float32)
    M[:, 0] = X.min(axis=1)
    M[:, 1] = X.max(axis=1)
    M[:, 2] = X.std(axis=1)
    M[:, 3] = np.median(X, axis=1)
    M[:, 4] = X.mean(axis=1)
    return M


def compute_dc_features(
    data: HeteroData,
    state: Dict,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    n_bus = state['n_bus']
    if state['L_reduced_solve'] is None:
        return (np.zeros((n_bus, 3), dtype=np.float32),
                {'ac_line': np.zeros((0, 4), dtype=np.float32),
                 'transformer': np.zeros((0, 4), dtype=np.float32)})

    P_inj_raw = np.zeros(n_bus, dtype=np.float64)
    if ('generator', 'generator_link', 'bus') in data.edge_types:
        g2b = data[('generator', 'generator_link', 'bus')].edge_index[1].cpu().numpy()
        gx = data['generator'].x.cpu().numpy()
        Pmin = np.nan_to_num(gx[:, GEN_PMIN_IDX].astype(np.float64))
        Pmax = np.nan_to_num(gx[:, GEN_PMAX_IDX].astype(np.float64))
        Plo = np.minimum(Pmin, Pmax); Phi = np.maximum(Pmin, Pmax)
        Pmid = 0.5 * (Plo + Phi)
        np.add.at(P_inj_raw, g2b, Pmid)
    if ('load', 'load_link', 'bus') in data.edge_types:
        l2b = data[('load', 'load_link', 'bus')].edge_index[1].cpu().numpy()
        Pd = data['load'].x[:, LOAD_PD_IDX].cpu().numpy().astype(np.float64)
        np.add.at(P_inj_raw, l2b, -Pd)

    e_i = state['edges_i']; e_j = state['edges_j']
    e_b = state['edges_b']; e_phi = state['edges_phi']
    P_inj_solve = P_inj_raw.copy()
    if e_i.size > 0:
        contrib = e_b * e_phi
        np.add.at(P_inj_solve, e_i, contrib)
        np.add.at(P_inj_solve, e_j, -contrib)

    slack_idx = state['slack_idx']
    P_inj_solve[slack_idx] -= P_inj_solve.sum()

    mask = state['reduced_mask']
    P_red = P_inj_solve[mask]
    theta_red = state['L_reduced_solve'](P_red)
    theta_dc = np.zeros(n_bus, dtype=np.float64)
    theta_dc[mask] = theta_red

    dc_stress = np.zeros(n_bus, dtype=np.float64)
    deg = np.zeros(n_bus, dtype=np.float64)
    if e_i.size > 0:
        dth = theta_dc[e_i] - theta_dc[e_j] - e_phi
        np.add.at(dc_stress, e_i, np.abs(dth))
        np.add.at(dc_stress, e_j, np.abs(dth))
        np.add.at(deg, e_i, 1.0)
        np.add.at(deg, e_j, 1.0)
    dc_stress = dc_stress / np.maximum(deg, 1.0)

    bus_pe = np.stack([
        theta_dc,
        dc_stress,
        P_inj_raw,
    ], axis=1).astype(np.float32)

    branch_pe = {}
    for rel in ('ac_line', 'transformer'):
        start, count = state['edge_offsets'].get(rel, (0, 0))
        if count == 0:
            branch_pe[rel] = np.zeros((0, 4), dtype=np.float32)
            continue
        sl = slice(start, start + count)
        i_sl = e_i[sl]; j_sl = e_j[sl]; b_sl = e_b[sl]; phi_sl = e_phi[sl]
        dtheta = theta_dc[i_sl] - theta_dc[j_sl] - phi_sl
        P_dc = b_sl * dtheta

        if rel == 'ac_line':
            ea = data[AC_LINE_KEY].edge_attr.cpu().numpy()
            rate = ea[:, AC_LINE_RATE_A_IDX]
        else:
            ea = data[TRANSFORMER_KEY].edge_attr.cpu().numpy()
            rate = ea[:, TR_RATE_A_IDX]
        rate_safe = np.where(rate > 1e-9, rate, 1.0)
        loading = np.abs(P_dc) / rate_safe
        loading = np.minimum(loading, 2.0)
        sign = np.sign(P_dc).astype(np.float32)

        branch_pe[rel] = np.stack([
            dtheta.astype(np.float32),
            P_dc.astype(np.float32),
            loading.astype(np.float32),
            sign,
        ], axis=1)

    return bus_pe, branch_pe


_PE_ATTACHED_FLAG = "_gridsfm_pe_attached"


def attach_pe_features_(
    data: HeteroData,
    cache: Optional[LaplacianFactorizationCache] = None,
) -> HeteroData:
    if getattr(data, _PE_ATTACHED_FLAG, False):
        return data
    need_ac_pe = ('branch_ac' in data.node_types and
                  data['branch_ac'].x.size(0) > 0)
    need_tr_pe = ('branch_tr' in data.node_types and
                  data['branch_tr'].x.size(0) > 0)

    if cache is None:
        cache = DEFAULT_LAPLACIAN_CACHE
    state = cache.get_or_build(data)
    bus_dc, branch_dc = compute_dc_features(data, state)

    er_moments = compute_er_landmark_moments(state)
    bus_x = data['bus'].x
    pe_bus = torch.from_numpy(np.concatenate([er_moments, bus_dc], axis=1)
                              ).to(dtype=bus_x.dtype, device=bus_x.device)
    data['bus'].x = torch.cat([bus_x, pe_bus], dim=1)

    if need_ac_pe:
        ac_dc = branch_dc['ac_line']
        if ac_dc.shape[0] != int(data['branch_ac'].x.size(0)):
            raise RuntimeError(
                f"branch_ac PE row count {ac_dc.shape[0]} != node count "
                f"{int(data['branch_ac'].x.size(0))}; cycle/branch transform "
                f"must run BEFORE attach_pe_features_"
            )
        bx = data['branch_ac'].x
        pe_ac = torch.from_numpy(ac_dc).to(dtype=bx.dtype, device=bx.device)
        data['branch_ac'].x = torch.cat([bx, pe_ac], dim=1)

    if need_tr_pe:
        tr_dc = branch_dc['transformer']
        if tr_dc.shape[0] != int(data['branch_tr'].x.size(0)):
            raise RuntimeError(
                f"branch_tr PE row count {tr_dc.shape[0]} != node count "
                f"{int(data['branch_tr'].x.size(0))}"
            )
        bx = data['branch_tr'].x
        pe_tr = torch.from_numpy(tr_dc).to(dtype=bx.dtype, device=bx.device)
        data['branch_tr'].x = torch.cat([bx, pe_tr], dim=1)

    expected_bus_cols = 4 + 5 + 3 + 4
    if int(data['bus'].x.size(1)) < expected_bus_cols:
        bus_x = data['bus'].x
        bus_type = bus_x[:, BUS_TYPE_IDX].long().clamp(1, 4) - 1
        bus_type_oh = torch.nn.functional.one_hot(bus_type, num_classes=4).to(
            dtype=bus_x.dtype)
        data['bus'].x = torch.cat([bus_x, bus_type_oh], dim=1)

    if not hasattr(data['bus'], 'v_setpoint'):
        bus_x = data['bus'].x
        n_bus = bus_x.size(0)
        v_setpoint = torch.zeros(n_bus, 1, dtype=bus_x.dtype, device=bus_x.device)
        if ('generator', 'generator_link', 'bus') in data.edge_types:
            ei = data[('generator', 'generator_link', 'bus')].edge_index
            gen_x = data['generator'].x
            if gen_x.size(0) > 0 and ei.numel() > 0:
                src_vg = gen_x[ei[0], GEN_VG_IDX]
                ones = torch.ones_like(src_vg)
                vg_sum = torch.zeros(n_bus, device=bus_x.device, dtype=src_vg.dtype)
                vg_cnt = torch.zeros(n_bus, device=bus_x.device, dtype=src_vg.dtype)
                vg_sum.scatter_add_(0, ei[1], src_vg)
                vg_cnt.scatter_add_(0, ei[1], ones)
                v_mean = vg_sum / vg_cnt.clamp_min(1.0)
                v_setpoint[:, 0] = v_mean.to(dtype=bus_x.dtype)
        data['bus'].v_setpoint = v_setpoint

    if not hasattr(data, 'g_ctx'):
        data.g_ctx = _compute_graph_context(data).to(
            dtype=data['bus'].x.dtype, device=data['bus'].x.device,
        )

    setattr(data, _PE_ATTACHED_FLAG, True)
    return data


def _compute_graph_context(data: HeteroData) -> Tensor:
    if hasattr(data, "num_graphs") and int(getattr(data, "num_graphs", 1) or 1) > 1:
        raise ValueError(
            "_compute_graph_context is single-graph only. Call it on each "
            "HeteroData before batching, or use batch_data_list(...) which "
            "stacks per-graph g_ctx vectors."
        )
    eps = 1e-6
    n_bus = float(data['bus'].x.size(0)) if 'bus' in data.node_types else 0.0
    n_gen = float(data['generator'].x.size(0)) if 'generator' in data.node_types else 0.0

    if 'load' in data.node_types and data['load'].x.size(0) > 0:
        Pd_tot = float(data['load'].x[:, LOAD_PD_IDX].sum())
        Qd_tot = float(data['load'].x[:, LOAD_QD_IDX].sum())
    else:
        Pd_tot = Qd_tot = 0.0

    if n_gen > 0:
        gx = data['generator'].x
        Pmax_tot = float(gx[:, GEN_PMAX_IDX].sum())
        Qmax_tot = float(gx[:, GEN_QMAX_IDX].sum())
        Qmin_tot = float(gx[:, GEN_QMIN_IDX].sum())
        Qcap_tot = max(Qmax_tot - Qmin_tot, eps)
    else:
        Pmax_tot = eps
        Qcap_tot = eps

    V_spread = float((data['bus'].x[:, BUS_VMAX_IDX] - data['bus'].x[:, BUS_VMIN_IDX]).mean()) \
        if n_bus > 0 else 0.0

    rates = []
    abs_ys = []
    if ('bus', 'ac_line', 'bus') in data.edge_types:
        ea = data[('bus', 'ac_line', 'bus')].edge_attr
        if ea.size(0) > 0:
            rates.append(ea[:, AC_LINE_RATE_A_IDX])
            r = ea[:, AC_LINE_R_IDX]; x = ea[:, AC_LINE_X_IDX]
            z2 = (r.pow(2) + x.pow(2)).clamp_min(eps)
            abs_ys.append(z2.rsqrt())
    if ('bus', 'transformer', 'bus') in data.edge_types:
        ea = data[('bus', 'transformer', 'bus')].edge_attr
        if ea.size(0) > 0:
            rates.append(ea[:, TR_RATE_A_IDX])
            r = ea[:, TR_R_IDX]; x = ea[:, TR_X_IDX]
            z2 = (r.pow(2) + x.pow(2)).clamp_min(eps)
            abs_ys.append(z2.rsqrt())

    if rates:
        all_rates = torch.cat([t.float() for t in rates])
        all_ys    = torch.cat([t.float() for t in abs_ys])
        mean_rate  = float(all_rates.mean())
        mean_abs_y = float(all_ys.mean())
        max_abs_y  = float(all_ys.max())
    else:
        mean_rate = mean_abs_y = max_abs_y = 0.0

    g_ctx = torch.tensor([
        math.log(max(n_bus, 1.0)),
        math.log(n_gen + 1.0),
        Pd_tot / max(Pmax_tot, eps),
        Qd_tot / max(Qcap_tot, eps),
        V_spread,
        math.log1p(mean_rate),
        math.log1p(mean_abs_y),
        math.log1p(max_abs_y),
    ], dtype=torch.float32).unsqueeze(0)
    return g_ctx
