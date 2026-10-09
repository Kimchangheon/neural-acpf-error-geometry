"""Training loss for GridSFM fine-tuning. OPFData format only.

`compute_loss` is the entry point: tanh-capped squared error on
θ/V/Pg/Qg + BCE on feas-logit + log-MSE on cost + explicit KCL P/Q +
branch flow P/Q + thermal loading + thermal-limit barrier + stress-feas
regularizer. Each regression term is masked to the feasible-graph
subset of the batch (the synth-infeasible wrapper sets
`batch.feasible == 0` on perturbed samples).

Column indices (bus / generator / edge_attr / edge_label) are pinned to
the OPFData schema. Batches must come from `OPFDataAdapterDataset` (or
the equivalent OPFData PyG loader); other HeteroData formats are not
supported.

Single-GPU only (no DDP `find_unused_parameters` plumbing).

Inputs (set by the model forward):
  * `batch['bus'].pred` is `[N_bus, 2] = (θ, V)`
  * `batch['generator'].pred` is `[N_gen, 2] = (Pg, Qg)`
  * `batch[('bus', et, 'bus')].edge_flow_pred` is `[E, 4] = (Pij, Qij, Pji, Qji)`
  * `batch.feas_logit` is `[G]`
  * `batch.feasible` is `[G]` ground-truth feasibility from the dataset
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .blocks import _slack_or_mean_anchor
from .schema import (
    GEN_CP0_IDX, GEN_CP1_IDX, GEN_CP2_IDX,
    GEN_PMAX_IDX, GEN_QMAX_IDX,
    LOAD_PD_IDX, LOAD_QD_IDX,
    SHUNT_BS_IDX, SHUNT_GS_IDX,
)
from .stress_features import STRESS_VIOL_DIMS, compute_physics_stress


# Soft-cap on per-element squared error, applied as `cap * tanh(err / cap)`.
PER_ELEM_CAP = 100.0

# 80% worst-bus + 20% per-graph mean weighting for the KCL term.
_KCL_WORST_TILT = 0.8


# ─── Edge schema (dict-of-dicts for _slice_edge_cols) ──────────────────

_OPF_EDGE_SCHEMA: Dict[Tuple[str, str, str], Dict[str, int]] = {
    ('bus', 'ac_line', 'bus'): {
        'angmin_idx': 0, 'angmax_idx': 1,
        'b_fr_idx': 2, 'b_to_idx': 3,
        'r_idx': 4, 'x_idx': 5,
        'rate_a_idx': 6,
    },
    ('bus', 'transformer', 'bus'): {
        'angmin_idx': 0, 'angmax_idx': 1,
        'r_idx': 2, 'x_idx': 3,
        'rate_a_idx': 4,
        'tap_idx': 7, 'shift_idx': 8,
        'b_fr_idx': 9, 'b_to_idx': 10,
    },
}


def _slice_edge_cols(edge_attr: torch.Tensor, schema: Dict[str, int]) -> Dict[str, torch.Tensor]:
    return {k: edge_attr[:, idx] for k, idx in schema.items() if idx is not None}


# ─── Per-graph aggregation helpers ──────────────────────────────────────

def _macro_mean(x: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
    """Per-graph mean, then mean across graphs (cross-grid balanced)."""
    G = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 0
    if G == 0:
        return x.new_zeros(())
    num = torch.zeros(G, device=x.device, dtype=x.dtype).scatter_add_(0, batch_idx, x)
    cnt = torch.zeros(G, device=x.device, dtype=x.dtype).scatter_add_(
        0, batch_idx, torch.ones_like(x)).clamp_min_(1.0)
    return (num / cnt).mean()


def _per_graph_mean(per_elem: torch.Tensor, batch_idx: torch.Tensor,
                    n_graphs: int) -> torch.Tensor:
    """Per-graph mean of a [N] tensor. Returns [n_graphs]. Empty graphs get 0."""
    dtype = per_elem.dtype
    sums = torch.zeros(n_graphs, dtype=dtype, device=per_elem.device)
    cnt = torch.zeros(n_graphs, dtype=dtype, device=per_elem.device)
    sums.scatter_add_(0, batch_idx, per_elem)
    cnt.scatter_add_(0, batch_idx, torch.ones_like(batch_idx, dtype=dtype))
    return sums / cnt.clamp_min(1.0)


# ─── Feasibility masks ─────────────────────────────────────────────────

def build_feasible_masks(batch) -> Dict[str, object]:
    """Build masks for feasible graphs (`batch.feasible == 1`).

    Returns dict with: `gmask` (G,), `bus_mask` / `gen_mask` (N*,) or None,
    `edge_mask` = {'ac_line': (E,), 'transformer': (E,)} or {}.
    """
    dev = batch['bus'].x.device
    gmask = (batch.feasible.to(dev) > 0).view(-1)

    def mask_from(nt):
        return gmask[batch[nt].batch] if nt in batch.node_types else None

    edge_mask: Dict[str, torch.Tensor] = {}
    for et_name in ('ac_line', 'transformer'):
        key = ('bus', et_name, 'bus')
        if key in batch.edge_types:
            ei = batch[key].edge_index
            src_b = batch['bus'].batch[ei[0]]
            dst_b = batch['bus'].batch[ei[1]]
            edge_mask[et_name] = gmask[src_b] & gmask[dst_b]

    return dict(gmask=gmask, bus_mask=mask_from('bus'),
                gen_mask=mask_from('generator'), edge_mask=edge_mask)


# ─── Schema-aware edge gather (τ, φ, per-end charging) ─────────────────

def _gather_edges_schema(batch):
    """Gather i, j, r, x, b_fr, b_to, tau, phi across ac_line + transformer edge types.

    Defaults for missing schema fields: tau=1, phi=0, b_fr=b_to=0. Respects
    `edge_active_mask` when present.
    """
    dev = batch['bus'].x.device
    i_all, j_all = [], []
    r_all, x_all = [], []
    bfr_all, bto_all = [], []
    tau_all, phi_all = [], []

    for et_name in ('ac_line', 'transformer'):
        key = ('bus', et_name, 'bus')
        if key not in batch.edge_types or key not in _OPF_EDGE_SCHEMA:
            continue

        rel = batch[key]
        ei = rel.edge_index
        ea = rel.edge_attr
        cols = _slice_edge_cols(ea, _OPF_EDGE_SCHEMA[key])

        r = cols.get('r_idx', None)
        x = cols.get('x_idx', None)
        if r is None or x is None:
            continue

        b_fr = cols.get('b_fr_idx', torch.zeros_like(r))
        b_to = cols.get('b_to_idx', torch.zeros_like(r))
        tau = cols.get('tap_idx', torch.ones_like(r))
        phi = cols.get('shift_idx', torch.zeros_like(r))

        i_idx, j_idx = ei[0], ei[1]

        em = (rel.edge_active_mask.to(dev).bool()
              if hasattr(rel, "edge_active_mask") and rel.edge_active_mask is not None
              else None)
        if em is not None:
            i_idx, j_idx = i_idx[em], j_idx[em]
            r, x = r[em], x[em]
            b_fr, b_to = b_fr[em], b_to[em]
            tau, phi = tau[em], phi[em]

        i_all.append(i_idx); j_all.append(j_idx)
        r_all.append(r); x_all.append(x)
        bfr_all.append(b_fr); bto_all.append(b_to)
        tau_all.append(tau); phi_all.append(phi)

    if not i_all:
        E0L = torch.tensor([], dtype=torch.long, device=dev)
        E0F = torch.tensor([], dtype=torch.float32, device=dev)
        return E0L, E0L, E0F, E0F, E0F, E0F, E0F, E0F

    return (torch.cat(i_all), torch.cat(j_all),
            torch.cat(r_all), torch.cat(x_all),
            torch.cat(bfr_all), torch.cat(bto_all),
            torch.cat(tau_all), torch.cat(phi_all))


# ─── π-model branch flows (off-nominal tap τ, phase φ; bidirectional) ──

def _flows_rx_bidir(theta: torch.Tensor, V: torch.Tensor,
                    i_idx: torch.Tensor, j_idx: torch.Tensor,
                    r: torch.Tensor, x: torch.Tensor,
                    b_fr: torch.Tensor, b_to: torch.Tensor,
                    tau: torch.Tensor, phi: torch.Tensor):
    z2 = (r * r + x * x).clamp_min(1e-6)
    g = r / z2
    b = -x / z2

    Vi = V[i_idx]; Vj = V[j_idx]
    Vi2 = Vi * Vi; Vj2 = Vj * Vj

    tau = tau.clamp_min(1e-8)
    inv_tau = 1.0 / tau
    inv_tau2 = inv_tau * inv_tau

    dth = theta[i_idx] - theta[j_idx] - phi
    c = torch.cos(dth); s = torch.sin(dth)

    Pij = (Vi2 * g * inv_tau2) - (Vi * Vj * inv_tau) * (g * c + b * s)
    Qij = -(Vi2 * (b + b_fr) * inv_tau2) + (Vi * Vj * inv_tau) * (b * c - g * s)
    Pji = (Vj2 * g) - (Vi * Vj * inv_tau) * (g * c - b * s)
    Qji = -(Vj2 * (b + b_to)) + (Vi * Vj * inv_tau) * (b * c + g * s)

    return Pij, Qij, Pji, Qji


# ─── Per-graph capacity scales (Σ Pmax / Σ Qmax) for KCL normalization ─

def _per_graph_capacity_scales(batch):
    """Σ Pmax / Σ Qmax per graph. Capacity-normalization (not demand) keeps
    the loss INVARIANT across synth perturbation modes (Pd-spike modes
    inflate demand and would otherwise make infeas loss smaller than feas).
    """
    dev = batch['bus'].x.device
    G = int(batch['bus'].batch.max().item()) + 1
    if 'generator' not in batch.node_types:
        Cp = torch.ones(G, device=dev)
        return Cp, Cp
    xg = batch['generator'].x
    Pmax = torch.nan_to_num(xg[:, GEN_PMAX_IDX]).abs()
    Qmax = torch.nan_to_num(xg[:, GEN_QMAX_IDX]).abs()
    gidx = batch['generator'].batch
    Cp = torch.zeros(G, device=dev).scatter_add_(0, gidx, Pmax).clamp_min(1e-6)
    Cq = torch.zeros(G, device=dev).scatter_add_(0, gidx, Qmax).clamp_min(1e-6)
    return Cp, Cq


# ─── Device-level shunt injections (Psh = Gs·V², Qsh = +Bs·V²) ─────────

def _compute_bus_shunt_injections(batch, V: torch.Tensor):
    dev = V.device
    Nb = V.size(0)
    Psh_bus = torch.zeros(Nb, device=dev, dtype=V.dtype)
    Qsh_bus = torch.zeros(Nb, device=dev, dtype=V.dtype)

    have_device = (('shunt' in batch.node_types)
                   and (('shunt', 'shunt_link', 'bus') in batch.edge_types))
    if (not have_device or getattr(batch['shunt'], 'x', None) is None
            or batch['shunt'].x.numel() == 0):
        return Psh_bus, Qsh_bus

    s2b = batch['shunt', 'shunt_link', 'bus'].edge_index[1]
    xs = batch['shunt'].x
    if xs.size(1) >= 2:
        Bs, Gs = xs[:, SHUNT_BS_IDX], xs[:, SHUNT_GS_IDX]
    elif xs.size(1) == 1:
        Bs = xs[:, SHUNT_BS_IDX]
        Gs = torch.zeros_like(Bs)
    else:
        return Psh_bus, Qsh_bus

    V2_s = V[s2b] * V[s2b]
    Psh_bus.scatter_add_(0, s2b, Gs * V2_s)
    Qsh_bus.scatter_add_(0, s2b, +Bs * V2_s)
    return Psh_bus, Qsh_bus


# ─── Predicted flows + KCL residuals ───────────────────────────────────

def _predicted_flows(batch):
    """π-model edge flows from the model's predicted (θ, V). Returns
    (i, j, Pij, Qij, Pji, Qji). `Pij..Qji` are None on graphs with no
    edges.

    Intentionally recomputes from (θ, V) rather than reading the model's
    separate `edge_flow_pred` head: KCL is a PHYSICS check on whether
    the predicted (θ, V) conserves power, and `edge_flow_pred` comes
    from an independent head that isn't guaranteed to be consistent
    with (θ, V).
    """
    theta = batch['bus'].pred[:, 0]
    V = batch['bus'].pred[:, 1]
    i, j, r, x, b_fr, b_to, tau, phi = _gather_edges_schema(batch)
    if i.numel() == 0:
        return i, j, None, None, None, None
    Pij, Qij, Pji, Qji = _flows_rx_bidir(theta, V, i, j, r, x, b_fr, b_to, tau, phi)
    return i, j, Pij, Qij, Pji, Qji


def _kcl_residuals(batch, i, j, Pij, Qij, Pji, Qji, V):
    """Per-bus KCL residuals: residP = flowOut_P - (Pgen - Pload - Pshunt),
    residQ = flowOut_Q - (Qgen - Qload + Qshunt).
    """
    dev = V.device
    Nb = V.shape[0]
    z = torch.zeros(Nb, device=dev)

    if Pij is not None:
        flowP = z.clone().scatter_add(0, i, Pij).scatter_add(0, j, Pji)
        flowQ = z.clone().scatter_add(0, i, Qij).scatter_add(0, j, Qji)
    else:
        flowP = z.clone()
        flowQ = z.clone()

    g2b = batch['generator', 'generator_link', 'bus'].edge_index[1]
    Pgen = z.clone().scatter_add(0, g2b, batch['generator'].pred[:, 0])
    Qgen = z.clone().scatter_add(0, g2b, batch['generator'].pred[:, 1])

    if 'load' in batch.node_types:
        l2b = batch['load', 'load_link', 'bus'].edge_index[1]
        Pload = z.clone().scatter_add(0, l2b, batch['load'].x[:, LOAD_PD_IDX])
        Qd = (batch['load'].x[:, LOAD_QD_IDX] if batch['load'].x.size(1) > 1
              else torch.zeros(l2b.size(0), device=dev, dtype=V.dtype))
        Qload = z.clone().scatter_add(0, l2b, Qd)
    else:
        Pload = z
        Qload = z

    Psh, Qsh = _compute_bus_shunt_injections(batch, V)
    residP = flowP - (Pgen - Pload - Psh)
    residQ = flowQ - (Qgen - Qload + Qsh)
    return residP, residQ


# ─── Shared edge-flow gather (per-edge tensors + bidir flows) ──────────

def _gather_flow_inputs(batch, edge_mask, theta, V, *,
                        rated_threshold: float = 1e-6):
    """Generator yielding per-edge-type dicts. Used by flow_pq_mse_loss,
    loading_mse_loss, and thermal_limit_loss.

    Per ('bus', et, 'bus'):
      1. Schema column extraction.
      2. Active mask: explicit `edge_mask[et]` first, else `edge_active_mask`.
      3. Rated + non-degenerate filter: smax > rated_threshold & |x| > 0.01.
      4. Flow source: cached `edge_flow_pred` if present (expand if compressed
         by edge_active_mask), else π-model recompute.

    `rated_threshold` differs by caller on purpose. `flow_pq_mse_loss`
    passes `1e-6` (supervised MAE should cover every solver-reported
    branch); `loading_mse_loss` and `thermal_limit_loss` pass `1e-3`
    (thermal terms need a meaningful denominator, so dropped near-zero
    placeholder ratings).
    """
    dev = theta.device
    bus_batch = batch['bus'].batch
    for et_name in ('ac_line', 'transformer'):
        key = ('bus', et_name, 'bus')
        if key not in batch.edge_types or key not in _OPF_EDGE_SCHEMA:
            continue
        ed = batch[key]
        if not hasattr(ed, 'edge_label') or ed.edge_label is None:
            continue
        ei = ed.edge_index
        ea = ed.edge_attr
        gt = ed.edge_label
        if gt.size(0) == 0:
            continue

        cols = _slice_edge_cols(ea, _OPF_EDGE_SCHEMA[key])
        r = cols.get('r_idx', None)
        x_col = cols.get('x_idx', None)
        if r is None or x_col is None:
            continue
        b_fr = cols.get('b_fr_idx', torch.zeros_like(r))
        b_to = cols.get('b_to_idx', torch.zeros_like(r))
        smax = cols.get('rate_a_idx', torch.zeros_like(r))
        tau = cols.get('tap_idx', torch.ones_like(r))
        phi_col = cols.get('shift_idx', torch.zeros_like(r))

        i_idx, j_idx = ei[0], ei[1]

        em_active = (
            ed.edge_active_mask.to(dev).bool()
            if hasattr(ed, 'edge_active_mask') and ed.edge_active_mask is not None
            else None
        )
        em_feas = (
            edge_mask[et_name].to(dev).bool()
            if edge_mask is not None and et_name in edge_mask
            else None
        )
        if em_feas is not None and em_active is not None:
            em = em_feas & em_active
        elif em_feas is not None:
            em = em_feas
        elif em_active is not None:
            em = em_active
        else:
            em = None

        if em is not None:
            i_idx, j_idx = i_idx[em], j_idx[em]
            r, x_col = r[em], x_col[em]
            b_fr, b_to = b_fr[em], b_to[em]
            smax, tau, phi_col = smax[em], tau[em], phi_col[em]
            gt = gt[em]

        keep = (smax > rated_threshold) & (x_col.abs() > 0.01)
        if not keep.any():
            continue
        i_idx, j_idx = i_idx[keep], j_idx[keep]
        r, x_col = r[keep], x_col[keep]
        b_fr, b_to = b_fr[keep], b_to[keep]
        smax, tau, phi_col = smax[keep], tau[keep], phi_col[keep]
        gt = gt[keep]

        ed = batch[key]
        if (hasattr(ed, 'edge_flow_pred')
                and ed.edge_flow_pred is not None):
            _efp = ed.edge_flow_pred
            _n_full = ed.edge_index.size(1)
            _am = getattr(ed, 'edge_active_mask', None)
            if _efp.size(0) == _n_full:
                pass
            elif _am is not None:
                _n_active = int(_am.sum().item())
                if _efp.size(0) == _n_active:
                    _efp_full = torch.zeros(_n_full, _efp.size(1),
                                            device=_efp.device, dtype=_efp.dtype)
                    _efp_full[_am.to(_efp.device).bool()] = _efp
                    _efp = _efp_full
                else:
                    raise AssertionError(
                        f'edge_flow_pred size {_efp.size(0)} matches neither '
                        f'full ({_n_full}) nor active ({_n_active}) on {key}.'
                    )
            else:
                raise AssertionError(
                    f'edge_flow_pred size ({_efp.size(0)}) != full edge count '
                    f'({_n_full}) on {key} with no edge_active_mask.'
                )
            if em is not None:
                _efp = _efp[em]
            _efp = _efp[keep]
            Pij, Qij, Pji, Qji = _efp[:, 0], _efp[:, 1], _efp[:, 2], _efp[:, 3]
        else:
            Pij, Qij, Pji, Qji = _flows_rx_bidir(
                theta, V, i_idx, j_idx, r, x_col, b_fr, b_to, tau, phi_col)

        yield {
            'Pij': Pij, 'Qij': Qij, 'Pji': Pji, 'Qji': Qji,
            'gt': gt,
            'smax': smax,
            'batch_idx': bus_batch[i_idx],
        }


# ─── Per-edge thermal loading ──────────────────────────────────────────

_RATED_THRESHOLD = 0.001  # smax > this is "rated"; smaller values are placeholder ratings


def _thermal_loading(Pij, Qij, Pji, Qji, smax):
    """Per-edge max(|Sij|, |Sji|) / smax_eff. Branches with smax <=
    `_RATED_THRESHOLD` are treated as unrated (loading = 0) to skip
    placeholder ratings. `smax_eff = sqrt(smax^2 + 1e-4)` accounts for
    Ipopt's solver tolerance.
    """
    Sij = torch.sqrt(Pij * Pij + Qij * Qij + 1e-12)
    Sji = torch.sqrt(Pji * Pji + Qji * Qji + 1e-12)
    Speak = torch.maximum(Sij, Sji)
    rated = smax > _RATED_THRESHOLD
    loading = torch.zeros_like(Speak)
    if rated.any():
        smax_eff = torch.sqrt(smax[rated].square() + 1e-4).clamp_min(1e-6)
        loading[rated] = Speak[rated] / smax_eff
    return loading, rated


# ─── Thermal limit loss (soft top-k barrier, log1p-compressed) ─────────

_THERM_BETA = 8.0       # softmax temperature for top-k thermal focus
_THERM_EPS_MEAN = 0.05  # weight on the plain-mean term alongside soft-focus


def thermal_limit_loss(batch, edge_mask=None):
    """Soft top-k thermal-limit barrier per graph, reduced as mean over graphs.

      v_e_raw = relu(|S_e|/Smax_e - 1)
      v_e     = log1p(v_e_raw)
      L_g     = sum_e softmax_g(β v_e) v_e + ε_mean · mean_g(v_e)
    """
    dev = batch['bus'].pred.device
    bus_batch = batch['bus'].batch
    G = int(getattr(batch, 'num_graphs', 0)
            or (int(bus_batch.max().item()) + 1 if bus_batch.numel() > 0 else 1))
    z = batch['bus'].pred.new_zeros(())

    theta = batch['bus'].pred[:, 0]
    V = batch['bus'].pred[:, 1]
    all_Pij, all_Qij, all_Pji, all_Qji = [], [], [], []
    all_smax, all_gid = [], []
    for d in _gather_flow_inputs(batch, edge_mask, theta, V,
                                 rated_threshold=_RATED_THRESHOLD):
        all_Pij.append(d['Pij']); all_Qij.append(d['Qij'])
        all_Pji.append(d['Pji']); all_Qji.append(d['Qji'])
        all_smax.append(d['smax'])
        all_gid.append(d['batch_idx'])

    if not all_Pij:
        return z

    loading, rated = _thermal_loading(
        torch.cat(all_Pij), torch.cat(all_Qij),
        torch.cat(all_Pji), torch.cat(all_Qji),
        torch.cat(all_smax),
    )
    if not rated.any():
        return z

    v = torch.log1p((loading[rated] - 1.0).clamp_min(0.0))
    gid = torch.cat(all_gid)[rated]

    x = _THERM_BETA * v
    max_x = torch.full((G,), -1e30, device=dev, dtype=v.dtype) \
        .scatter_reduce(0, gid, x, reduce="amax", include_self=True)
    ex = torch.exp(x - max_x[gid])
    Z = torch.zeros(G, device=dev, dtype=v.dtype) \
        .scatter_add(0, gid, ex).clamp_min(1e-6)
    w = ex / Z[gid]

    soft_focus = torch.zeros(G, device=dev, dtype=v.dtype) \
        .scatter_add(0, gid, w * v)
    plain_sum = torch.zeros(G, device=dev, dtype=v.dtype) \
        .scatter_add(0, gid, v)
    counts = torch.zeros(G, device=dev, dtype=v.dtype) \
        .scatter_add(0, gid, torch.ones_like(v))
    plain_mean = plain_sum / counts.clamp_min(1.0)

    return (soft_focus + _THERM_EPS_MEAN * plain_mean).mean()


# ─── Flow P/Q MSE (supervised against solver edge_label, raw pu²) ──────

def flow_pq_mse_loss(batch, *, edge_mask=None):
    """Branch flow P and Q MSE in pu², macro-averaged across edge types
    and graphs. Returns `(L_P, L_Q)`. The 1e-6 rated-threshold is the
    'non-degenerate' floor (matches the supervised flow MAE in eval).
    """
    theta = batch['bus'].pred[:, 0]
    V = batch['bus'].pred[:, 1]
    errs_P_all, errs_Q_all, batch_all = [], [], []
    for d in _gather_flow_inputs(batch, edge_mask, theta, V, rated_threshold=1e-6):
        gt = d['gt']
        err_P = (d['Pij'] - gt[:, 2]).pow(2) + (d['Pji'] - gt[:, 0]).pow(2)
        err_Q = (d['Qij'] - gt[:, 3]).pow(2) + (d['Qji'] - gt[:, 1]).pow(2)
        errs_P_all.append(err_P)
        errs_Q_all.append(err_Q)
        batch_all.append(d['batch_idx'])

    if not errs_P_all:
        z = batch['bus'].pred.new_zeros(())
        return z, z.clone()
    batches = torch.cat(batch_all)
    return (_macro_mean(torch.cat(errs_P_all), batches),
            _macro_mean(torch.cat(errs_Q_all), batches))


# ─── Loading MSE (supervised, log1p-compressed) ─────────────────────────

def loading_mse_loss(batch, *, edge_mask=None):
    """`(load_pred - load_true)^2` macro-averaged.
    `load = max(|S_ij|, |S_ji|) / rate_a` (worst-end convention).
    `log1p` on per-branch squared error bounds extreme outliers.
    """
    theta = batch['bus'].pred[:, 0]
    V = batch['bus'].pred[:, 1]

    errs_all, batch_all = [], []
    for d in _gather_flow_inputs(batch, edge_mask, theta, V,
                                 rated_threshold=_RATED_THRESHOLD):
        gt = d['gt']
        Pij, Qij, Pji, Qji = d['Pij'], d['Qij'], d['Pji'], d['Qji']
        smax_e = torch.sqrt(d['smax'].square() + 1e-4).clamp_min(1e-3)
        S_pred_ij = (Pij.pow(2) + Qij.pow(2)).clamp_min(1e-6).sqrt()
        S_pred_ji = (Pji.pow(2) + Qji.pow(2)).clamp_min(1e-6).sqrt()
        load_pred = torch.maximum(S_pred_ij, S_pred_ji) / smax_e
        S_true_ij = (gt[:, 2].pow(2) + gt[:, 3].pow(2)).clamp_min(1e-6).sqrt()
        S_true_ji = (gt[:, 0].pow(2) + gt[:, 1].pow(2)).clamp_min(1e-6).sqrt()
        load_true = torch.maximum(S_true_ij, S_true_ji) / smax_e
        errs_all.append(torch.log1p((load_pred - load_true).pow(2)))
        batch_all.append(d['batch_idx'])

    if not errs_all:
        return batch['bus'].pred.new_zeros(())
    return _macro_mean(torch.cat(errs_all), torch.cat(batch_all))


# ─── KCL residual loss (capacity-normalized, mean + worst-bus tilt) ────

_KCL_WORST_MARGIN = 0.01  # ignore per-bus residuals below 1% capacity in worst-bus tilt


def kcl_residual_loss(batch, *, bus_mask=None):
    """Differentiable per-graph KCL loss. Capacity-normalized per-bus
    residuals, combined as `_KCL_WORST_TILT`-weighted blend of mean +
    worst-bus penalty per graph (worst-bus uses scatter_reduce amax over
    residuals beyond `_KCL_WORST_MARGIN`). Returns `(L_P, L_Q)` per-graph
    tensors (shape `[G]`) — caller masks and reduces.
    """
    dev = batch['bus'].pred.device
    gbus = batch['bus'].batch
    G = int(gbus.max().item()) + 1 if gbus.numel() > 0 else 1
    V = batch['bus'].pred[:, 1]

    i, j, Pij, Qij, Pji, Qji = _predicted_flows(batch)
    residP, residQ = _kcl_residuals(batch, i, j, Pij, Qij, Pji, Qji, V)

    if bus_mask is not None:
        if not bus_mask.any():
            zg = torch.zeros(G, device=dev)
            return zg, zg.clone()
        residP = residP[bus_mask]
        residQ = residQ[bus_mask]
        gbus = gbus[bus_mask]

    Dp_g, Dq_g = _per_graph_capacity_scales(batch)
    normP = (residP / Dp_g[gbus].clamp_min(1e-6)).abs()
    normQ = (residQ / Dq_g[gbus].clamp_min(1e-6)).abs()

    ones = torch.ones_like(normP)
    cnt = torch.zeros(G, device=dev).scatter_add_(0, gbus, ones).clamp_min_(1.0)
    P_mean = torch.zeros(G, device=dev).scatter_add_(0, gbus, normP) / cnt
    Q_mean = torch.zeros(G, device=dev).scatter_add_(0, gbus, normQ) / cnt

    thrP = (normP - _KCL_WORST_MARGIN).clamp_min(0.0)
    thrQ = (normQ - _KCL_WORST_MARGIN).clamp_min(0.0)
    P_worst = torch.zeros(G, device=dev).scatter_reduce_(
        0, gbus, thrP, reduce='amax', include_self=True)
    Q_worst = torch.zeros(G, device=dev).scatter_reduce_(
        0, gbus, thrQ, reduce='amax', include_self=True)

    w = _KCL_WORST_TILT
    return (w * P_worst + (1.0 - w) * P_mean,
            w * Q_worst + (1.0 - w) * Q_mean)


# ─── Per-graph generation cost ─────────────────────────────────────────

def _per_graph_gen_cost(batch, Pg: torch.Tensor) -> torch.Tensor:
    """Σ_g (c2 Pg² + c1 Pg + c0) scattered into a [G] tensor. Shared
    between `compute_loss` and `eval_pass` so the cost formula and its
    column conventions live in exactly one place.
    """
    gx = batch['generator'].x
    c2, c1, c0 = gx[:, GEN_CP2_IDX], gx[:, GEN_CP1_IDX], gx[:, GEN_CP0_IDX]
    per_gen = c2 * Pg.pow(2) + c1 * Pg + c0
    G = int(batch['bus'].batch.max().item()) + 1
    return per_gen.new_zeros(G).scatter_add(
        0, batch['generator'].batch, per_gen)


# ─── Top-level compute_loss ─────────────────────────────────────────────

def compute_loss(
    batch,
    lambda_feas: float = 0.1,
    lambda_cost: float = 1.0,
    lambda_stress_feas: float = 0.1,
    lambda_kcl_p: float = 1.0,
    lambda_kcl_q: float = 1.0,
    lambda_br_p: float = 1.0,
    lambda_br_q: float = 1.0,
    lambda_therm: float = 1.0,
    lambda_thermal_limit: float = 5.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Tanh-capped squared error on θ/V/Pg/Qg + BCE on feas + log-MSE on
    cost + log1p-compressed KCL P/Q + log1p-compressed branch flow P/Q +
    thermal loading + thermal-limit barrier + stress-feas regularizer.

    All regression terms are gated to feasible graphs; L_feas covers the
    full batch (classifier on the feasibility label).

    Returns (L_total, parts_dict). Parts keys mirror the lambda kwarg
    names (`L_X` ↔ `lambda_X`): `parts['L_kcl_p' / 'L_kcl_q' / 'L_br_p' /
    'L_br_q']` are RAW pre-`log1p` values; the actual contribution to
    L_total is `lambda_X * log1p(parts[L_X])`. Other `parts['L_*']`
    enter L_total directly under their matching lambdas.
    """
    masks = build_feasible_masks(batch)
    gmask = masks['gmask']
    bus_mask = masks['bus_mask']
    gen_mask = masks['gen_mask']
    edge_mask = masks['edge_mask']

    n_g_full = int(getattr(batch, 'num_graphs', 1) or 1)
    bp = batch['bus'].pred           # [Nb, 2] = (θ, V)
    bt = batch['bus'].y              # [Nb, 2]
    gp = batch['generator'].pred     # [Ng, 2] = (Pg, Qg)
    gt = batch['generator'].y        # [Ng, 2]

    z = bp.new_zeros(())

    if bus_mask is not None and bus_mask.any():
        theta_gt_anchored = _slack_or_mean_anchor(batch, bt[:, 0], n_graphs=n_g_full)
        d_theta = torch.atan2(
            torch.sin(bp[bus_mask, 0] - theta_gt_anchored[bus_mask]),
            torch.cos(bp[bus_mask, 0] - theta_gt_anchored[bus_mask]),
        )
        sq_theta = d_theta.pow(2)

        Vt_masked = bt[bus_mask, 1]
        Vp_masked = bp[bus_mask, 1]
        n_g = int(gmask.numel()) if gmask is not None else 1
        bus_batch = batch['bus'].batch[bus_mask]
        V_mean = _per_graph_mean(Vt_masked, bus_batch, n_g)
        V_var = _per_graph_mean((Vt_masked - V_mean[bus_batch]).pow(2), bus_batch, n_g)
        V_std = V_var.sqrt().clamp_min(0.01)
        sq_V = ((Vp_masked - Vt_masked) / V_std[bus_batch]).pow(2)

        capped_sq_theta = PER_ELEM_CAP * torch.tanh(sq_theta / PER_ELEM_CAP)
        capped_sq_V = PER_ELEM_CAP * torch.tanh(sq_V / PER_ELEM_CAP)
        theta_pg = _per_graph_mean(capped_sq_theta, bus_batch, n_g)
        V_pg = _per_graph_mean(capped_sq_V, bus_batch, n_g)
        L_theta = theta_pg[gmask].mean()
        L_V = V_pg[gmask].mean()
    else:
        L_theta = z; L_V = z

    if gen_mask is not None and gen_mask.any():
        sq_Pg = (gp[gen_mask, 0] - gt[gen_mask, 0]).pow(2)
        sq_Qg = (gp[gen_mask, 1] - gt[gen_mask, 1]).pow(2)
        n_g = int(gmask.numel()) if gmask is not None else 1
        gen_batch = batch['generator'].batch[gen_mask]
        capped_sq_Pg = PER_ELEM_CAP * torch.tanh(sq_Pg / PER_ELEM_CAP)
        capped_sq_Qg = PER_ELEM_CAP * torch.tanh(sq_Qg / PER_ELEM_CAP)
        Pg_pg = _per_graph_mean(capped_sq_Pg, gen_batch, n_g)
        Qg_pg = _per_graph_mean(capped_sq_Qg, gen_batch, n_g)
        L_Pg = Pg_pg[gmask].mean()
        L_Qg = Qg_pg[gmask].mean()
    else:
        L_Pg = z; L_Qg = z

    L_cost = z
    if gmask is not None and gmask.any():
        cp_f = _per_graph_gen_cost(batch, gp[:, 0])[gmask]
        cg_f = _per_graph_gen_cost(batch, gt[:, 0])[gmask].detach()
        L_cost = (torch.log1p(cp_f.clamp_min(0.0))
                  - torch.log1p(cg_f.clamp_min(0.0))).pow(2).mean()

    feas_label = batch.feasible.to(batch['bus'].x.device).float().view(-1)
    feas_logit = batch.feas_logit.view(-1)
    L_feas = F.binary_cross_entropy_with_logits(feas_logit, feas_label)

    L_stress_feas = z
    if gmask is not None and gmask.any():
        ac_key = ('bus', 'ac_line', 'bus')
        tr_key = ('bus', 'transformer', 'bus')
        ac_flows = (batch[ac_key].edge_flow_pred
                    if ac_key in batch.edge_types
                    and hasattr(batch[ac_key], 'edge_flow_pred') else None)
        tr_flows = (batch[tr_key].edge_flow_pred
                    if tr_key in batch.edge_types
                    and hasattr(batch[tr_key], 'edge_flow_pred') else None)
        stress = compute_physics_stress(
            batch, V=bp[:, 1], theta=bp[:, 0],
            Pg=gp[:, 0], Qg=gp[:, 1],
            ac_flows=ac_flows, tr_flows=tr_flows, n_graphs=n_g_full,
        )
        if torch.isfinite(stress).all():
            viol = stress[gmask][:, :STRESS_VIOL_DIMS].abs()
            L_stress_feas = torch.log1p(viol).mean()

    L_kcl_P = L_kcl_Q = z
    if gmask is not None and gmask.any():
        L_kcl_P_per_g, L_kcl_Q_per_g = kcl_residual_loss(
            batch, bus_mask=bus_mask)
        L_kcl_P = L_kcl_P_per_g[gmask].mean()
        L_kcl_Q = L_kcl_Q_per_g[gmask].mean()

    L_brP = L_brQ = L_therm = L_therm_lim = z
    if gmask is not None and gmask.any():
        L_brP, L_brQ = flow_pq_mse_loss(batch, edge_mask=edge_mask)
        L_therm = loading_mse_loss(batch, edge_mask=edge_mask)
        L_therm_lim = thermal_limit_loss(batch, edge_mask=edge_mask)

    L = (L_theta + L_V + L_Pg + L_Qg
         + lambda_feas * L_feas
         + lambda_cost * L_cost
         + lambda_stress_feas * L_stress_feas
         + lambda_kcl_p * torch.log1p(L_kcl_P)
         + lambda_kcl_q * torch.log1p(L_kcl_Q)
         + lambda_br_p * torch.log1p(L_brP)
         + lambda_br_q * torch.log1p(L_brQ)
         + lambda_therm * L_therm
         + lambda_thermal_limit * L_therm_lim)

    def _f(t):
        return float(t.detach()) if isinstance(t, torch.Tensor) else 0.0

    parts = {
        'L_total':       _f(L),
        'L_theta':       _f(L_theta),
        'L_V':           _f(L_V),
        'L_Pg':          _f(L_Pg),
        'L_Qg':          _f(L_Qg),
        'L_feas':        _f(L_feas),
        'L_cost':        _f(L_cost),
        'L_stress_feas': _f(L_stress_feas),
        'L_kcl_p':       _f(L_kcl_P),
        'L_kcl_q':       _f(L_kcl_Q),
        'L_br_p':        _f(L_brP),
        'L_br_q':        _f(L_brQ),
        'L_therm':       _f(L_therm),
        'L_therm_lim':   _f(L_therm_lim),
    }
    return L, parts
