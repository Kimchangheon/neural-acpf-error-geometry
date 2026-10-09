"""Eval helper for GridSFM fine-tuning. OPFData format only.

`eval_pass(model, loader)` runs the model over a DataLoader and returns
the headline FT metrics: per-element MAEs on θ/V/Pg/Qg, per-graph cost
MAPE, branch flow P/Q MAE, KCL P/Q residual, per-edge thermal loading,
and feas-head accuracy. Regression metrics filter to feasible graphs
(synth-perturbed samples contribute nothing); `feas_acc` covers the
whole batch.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .loss import (
    _gather_flow_inputs,
    _kcl_residuals,
    _per_graph_capacity_scales,
    _per_graph_gen_cost,
    _predicted_flows,
    _thermal_loading,
    build_feasible_masks,
    compute_loss,
)


@torch.no_grad()
def eval_pass(
    model,
    loader,
    device=None,
    loss_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Returns a metrics dict for the FT eval pass.

    Keys:
      loss:        average `compute_loss(batch, **loss_kwargs)` over the loader
      cost_mape:   per-graph |Σcost_pred - Σcost_gt| / max(Σcost_gt, 1)
      pg_mae, qg_mae, V_mae, theta_mae:  per-element MAE on feasible nodes
      brP_mae, brQ_mae:  per-end MAE on feasible, non-degenerate branches
                         against the solver's `edge_label` (denominator
                         is `2 * n_branches_after_filter`)
      kcl_P_resid, kcl_Q_resid:  per-bus |KCL residual| / per-graph
                                  capacity, averaged across feasible buses
      thermal_max_loading:  mean across feasible-with-rated-edges graphs
                            of `max_e |S_e| / smax_e` (>1.0 = violation);
                            NaN if no rated edges seen anywhere
      thermal_frac_overload:  fraction of feasible rated edges with
                              loading > 1.0; NaN if no rated edges
      feas_acc:    binary accuracy of `feas_logit > 0` against
                   `batch.feasible` (covers all graphs, not just feasible)
      n_graphs:    number of feasible graphs counted

    `loss` reuses `loss_kwargs` so it stays comparable to the FT
    train-loss when lambdas are tuned. `theta_mae` uses circular
    distance via `atan2(sin(d), cos(d))` to wrap `(pred - gt)` into
    `[-π, π]` so a 2π wrapping discrepancy doesn't inflate the metric.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    loss_kwargs = loss_kwargs or {}

    sum_loss, n_iters = 0.0, 0
    sum_pg = sum_qg = sum_v = sum_theta = 0.0
    n_gens = n_buses = 0
    sum_cost_abs_pct, n_graphs_feas = 0.0, 0
    sum_brP = sum_brQ = 0.0
    n_br = 0
    sum_kcl_P = sum_kcl_Q = 0.0
    n_kcl_buses = 0
    sum_max_loading = 0.0
    n_thermal_graphs = 0
    n_overload = n_rated = 0
    n_feas_correct = n_feas_total = 0

    for batch in loader:
        batch = batch.to(device)
        model(batch)  # writes pred / feas_logit onto batch
        loss, _ = compute_loss(batch, **loss_kwargs)
        # Symmetric with `finetune_opfdata`: a single NaN batch would
        # poison the running average; skip and keep going.
        if torch.isfinite(loss):
            sum_loss += float(loss.item())
            n_iters += 1

        pred_bin = (batch.feas_logit.view(-1) > 0).long()
        true_bin = batch.feasible.view(-1).long()
        n_feas_correct += int((pred_bin == true_bin).sum())
        n_feas_total += pred_bin.numel()

        masks = build_feasible_masks(batch)
        bus_mask = masks["bus_mask"]
        gen_mask = masks["gen_mask"]
        edge_mask = masks["edge_mask"]
        gmask = masks["gmask"]

        bp = batch["bus"].pred
        bt = batch["bus"].y
        gp = batch["generator"].pred
        gt = batch["generator"].y

        if bus_mask is not None and bus_mask.any():
            d_th = bp[bus_mask, 0] - bt[bus_mask, 0]
            theta_err = torch.atan2(torch.sin(d_th), torch.cos(d_th)).abs()
            sum_theta += float(theta_err.sum())
            sum_v     += float((bp[bus_mask, 1] - bt[bus_mask, 1]).abs().sum())
            n_buses   += int(bus_mask.sum())
        if gen_mask is not None and gen_mask.any():
            sum_pg += float((gp[gen_mask, 0] - gt[gen_mask, 0]).abs().sum())
            sum_qg += float((gp[gen_mask, 1] - gt[gen_mask, 1]).abs().sum())
            n_gens += int(gen_mask.sum())

        if gmask is not None and gmask.any():
            cp_g = _per_graph_gen_cost(batch, gp[:, 0])
            cg_g = _per_graph_gen_cost(batch, gt[:, 0])
            rel = (cp_g[gmask] - cg_g[gmask]).abs() / cg_g[gmask].clamp_min(1.0)
            sum_cost_abs_pct += float(rel.sum())
            n_graphs_feas += int(gmask.sum())

            # Branch flow MAE + thermal loading: shared edge gather.
            # gt cols are [Pji, Qji, Pij, Qij].
            theta = bp[:, 0]
            V = bp[:, 1]
            all_Pij, all_Qij, all_Pji, all_Qji = [], [], [], []
            all_smax, all_gid = [], []
            for d in _gather_flow_inputs(batch, edge_mask, theta, V,
                                          rated_threshold=1e-6):
                Pij, Qij = d["Pij"], d["Qij"]
                Pji, Qji = d["Pji"], d["Qji"]
                gt_edge = d["gt"]
                sum_brP += float((Pij - gt_edge[:, 2]).abs().sum()
                                  + (Pji - gt_edge[:, 0]).abs().sum())
                sum_brQ += float((Qij - gt_edge[:, 3]).abs().sum()
                                  + (Qji - gt_edge[:, 1]).abs().sum())
                n_br    += int(Pij.numel()) * 2
                all_Pij.append(Pij); all_Qij.append(Qij)
                all_Pji.append(Pji); all_Qji.append(Qji)
                all_smax.append(d["smax"])
                all_gid.append(d["batch_idx"])

            if all_Pij:
                Pij_c = torch.cat(all_Pij); Qij_c = torch.cat(all_Qij)
                Pji_c = torch.cat(all_Pji); Qji_c = torch.cat(all_Qji)
                smax_c = torch.cat(all_smax)
                gid_c = torch.cat(all_gid)
                loading, rated = _thermal_loading(Pij_c, Qij_c, Pji_c, Qji_c, smax_c)
                if rated.any():
                    loading_r = loading[rated]
                    gid_r = gid_c[rated]
                    n_g = int(gmask.numel())
                    max_load_pg = torch.zeros(n_g, device=loading.device,
                                               dtype=loading.dtype).scatter_reduce_(
                        0, gid_r, loading_r, reduce="amax", include_self=True)
                    # Only count graphs that actually had ≥1 rated edge.
                    contrib = gmask & (max_load_pg > 0)
                    sum_max_loading += float(max_load_pg[contrib].sum())
                    n_thermal_graphs += int(contrib.sum())
                    n_overload += int((loading_r > 1.0).sum())
                    n_rated += int(loading_r.numel())

            if bus_mask is not None and bus_mask.any():
                i_kcl, j_kcl, Pij_k, Qij_k, Pji_k, Qji_k = _predicted_flows(batch)
                residP, residQ = _kcl_residuals(batch, i_kcl, j_kcl,
                                                 Pij_k, Qij_k, Pji_k, Qji_k, V)
                gbus = batch["bus"].batch
                Cp, Cq = _per_graph_capacity_scales(batch)
                normP = (residP[bus_mask] / Cp[gbus[bus_mask]].clamp_min(1e-6)).abs()
                normQ = (residQ[bus_mask] / Cq[gbus[bus_mask]].clamp_min(1e-6)).abs()
                sum_kcl_P += float(normP.sum())
                sum_kcl_Q += float(normQ.sum())
                n_kcl_buses += int(bus_mask.sum())

    # NaN (not 0.0) when no element contributed — 0.0 would falsely
    # signal perfect performance on an empty / all-skipped loader.
    nan = float("nan")
    def _avg(num, den): return num / den if den > 0 else nan
    return {
        "loss":      _avg(sum_loss, n_iters),
        "cost_mape": _avg(sum_cost_abs_pct, n_graphs_feas),
        "pg_mae":    _avg(sum_pg, n_gens),
        "qg_mae":    _avg(sum_qg, n_gens),
        "V_mae":     _avg(sum_v, n_buses),
        "theta_mae": _avg(sum_theta, n_buses),
        "brP_mae":   _avg(sum_brP, n_br),
        "brQ_mae":   _avg(sum_brQ, n_br),
        "kcl_P_resid": _avg(sum_kcl_P, n_kcl_buses),
        "kcl_Q_resid": _avg(sum_kcl_Q, n_kcl_buses),
        "thermal_max_loading":  _avg(sum_max_loading, n_thermal_graphs),
        "thermal_frac_overload": _avg(n_overload, n_rated),
        "feas_acc":  _avg(n_feas_correct, n_feas_total),
        "n_graphs":  n_graphs_feas,
    }
