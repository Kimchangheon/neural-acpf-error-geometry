#!/usr/bin/env python3
"""
Shared pieces for the AC-OPF surrogate task.

The power-flow task hands a model the full injection vector and asks for the
voltages that balance it. The OPF task is different in kind: only the loads are
given, the generator dispatch is part of the answer, and the answer must respect
the generation limits, the voltage band, and the branch ratings.

A voltage-only head is still sufficient. Once the voltage profile is predicted,
the implied injection follows from the network equation

    S_implied = V * conj(Y V),

so the dispatch is recovered rather than predicted separately, and the three
surrogates keep their existing per-bus voltage heads.

That changes what the supervision and the metrics can say per bus:

  * at a bus with no controllable generation the injection is known, so the
    power balance is an equality and its residual is a genuine error;
  * at a controllable bus the injection is free within its limits, so the
    meaningful quantity is not a residual but the amount by which the implied
    dispatch falls outside `[p_min, p_max] x [q_min, q_max]`.

Reporting a plain power-balance residual over all buses on OPF data would
therefore count correct redispatch as error. Everything here masks by
`Gen_controllable` accordingly.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

# Columns the OPF parquets carry in addition to the power-flow schema. Pass this
# to ChanghunDataset(extra_binary_columns=...); power-flow parquets simply do
# not have them and the loader skips whatever is absent.
OPF_EXTRA_COLUMNS = [
    "S_load",
    "S_gen_opf",
    "Gen_p_min",
    "Gen_p_max",
    "Gen_q_min",
    "Gen_q_max",
    "Gen_p_avail",
    "Gen_cost_c1",
    "Gen_controllable",
    "Gen_box_injection",
    "Bus_vmin",
    "Bus_vmax",
    "Branch_rate_a",
]


def has_opf_columns(batch: Dict[str, torch.Tensor]) -> bool:
    return "Gen_controllable" in batch and "Gen_p_max" in batch


def _flat(x: torch.Tensor) -> torch.Tensor:
    """Collate adds a leading batch dimension of 1; drop it."""
    return x.squeeze(0) if x.dim() > 1 and x.shape[0] == 1 else x


def _per_graph_base(batch: Dict[str, torch.Tensor], n_nodes: torch.Tensor,
                    device, dtype=torch.float64) -> torch.Tensor:
    """Expand per-graph S_base to a per-bus vector."""
    s_base = batch.get("S_base", None)
    if s_base is None:
        return torch.ones(int(n_nodes.sum()), dtype=dtype, device=device)
    s_base = s_base.to(device=device, dtype=dtype).reshape(-1)
    return torch.repeat_interleave(s_base, n_nodes.to(device))


def opf_decision_space(batch: Dict[str, torch.Tensor], n_nodes: torch.Tensor,
                       device, dtype=torch.float64) -> Dict[str, torch.Tensor]:
    """Per-bus decision space in per-unit on the batch's target base."""
    base = _per_graph_base(batch, n_nodes, device, dtype)
    out: Dict[str, torch.Tensor] = {}
    for key in ("Gen_p_min", "Gen_p_max", "Gen_q_min", "Gen_q_max", "Gen_p_avail"):
        if key in batch:
            out[key] = _flat(batch[key]).to(device=device, dtype=dtype) / base
    # Cost coefficients: the parquet pipeline stores EUR/MW and needs the
    # base to reach EUR/pu; OPFData already stores per-unit coefficients and
    # passes cost_scale=1. Getting this wrong rescales the absolute cost but
    # not the cost *gap*, which is a ratio.
    cost_scale = batch.get("cost_scale", None)
    if cost_scale is None:
        cost_scale = base / 1e6
    elif not torch.is_tensor(cost_scale):
        cost_scale = torch.as_tensor(float(cost_scale), dtype=dtype, device=device)
    else:
        cost_scale = cost_scale.to(device=device, dtype=dtype)
    if "Gen_cost_c1" in batch:
        out["Gen_cost_c1"] = _flat(batch["Gen_cost_c1"]).to(device=device, dtype=dtype) * cost_scale
    if "Gen_cost_c2" in batch:
        # Quadratic term, present in PGLib/OPFData where the cost is
        # c2 p^2 + c1 p + c0. The pandapower parquets carry only a linear
        # slope, so this stays optional.
        out["Gen_cost_c2"] = _flat(batch["Gen_cost_c2"]).to(device=device, dtype=dtype) * cost_scale ** 2
    for key in ("Bus_vmin", "Bus_vmax"):
        if key in batch:
            out[key] = _flat(batch[key]).to(device=device, dtype=dtype)
    if "Gen_controllable" in batch:
        out["Gen_controllable"] = _flat(batch["Gen_controllable"]).to(device=device) > 0
    if "Branch_rate_a" in batch:
        out["Branch_rate_a"] = _flat(batch["Branch_rate_a"]).to(device=device, dtype=dtype)
    if "Gen_box_injection" in batch:
        # Per-bus constant, carried as a vector so it rides the same collate path
        # as Gen_controllable rather than needing new scalar plumbing.
        out["gen_box_on_injection"] = bool(
            (_flat(batch["Gen_box_injection"]) > 0).any().item())
    return out


def implied_injection(Y: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """S = V * conj(Y V) for a block-diagonal batch. V is [1, N, 2] as (mag, angle)."""
    mag = V[..., 0]
    ang = V[..., 1]
    u = torch.polar(mag.to(torch.float64), ang.to(torch.float64)).squeeze(0)
    from collate_blockdiag_optimized_complex_columns import ybus_matvec
    # Y may be dense [N,N] / [1,N,N] or sparse COO [N,N]; ybus_matvec handles both.
    Yu = ybus_matvec(Y if Y.is_sparse else Y.to(torch.complex128), u)
    return u * torch.conj(Yu)


def opf_residual_and_violation(
    Y: torch.Tensor,
    V: torch.Tensor,
    S_load: torch.Tensor,
    space: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Split the network equation into an equality part and a limit part.

    Returns per-bus tensors:
      dp/dq            power-balance residual, valid only where not controllable
      vp/vq            amount by which the implied dispatch leaves its box,
                       valid only where controllable
    """
    S_implied = implied_injection(Y, V)
    S_load = _flat(S_load).to(S_implied.device)
    ctrl = space.get("Gen_controllable")
    if ctrl is None:
        ctrl = torch.zeros(S_implied.shape[0], dtype=torch.bool, device=S_implied.device)

    resid = S_implied - S_load.to(torch.complex128)
    dp = torch.where(ctrl, torch.zeros_like(resid.real), resid.real.abs())
    dq = torch.where(ctrl, torch.zeros_like(resid.imag), resid.imag.abs())

    # What the box bounds. When S_start is loads-only the free quantity is the
    # generation, i.e. injection minus load. The pandapower ACOPF-backbone files
    # instead store S_start as the *pre-OPF injection* (base generation minus
    # load), so there the difference is a redispatch delta whose range is
    # dominated by the slack correction -- on case118 that puts the box midpoint
    # at -28 pu, which is not a generation level any model can use. Bounding the
    # post-OPF injection itself is observable and physically meaningful in both
    # conventions.
    if bool(space.get("gen_box_on_injection", False)):
        gen_p = S_implied.real
        gen_q = S_implied.imag
    else:
        gen_p = resid.real
        gen_q = resid.imag
    p_min = space.get("Gen_p_min"); p_max = space.get("Gen_p_max")
    q_min = space.get("Gen_q_min"); q_max = space.get("Gen_q_max")
    if p_min is None:
        vp = torch.zeros_like(gen_p); vq = torch.zeros_like(gen_q)
    else:
        vp = torch.clamp(p_min - gen_p, min=0.0) + torch.clamp(gen_p - p_max, min=0.0)
        vq = torch.clamp(q_min - gen_q, min=0.0) + torch.clamp(gen_q - q_max, min=0.0)
        vp = torch.where(ctrl, vp, torch.zeros_like(vp))
        vq = torch.where(ctrl, vq, torch.zeros_like(vq))
    return {"dp": dp, "dq": dq, "vp": vp, "vq": vq,
            "gen_p": torch.where(ctrl, gen_p, torch.zeros_like(gen_p)), "ctrl": ctrl}


def voltage_band_violation(V: torch.Tensor, space: Dict[str, torch.Tensor]) -> torch.Tensor:
    vmin = space.get("Bus_vmin"); vmax = space.get("Bus_vmax")
    mag = V[..., 0].squeeze(0).to(torch.float64)
    if vmin is None or vmax is None:
        return torch.zeros_like(mag)
    return torch.clamp(vmin - mag, min=0.0) + torch.clamp(mag - vmax, min=0.0)


def dispatch_cost(gen_p: torch.Tensor, space: Dict[str, torch.Tensor]) -> torch.Tensor:
    c1 = space.get("Gen_cost_c1")
    if c1 is None:
        return torch.zeros((), dtype=gen_p.dtype, device=gen_p.device)
    cost = (c1 * gen_p).sum()
    c2 = space.get("Gen_cost_c2")
    if c2 is not None:
        cost = cost + (c2 * gen_p * gen_p).sum()
    return cost


def target_dispatch(batch: Dict[str, torch.Tensor], space: Dict[str, torch.Tensor],
                    device, dtype=torch.float64) -> Optional[torch.Tensor]:
    """Per-bus reference generation, as complex power in per-unit.

    Both pipelines can supply this. The OPF parquets store the full injection
    (`S_newton`) alongside the load-only input (`S_start`), so generation is
    their difference; OPFData additionally ships `S_gen_opf` directly.
    """
    base = _per_graph_base(batch, batch["sizes"], device, dtype) if "sizes" in batch else None
    if bool(space.get("gen_box_on_injection", False)) and "S_newton" in batch:
        # Same convention as the box: the reference is the post-OPF injection.
        gen = _flat(batch["S_newton"]).to(device)
    elif "S_gen_opf" in batch:
        gen = _flat(batch["S_gen_opf"]).to(device)
    elif "S_newton" in batch and "S_start" in batch:
        gen = (_flat(batch["S_newton"]) - _flat(batch["S_start"])).to(device)
    else:
        return None
    gen = gen.to(torch.complex128)
    ctrl = space.get("Gen_controllable")
    if ctrl is not None:
        gen = torch.where(ctrl, gen, torch.zeros_like(gen))
    return gen


def dispatch_loss(
    Y: torch.Tensor,
    V_pred: torch.Tensor,
    S_load: torch.Tensor,
    space: Dict[str, torch.Tensor],
    target_gen: torch.Tensor,
    *,
    form: str = "mse",
) -> torch.Tensor:
    """Supervise the dispatch *implied* by the predicted voltages.

    Differentiable through V for every model, including those with no generator
    head, so it applies uniformly. It does not, however, reach a head that the
    architecture detaches from the voltage path -- see `gen_head_loss`.
    """
    parts = opf_residual_and_violation(Y, V_pred, S_load, space)
    ctrl = parts["ctrl"]
    if int(ctrl.sum()) == 0:
        return torch.zeros((), dtype=torch.float32, device=V_pred.device)
    resid = implied_injection(Y, V_pred) - _flat(S_load).to(V_pred.device).to(torch.complex128)
    dp = (resid.real - target_gen.real)[ctrl].to(torch.float32)
    dq = (resid.imag - target_gen.imag)[ctrl].to(torch.float32)
    if form == "huber":
        z = torch.zeros_like(dp)
        return torch.nn.functional.huber_loss(dp, z) + torch.nn.functional.huber_loss(dq, z)
    return (dp * dp).mean() + (dq * dq).mean()


def gen_head_loss(gen_pred: torch.Tensor, target_gen: torch.Tensor,
                  gen_bus: torch.Tensor, *, form: str = "mse") -> torch.Tensor:
    """Supervise a model's own generator head against the reference dispatch.

    Needed because GridSFM builds its angle prior from `head_Pg` through a
    detached path: without this term that head receives no gradient at all, so
    the prior stays at its initialisation. `gen_pred` is [n_gen, 2] = (Pg, Qg)
    and `gen_bus` maps each generator node to its bus.
    """
    if gen_pred is None or gen_pred.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=target_gen.device)
    tgt = target_gen[gen_bus.to(torch.long)]
    dp = (gen_pred[:, 0].to(torch.float64) - tgt.real).to(torch.float32)
    dq = (gen_pred[:, 1].to(torch.float64) - tgt.imag).to(torch.float32)
    if form == "huber":
        z = torch.zeros_like(dp)
        return torch.nn.functional.huber_loss(dp, z) + torch.nn.functional.huber_loss(dq, z)
    return (dp * dp).mean() + (dq * dq).mean()


def opf_loss(
    Y: torch.Tensor,
    V_pred: torch.Tensor,
    S_load: torch.Tensor,
    space: Dict[str, torch.Tensor],
    *,
    balance_weight: float = 1.0,
    limit_weight: float = 1.0,
    band_weight: float = 0.0,
    form: str = "logcosh",
) -> torch.Tensor:
    """Physics term for the OPF task: equality where known, limits where free."""
    parts = opf_residual_and_violation(Y, V_pred, S_load, space)

    def shape(x):
        x = x.to(torch.float32)
        if form == "mse":
            return (x * x).mean()
        if form == "huber":
            return torch.nn.functional.huber_loss(x, torch.zeros_like(x))
        return (torch.log(torch.cosh(torch.clamp(x, -20.0, 20.0)) + 1e-12)).mean()

    loss = balance_weight * (shape(parts["dp"]) + shape(parts["dq"]))
    loss = loss + limit_weight * (shape(parts["vp"]) + shape(parts["vq"]))
    if band_weight:
        loss = loss + band_weight * shape(voltage_band_violation(V_pred, space))
    return loss


@torch.no_grad()
def opf_metrics(
    Y: torch.Tensor,
    V_pred: torch.Tensor,
    V_true: torch.Tensor,
    S_load: torch.Tensor,
    space: Dict[str, torch.Tensor],
    n_nodes: torch.Tensor,
) -> Dict[str, float]:
    """Report OPF quality: balance where enforceable, feasibility, and cost gap."""
    pred = opf_residual_and_violation(Y, V_pred, S_load, space)
    true = opf_residual_and_violation(Y, V_true, S_load, space)

    ctrl = pred["ctrl"]
    free = ~ctrl
    n_free = int(free.sum().item())

    def summarize(x, mask):
        if int(mask.sum().item()) == 0:
            return 0.0, 0.0
        sel = x[mask]
        return float(sel.max().item()), float(sel.mean().item())

    dp_inf, dp_mean = summarize(pred["dp"], free)
    dq_inf, dq_mean = summarize(pred["dq"], free)
    vp_max, vp_mean = summarize(pred["vp"], ctrl)
    vq_max, vq_mean = summarize(pred["vq"], ctrl)

    band = voltage_band_violation(V_pred, space)
    band_max = float(band.max().item()) if band.numel() else 0.0
    band_frac = float((band > 1e-9).to(torch.float64).mean().item()) if band.numel() else 0.0

    cost_pred = float(dispatch_cost(pred["gen_p"], space).item())
    cost_true = float(dispatch_cost(true["gen_p"], space).item())
    denom = abs(cost_true) if abs(cost_true) > 1e-9 else 1.0

    return {
        "dp_inf_pu": dp_inf, "dq_inf_pu": dq_inf,
        "dp_mean_pu": dp_mean, "dq_mean_pu": dq_mean,
        "n_free_buses": n_free, "n_ctrl_buses": int(ctrl.sum().item()),
        "p_limit_max_pu": vp_max, "p_limit_mean_pu": vp_mean,
        "q_limit_max_pu": vq_max, "q_limit_mean_pu": vq_mean,
        "vband_max_pu": band_max, "vband_frac": band_frac,
        "cost_pred": cost_pred, "cost_true": cost_true,
        "cost_gap_rel": (cost_pred - cost_true) / denom,
    }


def format_opf_metrics(m: Dict[str, float]) -> str:
    return (
        f"balance(dPinf {m['dp_inf_pu']:.3e}, dQinf {m['dq_inf_pu']:.3e} pu over "
        f"{m['n_free_buses']:.0f} free buses) "
        f"limits(P {m['p_limit_max_pu']:.3e}, Q {m['q_limit_max_pu']:.3e} pu over "
        f"{m['n_ctrl_buses']:.0f} ctrl buses) "
        f"vband(max {m['vband_max_pu']:.3e}, frac {m['vband_frac']:.3f}) "
        f"cost(gap {m['cost_gap_rel']*100:.2f}%)"
    )
