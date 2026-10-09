"""Metrics that say whether a surrogate tracks the solver or collapses to a constant.

Motivated by GridSFM's own critique of MAE/MAPE (arXiv white paper, section 6.2,
"Beyond MAE"):

  * MAE is small whenever the quantity has a narrow natural range. Bus voltage
    magnitude in pu only spans ~0.95-1.10, so a flat predictor of V = 1.0 already
    posts a tiny MAE while carrying no information about the voltage profile.
  * MAPE is dominated by the small-denominator tail.
  * Neither says whether the model follows the solver's variation.

Their fix, reproduced here: a per-channel pooled linear regression of predicted
against reference values. Slope 1 means the model tracks the reference spread
one-for-one; slope well below 1 means it is collapsing toward the mean. R^2 says
how much of the reference variance is explained.

This is not academic for this project: a GridSFM run on OPFData case118 was
found to have collapsed its magnitude head only because two independently
initialised runs reported a |V| RMSE identical to five significant figures.
Slope ~ 0 with R^2 ~ 0 would have shown it directly, on a single run.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1).to(torch.float64)


def regression_diag(pred: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    """Least-squares fit of pred = a * ref + b, pooled over everything given.

    Returns slope, intercept, Pearson R, R^2, and the two standard deviations
    whose ratio is the blunt version of the same question.
    """
    p, r = _flat(pred), _flat(ref)
    n = p.numel()
    if n < 2:
        return {"slope": float("nan"), "intercept": float("nan"),
                "R": float("nan"), "R2": float("nan"),
                "std_pred": float("nan"), "std_ref": float("nan"), "n": int(n)}
    pm, rm = p.mean(), r.mean()
    dp, dr = p - pm, r - rm
    var_r = float((dr * dr).mean())
    var_p = float((dp * dp).mean())
    cov = float((dp * dr).mean())
    slope = cov / var_r if var_r > 0 else float("nan")
    denom = (var_p * var_r) ** 0.5
    if denom > 0:
        R = cov / denom
    elif var_p == 0.0:
        # A constant prediction explains none of the reference variance. R is
        # formally undefined here, but reporting nan invites reading it as a
        # failed computation when it is in fact the collapse this diagnostic
        # exists to detect; 0 is the meaningful value.
        R = 0.0
    else:
        R = float("nan")
    return {
        "slope": float(slope),
        "intercept": float(rm.new_tensor(0.0) + (pm - slope * rm)) if var_r > 0 else float("nan"),
        "R": float(R),
        "R2": float(R * R) if R == R else float("nan"),
        "std_pred": float(var_p ** 0.5),
        "std_ref": float(var_r ** 0.5),
        "n": int(n),
    }


def angle_wrap(d: torch.Tensor) -> torch.Tensor:
    """Wrap an angle difference into (-pi, pi]."""
    return torch.atan2(torch.sin(d), torch.cos(d))


def voltage_mae(V_pred: torch.Tensor, V_ref: torch.Tensor) -> Dict[str, float]:
    """MAE alongside RMSE for magnitude and angle.

    Reported together on purpose: MAE describes the typical bus, RMSE is pulled
    by the tail, and their ratio is a cheap read on how heavy that tail is.
    Angle error is wrapped before averaging, so a prediction near -pi against a
    reference near +pi counts as small rather than as 2*pi.
    """
    vp, vr = V_pred[..., 0].to(torch.float64), V_ref[..., 0].to(torch.float64)
    ap, ar = V_pred[..., 1].to(torch.float64), V_ref[..., 1].to(torch.float64)
    dmag = (vp - vr).abs()
    dang = angle_wrap(ap - ar).abs()
    deg = 180.0 / torch.pi
    return {
        "mae_vmag_pu": float(dmag.mean()),
        "rmse_vmag_pu": float((dmag * dmag).mean() ** 0.5),
        "mae_theta_deg": float(dang.mean() * deg),
        "rmse_theta_deg": float(((dang * dang).mean() ** 0.5) * deg),
        "medae_vmag_pu": float(dmag.median()),
        "medae_theta_deg": float(dang.median() * deg),
    }


def voltage_regression(V_pred: torch.Tensor, V_ref: torch.Tensor) -> Dict[str, Dict[str, float]]:
    """Per-channel regression diagnostics for the voltage state."""
    return {
        "vmag": regression_diag(V_pred[..., 0], V_ref[..., 0]),
        # Angles are regressed on their sine and cosine rather than the raw
        # value: a wrapped angle is circular, and a raw fit would be corrupted
        # by any pair straddling the branch cut.
        "theta_sin": regression_diag(torch.sin(V_pred[..., 1]), torch.sin(V_ref[..., 1])),
        "theta_cos": regression_diag(torch.cos(V_pred[..., 1]), torch.cos(V_ref[..., 1])),
    }


def format_diagnostics(mae: Dict[str, float],
                       reg: Dict[str, Dict[str, float]]) -> str:
    v, ts, tc = reg["vmag"], reg["theta_sin"], reg["theta_cos"]
    return (
        f"MAE |V| {mae['mae_vmag_pu']:.4e} pu, theta {mae['mae_theta_deg']:.4e} deg | "
        f"median |V| {mae['medae_vmag_pu']:.4e}, theta {mae['medae_theta_deg']:.4e} deg\n"
        f"Regression  |V| : slope {v['slope']:.4f} R {v['R']:.4f} R2 {v['R2']:.4f} "
        f"(std pred {v['std_pred']:.4e} vs ref {v['std_ref']:.4e})\n"
        f"Regression sin(theta): slope {ts['slope']:.4f} R2 {ts['R2']:.4f} | "
        f"cos(theta): slope {tc['slope']:.4f} R2 {tc['R2']:.4f}"
    )


# ---------------------------------------------------------------------------
# Per-bus (scenario-only) variants.
#
# GridSFM's section 6.2 runs its regression pooled over every bus and scenario,
# and for a multi-grid foundation model that is the right pooling: the model
# has never seen the grid, so reproducing its static voltage profile is part of
# what it must predict.  Every campaign in this project instead trains one
# model per grid on that same grid, where the static per-bus profile is
# memorisable from the training split and therefore free.  Pooled R^2 then
# rewards recalling which buses sit high, not tracking the scenario.
#
# The paper's own out-of-distribution table is the evidence: moving the
# released checkpoint to an unseen grid drops V from slope 0.874 / R^2 0.89 to
# slope 0.273 / R^2 0.111 while theta, Pg and cost survive.  A V channel that
# were genuinely scenario-driven would not collapse on a topology change.
#
# So both are reported.  The pooled numbers stay comparable with the paper;
# the per-bus numbers below subtract each bus's own mean from prediction and
# reference alike, leaving only the scenario-to-scenario variation, which is
# the question a single-grid surrogate is actually being asked.
# ---------------------------------------------------------------------------


def _reshape_per_bus(x: torch.Tensor, n_bus: int) -> Optional[torch.Tensor]:
    """[.., n_scen * n_bus] -> [n_scen, n_bus], or None if it does not divide."""
    flat = x.reshape(-1)
    if n_bus <= 0 or flat.numel() % n_bus:
        return None
    return flat.reshape(-1, n_bus)


def regression_diag_per_bus(pred: torch.Tensor, ref: torch.Tensor,
                            n_bus: int) -> Dict[str, float]:
    """regression_diag on each bus's deviation from its own mean."""
    p = _reshape_per_bus(pred.to(torch.float64), n_bus)
    r = _reshape_per_bus(ref.to(torch.float64), n_bus)
    if p is None or r is None or p.shape[0] < 2:
        return {"slope": float("nan"), "intercept": float("nan"),
                "R": float("nan"), "R2": float("nan"),
                "std_pred": float("nan"), "std_ref": float("nan"), "n": 0}
    return regression_diag(p - p.mean(dim=0), r - r.mean(dim=0))


def circular_mean(theta: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.atan2(torch.sin(theta).mean(dim=dim), torch.cos(theta).mean(dim=dim))


def angle_regression_per_bus(theta_pred: torch.Tensor, theta_ref: torch.Tensor,
                             n_bus: int) -> Dict[str, float]:
    """Regression on the wrapped angle deviation from each bus's circular mean.

    Once each bus is centred the deviations are small, so the branch cut that
    forced the sin/cos treatment in the pooled version is no longer in play and
    the angle can be regressed directly, in radians.
    """
    p = _reshape_per_bus(theta_pred.to(torch.float64), n_bus)
    r = _reshape_per_bus(theta_ref.to(torch.float64), n_bus)
    if p is None or r is None or p.shape[0] < 2:
        return {"slope": float("nan"), "intercept": float("nan"),
                "R": float("nan"), "R2": float("nan"),
                "std_pred": float("nan"), "std_ref": float("nan"), "n": 0}
    dp = angle_wrap(p - circular_mean(p))
    dr = angle_wrap(r - circular_mean(r))
    return regression_diag(dp, dr)


def sse_r2_per_bus(V_pred: torch.Tensor, V_ref: torch.Tensor,
                   n_bus: int) -> Dict[str, float]:
    """1 - SSE/SST against the per-bus constant predictor.

    This is the quantity the report tables call R^2.  It is NOT the regression
    R^2 above: it charges the model for bias and for amplitude error as well as
    for failing to track, and the two differ by (slope-1)^2 + bias^2/sigma^2.
    Both are reported so that the gap, which is exactly the miscalibration, is
    visible instead of being attributed to collapse.
    """
    vp = _reshape_per_bus(V_pred[..., 0].to(torch.float64), n_bus)
    vr = _reshape_per_bus(V_ref[..., 0].to(torch.float64), n_bus)
    tp = _reshape_per_bus(V_pred[..., 1].to(torch.float64), n_bus)
    tr = _reshape_per_bus(V_ref[..., 1].to(torch.float64), n_bus)
    if vp is None or vr is None or vp.shape[0] < 2:
        return {"vm_r2_sse_per_bus": float("nan"), "va_r2_sse_per_bus": float("nan"),
                "vm_sigma_per_bus": float("nan"), "va_sigma_per_bus_deg": float("nan")}
    vm_dev = vr - vr.mean(dim=0)
    va_dev = angle_wrap(tr - circular_mean(tr))
    vm_sst = float((vm_dev * vm_dev).sum())
    va_sst = float((va_dev * va_dev).sum())
    vm_sse = float(((vp - vr) ** 2).sum())
    va_sse = float((angle_wrap(tp - tr) ** 2).sum())
    deg = 180.0 / torch.pi
    n = float(vm_dev.numel())
    return {
        "vm_r2_sse_per_bus": 1.0 - vm_sse / max(vm_sst, 1e-30),
        "va_r2_sse_per_bus": 1.0 - va_sse / max(va_sst, 1e-30),
        "vm_sigma_per_bus": (vm_sst / n) ** 0.5,
        "va_sigma_per_bus_deg": float((va_sst / n) ** 0.5 * deg),
    }


def voltage_regression_per_bus(V_pred: torch.Tensor, V_ref: torch.Tensor,
                               n_bus: int) -> Dict[str, Dict[str, float]]:
    out = {
        "vmag": regression_diag_per_bus(V_pred[..., 0], V_ref[..., 0], n_bus),
        "theta": angle_regression_per_bus(V_pred[..., 1], V_ref[..., 1], n_bus),
    }
    out["sse"] = sse_r2_per_bus(V_pred, V_ref, n_bus)
    return out


def branch_angle_diagnostics(theta_pred: torch.Tensor, theta_ref: torch.Tensor,
                             f: torch.Tensor, t: torch.Tensor,
                             status: torch.Tensor, n_bus: int) -> Dict[str, Dict[str, float]]:
    """Per-edge angle-difference error and the gauge decomposition, in degrees.

    Branch flows depend on theta_i - theta_j, never on the absolute bus angle,
    so a model whose angles carry one scenario-dependent common offset posts a
    large per-bus theta RMSE while predicting every flow exactly.  Per-bus RMSE
    alone cannot separate that from real error.  GridSFM's paper makes the same
    point about a competitor ("per-bus theta MAE is inflated by an uncontrolled
    angle-gauge drift ... per-edge dtheta_ij MAE is 0.146 deg").

    Returns per_bus, per_bus_degauged (after removing each scenario's circular
    mean error), gauge (that offset), and per_edge over in-service branches.
    """
    deg = 180.0 / torch.pi
    tp = theta_pred.reshape(-1).to(torch.float64)
    tr = theta_ref.reshape(-1).to(torch.float64)
    d_bus = angle_wrap(tp - tr)

    st = status.reshape(-1).bool()
    fi, ti = f.reshape(-1).long(), t.reshape(-1).long()
    d_edge = angle_wrap(angle_wrap(tp[fi] - tp[ti]) - angle_wrap(tr[fi] - tr[ti]))[st]

    if n_bus > 0 and d_bus.numel() % n_bus == 0:
        per_scen = d_bus.reshape(-1, n_bus)
        gauge_s = circular_mean(per_scen, dim=1)
        degauged = angle_wrap(per_scen - gauge_s.unsqueeze(1)).reshape(-1)
    else:
        gauge_s = torch.zeros(1, dtype=torch.float64)
        degauged = d_bus

    def stats(x):
        e = x.abs() * deg
        # torch.quantile refuses inputs above 2**24 elements, and a 31-grid
        # corpus reaches 27M bus-scenario entries, so the percentile is taken
        # from a sort instead.  median goes the same way for consistency.
        srt = e.sort().values
        n = srt.numel()
        q = lambda f: float(srt[min(n - 1, int(f * (n - 1) + 0.5))])  # noqa: E731
        return {"mae": float(e.mean()), "rmse": float((e * e).mean() ** 0.5),
                "median": q(0.5), "p95": q(0.95),
                "max": float(srt[-1]), "n": int(n)}

    return {"per_bus": stats(d_bus), "per_bus_degauged": stats(degauged),
            "gauge": stats(gauge_s), "per_edge": stats(d_edge)}


def fit_per_bus_calibration(Vp_train, Vr_train, n_bus):
    """Per-bus mean alignment, fitted on the training split.

    For each bus, the mean prediction is aligned to the mean reference:
    dvm_i = mean_s(|V|_pred) - mean_s(|V|_ref), and the circular equivalent for
    theta.  The same fixed offset is then subtracted on unseen test scenarios.
    It uses no test data, so it is a correction any deployed model could carry,
    and it removes exactly the bias term of the decomposition
    RMSE^2 = bias^2 + (a-1)^2 sigma^2 + sigma_eps^2 -- which on these grids is
    the term that dominates.

    Returns (dvm[n_bus], dtheta[n_bus]) or None when n_bus does not divide.
    """
    vp = _reshape_per_bus(Vp_train[..., 0].to(torch.float64), n_bus)
    vr = _reshape_per_bus(Vr_train[..., 0].to(torch.float64), n_bus)
    tp = _reshape_per_bus(Vp_train[..., 1].to(torch.float64), n_bus)
    tr = _reshape_per_bus(Vr_train[..., 1].to(torch.float64), n_bus)
    if vp is None or vr is None or vp.shape[0] < 2:
        return None
    dvm = vp.mean(dim=0) - vr.mean(dim=0)
    dth = circular_mean(angle_wrap(tp - tr), dim=0)
    return dvm, dth


def apply_per_bus_calibration(V, offset, n_bus):
    """Subtract a fitted per-bus offset from a packed [1, n*n_bus, 2] state."""
    if offset is None:
        return V
    dvm, dth = offset
    reps = V[..., 0].numel() // n_bus
    out = V.clone()
    out[..., 0] = out[..., 0] - dvm.repeat(reps).reshape(out[..., 0].shape)
    out[..., 1] = angle_wrap(out[..., 1] - dth.repeat(reps).reshape(out[..., 1].shape))
    return out


def residual_stats(dp, dq):
    """Summaries of |dP|,|dQ| over their masked bus-scenario entries, in pu."""
    out = {}
    for tag, x in (("dp", dp), ("dq", dq)):
        srt = x.sort().values
        n = srt.numel()
        q = lambda f: float(srt[min(n - 1, int(f * (n - 1) + 0.5))])  # noqa: E731
        out.update({
            f"mean_{tag}": float(x.mean()), f"median_{tag}": q(0.5),
            f"p95_{tag}": q(0.95), f"p99_{tag}": q(0.99),
            f"rmse_{tag}": float((x * x).mean() ** 0.5), f"max_{tag}": float(srt[-1]),
            f"frac_{tag}_le_1e-2": float((x <= 1e-2).double().mean()),
            f"frac_{tag}_le_1e-3": float((x <= 1e-3).double().mean()),
            f"n_{tag}": int(n),
        })
    return out


def ac_residuals(Y, V, S_set, bus_type, matvec):
    """|dP| on non-slack buses and |dQ| on PQ buses, from S = V conj(YV).

    Recomputed in complex128 from the packed block-diagonal Y-bus, exactly as
    the standalone residual diagnostic does, so a calibrated state can be
    scored on the same footing as the raw one.  Bus-type convention is the
    parquet's: 1 = slack, 2 = PV, anything else PQ.
    """
    v = V[..., 0].reshape(-1).to(torch.float64)
    th = V[..., 1].reshape(-1).to(torch.float64)
    Vc = (v * torch.exp(1j * th)).to(torch.complex128)
    Sc = Vc * matvec(Y, Vc).conj()
    bt = bus_type.reshape(-1)
    p_mask = bt != 1
    q_mask = (bt != 1) & (bt != 2)
    dp = (S_set.real - Sc.real).abs()[p_mask]
    dq = (S_set.imag - Sc.imag).abs()[q_mask]
    return dp, dq
