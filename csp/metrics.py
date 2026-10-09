"""The metrics the paper reports.

Two of them are worth explaining before use.

**Power balance (PB).** The per-bus mismatch between the specified injection and
the injection the predicted voltage state actually produces,

    ΔS = S_spec − V ⊙ conj(Y V),    PB_i = sqrt((m_P,i ΔP_i)² + (m_Q,i ΔQ_i)²)

scored only where the quantity is specified: ΔP everywhere except the slack,
ΔQ at PQ buses only. Elsewhere the solver is free to choose, so a residual
there is not an error. Computed in complex128 — in float32 the mismatch of a
good surrogate is at the noise floor of the matrix–vector product itself.

**Within-bus R² and slope.** Voltage magnitudes on a transmission grid sit in a
narrow band around 1 pu, so a pooled R² is dominated by the between-bus spread
and a model that predicts each bus's mean scores well. `within_bus_r2_slope`
centres each bus on its own mean first, so it measures what is actually of
interest: whether the model tracks the scenario at that bus.

The slope `a_w` is the amplitude of that response. Reading the two together is
what separates "removed a wrong direction" (R² up, slope unchanged) from
"recovered missing response" (both up).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import Tensor

from ._util import angle_difference
from .projection import PV, SLACK


def power_balance(states: Tensor, ybus: Tensor, s_specified: Tensor,
                  bus_type: Tensor) -> Tensor:
    """Per-bus power-balance residual, ``[n_scenarios, n_bus]``, in per-unit.

    Parameters
    ----------
    states:
        ``[n_scenarios, n_bus, 2]`` voltage state.
    ybus:
        ``[n_bus, n_bus]`` complex bus admittance matrix, dense or sparse.
    s_specified:
        ``[n_scenarios, n_bus]`` complex specified injection.
    bus_type:
        ``[n_bus]``; 1 = slack, 2 = PV, anything else = PQ.
    """
    if states.dim() != 3 or states.shape[-1] != 2:
        raise ValueError(
            f"states must be [n_scenarios, n_bus, 2]; got {tuple(states.shape)}.")
    n_scen, n_bus, _ = states.shape
    y = ybus.to(torch.complex128)
    if y.shape[-2:] != (n_bus, n_bus):
        raise ValueError(
            f"ybus is {tuple(y.shape)} but the states have {n_bus} buses.")

    v = (states[..., 0].to(torch.float64)
         * torch.exp(1j * states[..., 1].to(torch.float64))).to(torch.complex128)
    yv = (torch.sparse.mm(y, v.T).T if y.is_sparse else v @ y.transpose(-2, -1))
    s_calc = v * yv.conj()

    s_spec = s_specified.reshape(n_scen, n_bus).to(torch.complex128)
    d_p = s_spec.real - s_calc.real
    d_q = s_spec.imag - s_calc.imag

    bt = bus_type.reshape(-1).to(states.device)
    m_p = (bt != SLACK).to(torch.float64)
    m_q = ((bt != SLACK) & (bt != PV)).to(torch.float64)
    return torch.sqrt((d_p * m_p) ** 2 + (d_q * m_q) ** 2)


def voltage_rmse(states: Tensor, reference: Tensor) -> float:
    """RMSE of the magnitude channel, in per-unit."""
    d = states[..., 0].to(torch.float64) - reference[..., 0].to(torch.float64)
    return float(torch.sqrt(d.square().mean()))


def angle_rmse_deg(states: Tensor, reference: Tensor) -> float:
    """RMSE of the angle channel, in degrees, wrapped."""
    d = angle_difference(states[..., 1].to(torch.float64),
                         reference[..., 1].to(torch.float64))
    return math.degrees(float(torch.sqrt(d.square().mean())))


def within_bus_r2_slope(states: Tensor, reference: Tensor) -> Dict[str, float]:
    """Scenario-tracking R² and slope for the magnitude channel.

    Each bus is centred on its own mean across scenarios before pooling, so
    the between-bus spread cannot inflate the score.
    """
    p = states[..., 0].to(torch.float64)
    r = reference[..., 0].to(torch.float64)
    if p.shape != r.shape:
        raise ValueError(
            f"states {tuple(p.shape)} and reference {tuple(r.shape)} differ.")
    dp = p - p.mean(dim=0, keepdim=True)
    dr = r - r.mean(dim=0, keepdim=True)
    var_p = float((dp * dp).sum())
    var_r = float((dr * dr).sum())
    cov = float((dp * dr).sum())
    return {"within_r2": (cov * cov / (var_p * var_r)) if var_p > 0 and var_r > 0 else 0.0,
            "within_slope": (cov / var_r) if var_r > 0 else float("nan")}


def evaluate(states: Tensor, reference: Tensor, *,
             ybus: Optional[Tensor] = None,
             s_specified: Optional[Tensor] = None,
             bus_type: Optional[Tensor] = None) -> Dict[str, float]:
    """Every metric the paper's tables report, for one set of states.

    Power-balance entries are included only when `ybus`, `s_specified` and
    `bus_type` are all supplied.
    """
    out = {"vmag_rmse": voltage_rmse(states, reference),
           "angle_rmse_deg": angle_rmse_deg(states, reference)}
    out.update(within_bus_r2_slope(states, reference))
    if ybus is not None and s_specified is not None and bus_type is not None:
        pb = power_balance(states, ybus, s_specified, bus_type)
        out["mean_pb"] = float(pb.mean(dim=1).mean())
        out["max_pb"] = float(pb.max())
    return out
