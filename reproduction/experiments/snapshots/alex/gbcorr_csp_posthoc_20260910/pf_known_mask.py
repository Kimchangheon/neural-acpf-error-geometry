"""Hold the quantities power flow specifies, instead of predicting them.

In AC power flow |V| is given at PV and slack buses and theta is given at the
slack; only the remaining unknowns are solved for.  PIGNN enforces this inside
its iteration (``dv.masked_fill(slack | pv, 0)``), which is why its PV and slack
buses are exact for free.  GridSFM feeds the setpoint into its voltage head and
GraphKit writes it into the bus row, but neither *holds* it -- and LUMINA does
not even carry it on the bus.

This applies the same hold at the adapter boundary, so it can be switched on for
any of the three without touching a vendored model.
"""
from typing import Optional

import torch


def apply_known_v(V: torch.Tensor, bus_type: torch.Tensor,
                  V_start: torch.Tensor) -> torch.Tensor:
    """Overwrite the specified entries of a prediction with their setpoints.

    V, V_start : [..., N, 2] as (magnitude, angle)
    bus_type   : [N] with 1 = slack, 2 = PV, anything else = PQ

    Nothing here is a solved quantity: |V| at PV/slack and theta at slack are
    inputs to power flow, so this leaks nothing.  Q at PV, and P and Q at the
    slack, are what the solver finds and are left untouched.
    """
    # bus_type and V_start usually arrive on the CPU while the prediction is on
    # the GPU; move both, not just V_start, or the masks land on the wrong device.
    bt = bus_type.reshape(-1).to(V.device)
    is_slack = bt == 1
    is_pv = bt == 2
    vs = V_start.reshape(-1, 2).to(device=V.device, dtype=V.dtype)
    shape = V.shape
    v = V.reshape(-1, 2).clone()
    n = bt.numel()
    if v.shape[0] % n:
        raise ValueError(f"prediction has {v.shape[0]} rows, not a multiple of {n} buses")
    reps = v.shape[0] // n
    hold_mag = (is_slack | is_pv).repeat(reps)
    hold_ang = is_slack.repeat(reps)
    if vs.shape[0] != v.shape[0]:
        vs = vs.repeat(v.shape[0] // vs.shape[0], 1)
    v[:, 0] = torch.where(hold_mag, vs[:, 0], v[:, 0])
    v[:, 1] = torch.where(hold_ang, vs[:, 1], v[:, 1])
    return v.reshape(shape)
