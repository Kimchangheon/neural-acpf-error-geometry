"""Step P: project onto the training-solution subspace.

    v_proj = mean + U Uᵀ (v - mean)

`U` is orthonormal, so this is an exact orthogonal projection: idempotent, and a
fixed linear operator, so gradients pass through it unchanged if it is used
inside a model rather than after one. It is rank-deficient by construction —
that is the point. Directions the training solutions never exhibit receive no
weight in the output.

Angles are projected in the same way about their circular mean, and re-wrapped.

Three scopes are available, and the paper uses the first:

``full``
    Project every bus coordinate.
``restore_known``
    Project everything, then write the prescribed quantities back. In the
    paper's setting this is numerically identical to ``full`` — the projection
    does not move the prescribed coordinates — but it is the safe choice if
    your basis is fitted differently.
``unknown_only``
    Fit and project only the coordinates power flow leaves free. This is a
    genuinely different operator and gives different numbers.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from ._util import wrap_angle
from .basis import Basis

#: Bus-type codes, following the MATPOWER/pandapower convention.
SLACK, PV = 1, 2


def project(states: Tensor, basis: Basis) -> Tensor:
    """Project a ``[..., n_bus, 2]`` state onto the subspace."""
    b = basis.to(device=states.device, dtype=states.dtype)
    _check_bus_count(states, b.n_bus)
    lead = states.shape[:-2]
    flat = states.reshape(-1, b.n_bus, 2)

    d_mag = flat[..., 0] - b.mean_mag
    d_ang = wrap_angle(flat[..., 1] - b.mean_ang)
    mag = b.mean_mag + (d_mag @ b.u_mag) @ b.u_mag.T
    ang = wrap_angle(b.mean_ang + (d_ang @ b.u_ang) @ b.u_ang.T)
    return torch.stack([mag, ang], dim=-1).reshape(*lead, b.n_bus, 2)


def unknown_masks(bus_type: Tensor) -> Tuple[Tensor, Tensor]:
    """Which coordinates AC power flow leaves free.

    Magnitude is prescribed at PV and slack buses; angle is prescribed at the
    slack. Returns ``(free_magnitude, free_angle)`` boolean masks over buses.
    """
    bt = bus_type.reshape(-1).to(torch.long)
    return (bt != SLACK) & (bt != PV), bt != SLACK


def fit_basis_unknown_only(reference_states: Tensor, rank: int,
                           bus_type: Tensor) -> Tuple[Basis, Tuple[Tensor, Tensor]]:
    """Fit a basis on the free coordinates only.

    The returned basis lives in the reduced coordinate space, so it must be
    used with `project_unknown_only` rather than `project`.
    """
    from .basis import fit_basis  # local import keeps the module graph acyclic

    free_mag, free_ang = unknown_masks(bus_type)
    n_free = int(max(free_mag.sum(), free_ang.sum()))
    if n_free == 0:
        raise ValueError("every coordinate is prescribed; nothing to project.")
    # Magnitude and angle have different free sets, so they are fitted apart and
    # recombined into one Basis whose two channels have different lengths.
    mag_basis = fit_basis(
        torch.stack([reference_states[..., free_mag, 0],
                     torch.zeros_like(reference_states[..., free_mag, 0])], dim=-1), rank)
    ang_basis = fit_basis(
        torch.stack([torch.zeros_like(reference_states[..., free_ang, 1]),
                     reference_states[..., free_ang, 1]], dim=-1), rank)
    basis = Basis(mag_basis.u_mag, ang_basis.u_ang,
                  mag_basis.mean_mag, ang_basis.mean_ang,
                  mag_basis.energy_mag, ang_basis.energy_ang)
    return basis, (free_mag, free_ang)


def project_unknown_only(states: Tensor, basis: Basis,
                         masks: Tuple[Tensor, Tensor]) -> Tensor:
    """Project only the free coordinates, leaving the prescribed ones as given."""
    b = basis.to(device=states.device, dtype=states.dtype)
    free_mag, free_ang = (m.to(states.device) for m in masks)
    out = states.clone()

    d_mag = states[..., free_mag, 0] - b.mean_mag
    out[..., free_mag, 0] = b.mean_mag + (d_mag @ b.u_mag) @ b.u_mag.T

    d_ang = wrap_angle(states[..., free_ang, 1] - b.mean_ang)
    out[..., free_ang, 1] = wrap_angle(b.mean_ang + (d_ang @ b.u_ang) @ b.u_ang.T)
    return out


def restore_setpoints(states: Tensor, bus_type: Tensor,
                      setpoints: Tensor) -> Tensor:
    """Write the prescribed quantities back after projecting.

    `setpoints` is ``[..., n_bus, 2]`` carrying the values the problem
    specifies: magnitude at PV and slack, angle at the slack. Anything else in
    it is ignored.
    """
    bt = bus_type.reshape(-1).to(states.device)
    if bt.numel() != states.shape[-2]:
        raise ValueError(
            f"bus_type has {bt.numel()} entries but the states have "
            f"{states.shape[-2]} buses."
        )
    sp = setpoints.reshape(-1, states.shape[-2], 2).to(
        device=states.device, dtype=states.dtype)
    if sp.shape[0] == 1:
        sp = sp.expand(states.reshape(-1, states.shape[-2], 2).shape[0], -1, -1)

    hold_mag = ((bt == SLACK) | (bt == PV))
    hold_ang = bt == SLACK
    out = states.reshape(-1, states.shape[-2], 2).clone()
    out[..., hold_mag, 0] = sp[..., hold_mag, 0]
    out[..., hold_ang, 1] = sp[..., hold_ang, 1]
    return out.reshape(states.shape)


def _check_bus_count(states: Tensor, n_bus: int) -> None:
    if states.dim() < 2 or states.shape[-1] != 2:
        raise ValueError(
            f"states must be [..., n_bus, 2]; got {tuple(states.shape)}.")
    if states.shape[-2] != n_bus:
        raise ValueError(
            f"the basis was fitted on {n_bus} buses but these states have "
            f"{states.shape[-2]}; it does not belong to this grid."
        )
