"""The training-solution subspace.

The paper's observation is that a surrogate's voltage error concentrates in a
low-dimensional subspace spanned by the solutions the grid actually produces.
This module estimates that subspace from reference states on the **training
split only** — fitting it on test data would make the correction unfalsifiable.

Magnitude and angle get separate bases. They are different physical quantities
with different units and different conditioning, and a joint SVD would let the
angle block, which varies over radians, dominate the magnitude block, which
varies over a few percent of a per-unit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor

from ._util import circular_mean, wrap_angle


@dataclass(frozen=True)
class Basis:
    """A rank-`k` subspace of reference voltage states, plus its centre.

    Attributes
    ----------
    u_mag, u_ang:
        ``[n_bus, k]`` orthonormal bases for the magnitude and angle channels.
    mean_mag, mean_ang:
        ``[n_bus]`` per-bus centres. ``mean_ang`` is a circular mean.
    energy_mag, energy_ang:
        Fraction of reference variance the `k` directions capture. Useful for
        choosing `k`: on a transmission grid the magnitude channel typically
        reaches 98--99 % well before k = 16.
    """

    u_mag: Tensor
    u_ang: Tensor
    mean_mag: Tensor
    mean_ang: Tensor
    energy_mag: float
    energy_ang: float

    @property
    def rank(self) -> int:
        return int(self.u_mag.shape[1])

    @property
    def n_bus(self) -> int:
        return int(self.mean_mag.numel())

    def to(self, *, device=None, dtype=None) -> "Basis":
        def m(t: Tensor) -> Tensor:
            return t.to(device=device or t.device, dtype=dtype or t.dtype)

        return Basis(m(self.u_mag), m(self.u_ang), m(self.mean_mag),
                     m(self.mean_ang), self.energy_mag, self.energy_ang)

    def state_dict(self) -> dict:
        return {"u_mag": self.u_mag, "u_ang": self.u_ang,
                "mean_mag": self.mean_mag, "mean_ang": self.mean_ang,
                "energy_mag": self.energy_mag, "energy_ang": self.energy_ang}

    @staticmethod
    def from_state_dict(d: dict) -> "Basis":
        return Basis(d["u_mag"], d["u_ang"], d["mean_mag"], d["mean_ang"],
                     float(d["energy_mag"]), float(d["energy_ang"]))


def fit_basis(reference_states: Tensor, rank: int) -> Basis:
    """Fit the rank-`k` training-solution subspace.

    Parameters
    ----------
    reference_states:
        ``[n_scenarios, n_bus, 2]`` solver solutions from the **training**
        split, last axis ``(magnitude, angle)`` with the angle in radians.
    rank:
        Requested `k`. Clipped to the data: it cannot exceed the number of
        scenarios or the number of buses.

    Notes
    -----
    Angles are centred on the per-bus circular mean and wrapped before the SVD.
    Without that, a bus whose angle sits near ±π contributes a spurious
    direction of magnitude 2π that has nothing to do with the physics.
    """
    states = _check_states(reference_states, name="reference_states")
    vm = states[..., 0].to(torch.float64)
    th = states[..., 1].to(torch.float64)

    mean_mag = vm.mean(dim=0)
    mean_ang = circular_mean(th, dim=0)
    d_mag = vm - mean_mag
    d_ang = wrap_angle(th - mean_ang)

    k = int(min(rank, d_mag.shape[0], d_mag.shape[1]))
    if k < 1:
        raise ValueError(f"rank must be at least 1, got {rank}")

    u_mag, energy_mag = _principal_directions(d_mag, k)
    u_ang, energy_ang = _principal_directions(d_ang, k)
    return Basis(u_mag, u_ang, mean_mag, mean_ang, energy_mag, energy_ang)


def _principal_directions(deviations: Tensor, k: int) -> Tuple[Tensor, float]:
    """Top-`k` right singular vectors, and the variance fraction they capture."""
    _, sv, vh = torch.linalg.svd(deviations, full_matrices=False)
    total = (sv ** 2).sum()
    energy = float(((sv ** 2).cumsum(0) / total)[k - 1]) if total > 0 else 0.0
    return vh[:k].T.contiguous(), energy


def _check_states(states: Tensor, *, name: str) -> Tensor:
    if states.dim() != 3 or states.shape[-1] != 2:
        raise ValueError(
            f"{name} must be [n_scenarios, n_bus, 2] with the last axis "
            f"(magnitude, angle); got shape {tuple(states.shape)}."
        )
    return states
