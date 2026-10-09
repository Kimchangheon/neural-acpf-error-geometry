"""Voltage-manifold basis and projection, shared by the model and the scorer.

The AC residual is paid on the component of the voltage error that is not a
profile the grid produces, and is uncorrelated with the component that is
(measured on GBnetwork: corr +0.97 against -0.16 over eight configurations).
Projecting a prediction onto the span of the training-split reference variation
removes the first by construction; post-hoc, on a trained PIGNN checkpoint and
with no retraining, that cut the worst-bus |dQ| from 138.3 to 3.7 pu at k=8.

This module exists because that projection is applied in two places -- inside
the model's forward pass during training, and to a stored checkpoint when
scoring -- and the post-hoc measurement is the acceptance check for the
in-training one.  Two copies of the same formula could drift apart and silently
invalidate that check, so there is one implementation here.
"""
from __future__ import annotations

from typing import Tuple

import torch

Basis = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def fit_basis(vm: torch.Tensor, th: torch.Tensor, k: int) -> Tuple[Basis, float, float]:
    """Rank-k orthonormal bases of the reference |V| and theta variation.

    `vm`, `th` are [n_scenarios, n_bus] reference states from the TRAINING
    split only.  Angles are centred on the per-bus circular mean and wrapped
    before the SVD, so a bus sitting near +-pi does not contribute a spurious
    direction.  Returns (basis, energy_v, energy_theta) where the energies are
    the fraction of reference variance the k directions capture.
    """
    vm = vm.to(torch.float64)
    th = th.to(torch.float64)
    vbar = vm.mean(dim=0)
    tbar = torch.atan2(torch.sin(th).mean(dim=0), torch.cos(th).mean(dim=0))
    dv = vm - vbar
    dt = torch.atan2(torch.sin(th - tbar), torch.cos(th - tbar))
    k = int(min(k, dv.shape[0], dv.shape[1]))

    def one(D):
        _, sv, Vh = torch.linalg.svd(D, full_matrices=False)
        energy = (sv ** 2).cumsum(0) / (sv ** 2).sum()
        return Vh[:k].T.contiguous(), float(energy[k - 1])

    Uv, ev = one(dv)
    Ut, et = one(dt)
    return (Uv, Ut, vbar, tbar), ev, et


def project_state(v: torch.Tensor, th: torch.Tensor, basis: Basis):
    """Project a voltage state onto vbar + span(U), preserving shape and dtype.

        v_proj = vbar + U U^T (v - vbar)

    and the circular equivalent for theta.  U is orthonormal, so this is an
    exact orthogonal projection and is idempotent.  It is a fixed linear
    operator, so gradients pass through exactly; it is also rank-deficient by
    construction, which is the intent -- parameter directions that act only
    outside the span receive no gradient.
    """
    Uv, Ut, vbar, tbar = (t.to(device=v.device, dtype=v.dtype) for t in basis)
    n = vbar.numel()
    flat = v.reshape(-1)
    if n <= 0 or flat.numel() % n:
        raise ValueError(
            f"manifold projection expects a multiple of {n} buses, got "
            f"{flat.numel()}; the basis does not match this grid."
        )
    dv = flat.reshape(-1, n) - vbar
    dt = torch.atan2(torch.sin(th.reshape(-1, n) - tbar),
                     torch.cos(th.reshape(-1, n) - tbar))
    vp = (vbar + (dv @ Uv) @ Uv.T).reshape(v.shape)
    tp = tbar + (dt @ Ut) @ Ut.T
    tp = torch.atan2(torch.sin(tp), torch.cos(tp)).reshape(th.shape)
    return vp, tp


def project_packed(V: torch.Tensor, basis: Basis) -> torch.Tensor:
    """project_state for a packed [..., n, 2] magnitude/angle tensor."""
    vp, tp = project_state(V[..., 0], V[..., 1], basis)
    return torch.stack([vp, tp], dim=-1)
