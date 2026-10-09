"""Output-side anchors shared by every power-flow driver.

Both are reparametrisations of what the network emits, not changes to the
network, so they attach at the point a driver produces its voltage state -- the
same place apply_known_v is already applied.

Measured on case14 with LUMINA (per-bus sigma 0.023129, 3.461 deg):

    control                      |V| 8.1164e-03 (R^2 0.877)   theta 0.136 deg
    theta anchor only            |V| 8.4492e-03               theta 3.182 deg
    theta anchor + supervision   |V| 8.1601e-03               theta 0.136 deg
    + magnitude anchor on prior  |V| 4.6610e-04 (R^2 0.99959) theta 0.130 deg

The magnitude anchor is the lever; adding the prior as an input column as well
moved 4.6610e-04 to 4.0063e-04, which is why only the output side is shared
here.  Anchoring on an UNINFORMATIVE reference is worse than not anchoring:
v_start, flat 1.0 at PQ buses, scored 0.641 against a 0.877 control.
"""
from __future__ import annotations

import torch

_PRIOR_CACHE = {}


def add_anchor_args(parser):
    parser.add_argument(
        "--theta_anchor", choices=("none", "start"), default="none",
        help="'start' emits wrap(theta_start + z) instead of the angle itself. "
             "The corpus DC start angle is 0.568 deg RMSE from the solution on "
             "case14, against a spread of 3.461 deg.")
    parser.add_argument(
        "--v_anchor", choices=("none", "start", "prior"), default="none",
        help="'prior' emits v_prior + z, with v_prior a chord step from the "
             "flat point. 'start' anchors on v_start, which is flat 1.0 at PQ "
             "buses and measured worse than no anchor at all.")
    parser.add_argument(
        "--vmag_prior_angle", choices=("flat", "start"), default="start",
        help="Where the chord step's mismatch is evaluated. Prior PQ |V| R^2: "
             "case14 0.9986 at 'start' vs 0.95 at 'flat'; GBnetwork +0.60 vs "
             "-4.56, because a flat-angle linearisation does not survive a "
             "grid whose angles span tens of degrees.")


def chord_prior(batch_cpu, n_bus, angle="start"):
    """Chord-step magnitude prior for every scenario, flat over the bus axis."""
    from vmag_prior import FlatPointJacobianPrior

    # Anchor construction deliberately lives on CPU: the Jacobian prior uses a
    # complex128 dense factorisation, while callers may already have moved the
    # collated sparse Ybus and state tensors to CUDA.  Mixing those devices in
    # FlatPointJacobianPrior used to crash GraphKit anchor runs before epoch 0.
    bt = batch_cpu["bus_type"].reshape(-1)[:n_bus].detach().cpu()
    ybus = batch_cpu["Ybus"].detach().cpu()
    key = (int(n_bus), int(bt.sum().item()), str(angle))
    prior = _PRIOR_CACHE.get(key)
    if prior is None:
        prior = FlatPointJacobianPrior(ybus, bt, mismatch_angle=angle)
        _PRIOR_CACHE[key] = prior
        print(f"[prior] chord-step Jacobian factorised once for {n_bus} buses "
              f"(mismatch at theta={angle})", flush=True)
    S = batch_cpu["S_start"].reshape(-1, n_bus).detach().cpu()
    VS = batch_cpu["V_start"].reshape(-1, n_bus, 2).detach().cpu()
    return prior.batch(S.real, S.imag, VS[..., 0], VS[..., 1]).reshape(-1)


def apply_anchors(V, batch_cpu, args):
    """Re-express V as a residual about the configured anchors.

    V is [..., 2] with (magnitude, angle) last, in the driver's own layout; the
    anchors are flat over the concatenated bus axis, which is how every driver
    in this repo lays a batch out.
    """
    v_mode = str(getattr(args, "v_anchor", "none"))
    th_mode = str(getattr(args, "theta_anchor", "none"))
    if v_mode == "none" and th_mode == "none":
        return V

    shape = V.shape
    flat = V.reshape(-1, 2)
    n_bus = int(batch_cpu["sizes"].long()[0].item())
    mag, th = flat[:, 0], flat[:, 1]

    if th_mode == "start":
        th0 = batch_cpu["V_start"].reshape(-1, 2)[:, 1].to(device=th.device, dtype=th.dtype)
        th = th + th0
    th = torch.atan2(torch.sin(th), torch.cos(th))

    if v_mode != "none":
        if v_mode == "start":
            v0 = batch_cpu["V_start"].reshape(-1, 2)[:, 0]
        else:
            v0 = chord_prior(batch_cpu, n_bus,
                             str(getattr(args, "vmag_prior_angle", "start")))
        mag = mag + v0.to(device=mag.device, dtype=mag.dtype)

    return torch.stack([mag, th], dim=-1).reshape(shape)


def chord_start_state(batch_cpu, angle="start"):
    """A replacement V_start whose PQ magnitudes come from the chord step.

    For a model that already iterates from V_start -- PIGNN starts there and its
    heads are zero-initialised, so V_pred equals V_start at epoch 0 -- the
    analogue of the output anchor used by the other drivers is a better starting
    iterate, not a term added after K Newton steps.

    Only PQ magnitudes move.  PV and slack magnitudes and the slack angle are
    passed through untouched, so anything downstream that treats V_start as the
    reference for the specified quantities keeps working unchanged.  The angle
    column is left at the corpus DC start, which is what --theta_anchor start
    uses in the other drivers.
    """
    n_bus = int(batch_cpu["sizes"].long()[0].item())
    v_prior = chord_prior(batch_cpu, n_bus, angle)
    VS = batch_cpu["V_start"]
    shape = VS.shape
    flat = VS.reshape(-1, 2).clone()
    flat[:, 0] = v_prior.to(device=flat.device, dtype=flat.dtype)
    return flat.reshape(shape)
