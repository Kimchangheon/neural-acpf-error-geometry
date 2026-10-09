"""Step C: remove the fixed per-bus bias.

Most of a surrogate's voltage error is not a failure to track the scenario. It
is a fixed offset that the same bus carries in every scenario. That part is
removable by subtraction, and the subtraction can be learned without touching
the test set.

The offset is therefore fitted on the **training split only**, from predictions
the model makes on training scenarios and the solver states for those same
scenarios. Fitting it on test data would make the correction unfalsifiable.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ._util import angle_difference, circular_sum_mean, wrap_angle


@dataclass(frozen=True)
class Offset:
    """Per-bus prediction bias, ``[n_bus]`` for each channel.

    ``d_ang`` is a circular mean of the wrapped prediction error, not an
    arithmetic one.
    """

    d_mag: Tensor
    d_ang: Tensor

    @property
    def n_bus(self) -> int:
        return int(self.d_mag.numel())

    def to(self, *, device=None, dtype=None) -> "Offset":
        def m(t: Tensor) -> Tensor:
            return t.to(device=device or t.device, dtype=dtype or t.dtype)

        return Offset(m(self.d_mag), m(self.d_ang))

    def state_dict(self) -> dict:
        return {"d_mag": self.d_mag, "d_ang": self.d_ang}

    @staticmethod
    def from_state_dict(d: dict) -> "Offset":
        return Offset(d["d_mag"], d["d_ang"])


def fit_offset(predictions: Tensor, reference_states: Tensor) -> Offset:
    """Fit the per-bus bias from paired training predictions and solutions.

    Parameters
    ----------
    predictions, reference_states:
        ``[n_scenarios, n_bus, 2]``, the same scenarios in the same order,
        from the **training** split.
    """
    pred = _check_pair(predictions, reference_states)
    ref = reference_states
    d_mag = (pred[..., 0].to(torch.float64) - ref[..., 0].to(torch.float64)).mean(dim=0)
    d_ang_each = angle_difference(pred[..., 1].to(torch.float64),
                                  ref[..., 1].to(torch.float64))
    d_ang = circular_sum_mean(torch.sin(d_ang_each).mean(dim=0),
                              torch.cos(d_ang_each).mean(dim=0))
    return Offset(d_mag, d_ang)


def fit_offset_streaming(pairs) -> Offset:
    """Same fit, for training splits too large to hold in memory.

    `pairs` yields ``(predictions, reference_states)`` batches, each
    ``[n_scenarios, n_bus, 2]``. Equivalent to `fit_offset` on the
    concatenation, to floating-point accumulation order.
    """
    sum_mag = sum_sin = sum_cos = None
    n = 0
    for pred, ref in pairs:
        pred = _check_pair(pred, ref)
        dm = (pred[..., 0].to(torch.float64) - ref[..., 0].to(torch.float64)).sum(dim=0)
        da = angle_difference(pred[..., 1].to(torch.float64),
                              ref[..., 1].to(torch.float64))
        sn, cs = torch.sin(da).sum(dim=0), torch.cos(da).sum(dim=0)
        if sum_mag is None:
            sum_mag, sum_sin, sum_cos = dm, sn, cs
        else:
            sum_mag = sum_mag + dm
            sum_sin = sum_sin + sn
            sum_cos = sum_cos + cs
        n += int(pred.shape[0])
    if not n:
        raise ValueError("fit_offset_streaming received no scenarios.")
    return Offset(sum_mag / n, circular_sum_mean(sum_sin / n, sum_cos / n))


def apply_offset(states: Tensor, offset: Offset) -> Tensor:
    """Subtract the bias. Returns a new tensor; the input is untouched."""
    off = offset.to(device=states.device, dtype=states.dtype)
    if off.n_bus != states.shape[-2]:
        raise ValueError(
            f"offset was fitted on {off.n_bus} buses but the states have "
            f"{states.shape[-2]}; the calibration does not belong to this grid."
        )
    mag = states[..., 0] - off.d_mag
    ang = wrap_angle(states[..., 1] - off.d_ang)
    return torch.stack([mag, ang], dim=-1)


def _check_pair(predictions: Tensor, reference_states: Tensor) -> Tensor:
    for name, t in (("predictions", predictions), ("reference_states", reference_states)):
        if t.dim() != 3 or t.shape[-1] != 2:
            raise ValueError(
                f"{name} must be [n_scenarios, n_bus, 2]; got {tuple(t.shape)}."
            )
    if predictions.shape != reference_states.shape:
        raise ValueError(
            "predictions and reference_states must be the same scenarios in the "
            f"same order; got {tuple(predictions.shape)} and "
            f"{tuple(reference_states.shape)}."
        )
    return predictions
