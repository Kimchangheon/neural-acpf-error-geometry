"""Angle handling, shared by every module here.

Voltage angles are circular. Averaging them arithmetically, or comparing them
without wrapping, produces errors of 2*pi at buses that happen to sit near the
branch cut. Every angle operation in this package goes through these helpers.
"""
from __future__ import annotations

import torch
from torch import Tensor


def wrap_angle(a: Tensor) -> Tensor:
    """Wrap to (-pi, pi]."""
    return torch.atan2(torch.sin(a), torch.cos(a))


def circular_mean(a: Tensor, dim: int = 0) -> Tensor:
    """Mean direction, i.e. the angle of the mean unit vector."""
    return torch.atan2(torch.sin(a).mean(dim=dim), torch.cos(a).mean(dim=dim))


def circular_sum_mean(sin_sum: Tensor, cos_sum: Tensor) -> Tensor:
    """Circular mean from accumulated sums, for streaming fits."""
    return torch.atan2(sin_sum, cos_sum)


def angle_difference(a: Tensor, b: Tensor) -> Tensor:
    """Wrapped a - b, so the result is always the shortest signed arc."""
    return wrap_angle(a - b)
