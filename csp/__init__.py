"""CSP — calibrated subspace projection for AC power-flow surrogates.

A neural power-flow surrogate can have small voltage error and still return a
state that violates the network equations. The error concentrates in a
low-dimensional subspace spanned by the solutions the grid actually produces,
which makes most of it removable *after* training:

1. **C** — subtract the per-bus bias measured on the training split.
2. **P_k** — project onto the rank-`k` training-solution subspace.

Neither step touches the model, so CSP applies to a frozen checkpoint and costs
one small matrix multiply per scenario.

Quickstart
----------
>>> from csp import CSP, evaluate
>>> transform = CSP.fit(train_predictions, train_references, rank=16)
>>> corrected = transform(test_predictions)
>>> evaluate(corrected, test_references, ybus=Y, s_specified=S, bus_type=bt)

Conventions
-----------
States are ``[n_scenarios, n_bus, 2]`` tensors whose last axis is
``(voltage magnitude in per-unit, voltage angle in radians)``. Bus types follow
the MATPOWER/pandapower convention: 1 = slack, 2 = PV, anything else = PQ.
Everything is fitted on the training split only.

See ``examples/quickstart.py`` for a complete runnable example that needs no
downloads, and ``docs/`` for the map back to the paper.
"""
from __future__ import annotations

from ._util import angle_difference, circular_mean, wrap_angle
from .basis import Basis, fit_basis
from .calibration import (Offset, apply_offset, fit_offset,
                          fit_offset_streaming)
from .metrics import (angle_rmse_deg, evaluate, power_balance, voltage_rmse,
                      within_bus_r2_slope)
from .projection import (fit_basis_unknown_only, project,
                         project_unknown_only, restore_setpoints,
                         unknown_masks)
from .transform import CSP

__version__ = "1.0.0"

__all__ = [
    "CSP",
    # basis
    "Basis", "fit_basis",
    # calibration
    "Offset", "fit_offset", "fit_offset_streaming", "apply_offset",
    # projection
    "project", "project_unknown_only", "fit_basis_unknown_only",
    "restore_setpoints", "unknown_masks",
    # metrics
    "evaluate", "power_balance", "voltage_rmse", "angle_rmse_deg",
    "within_bus_r2_slope",
    # angle helpers
    "wrap_angle", "circular_mean", "angle_difference",
]
