"""CSP: calibration followed by subspace projection.

This is the object most readers want. Fit it once on the training split, then
apply it to any prediction the model makes afterwards.

    csp = CSP.fit(train_predictions, train_references, rank=16)
    corrected = csp(test_predictions)

Nothing here touches the model. CSP is a transform on its output, so it costs
one small matrix multiply per scenario and needs no retraining.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from .basis import Basis, fit_basis
from .calibration import Offset, apply_offset, fit_offset
from .projection import (fit_basis_unknown_only, project, project_unknown_only,
                         restore_setpoints)

Scope = str  # "full" | "restore_known" | "unknown_only"
_SCOPES = ("full", "restore_known", "unknown_only")


@dataclass
class CSP:
    """A fitted calibration and projection pair.

    Parameters
    ----------
    offset:
        Per-bus bias from step C.
    basis:
        Training-solution subspace from step P.
    scope:
        Which coordinates the projection acts on; see `csp.projection`.
    bus_type:
        Required for the ``restore_known`` and ``unknown_only`` scopes.
    free_masks:
        Cached masks for ``unknown_only``.
    """

    offset: Offset
    basis: Basis
    scope: Scope = "full"
    bus_type: Optional[Tensor] = None
    free_masks: Optional[Tuple[Tensor, Tensor]] = None

    # ---------------------------------------------------------------- fitting

    @classmethod
    def fit(cls, train_predictions: Tensor, train_references: Tensor,
            rank: int, *, scope: Scope = "full",
            bus_type: Optional[Tensor] = None) -> "CSP":
        """Fit both steps on the training split.

        Parameters
        ----------
        train_predictions:
            ``[n_scenarios, n_bus, 2]`` model output on training scenarios.
        train_references:
            ``[n_scenarios, n_bus, 2]`` solver solutions for the same
            scenarios, in the same order.
        rank:
            The `k` of CSP_k. Choose it on a validation split, never on test.
            `Basis.energy_mag` reports how much reference variance `k`
            captures, which is a reasonable way to shortlist candidates.
        """
        if scope not in _SCOPES:
            raise ValueError(f"scope must be one of {_SCOPES}, got {scope!r}")
        if scope in ("restore_known", "unknown_only") and bus_type is None:
            raise ValueError(f"scope {scope!r} needs bus_type")

        offset = fit_offset(train_predictions, train_references)
        if scope == "unknown_only":
            basis, masks = fit_basis_unknown_only(train_references, rank, bus_type)
            return cls(offset, basis, scope, bus_type, masks)
        return cls(offset, fit_basis(train_references, rank), scope, bus_type, None)

    # --------------------------------------------------------------- applying

    def calibrate(self, states: Tensor) -> Tensor:
        """Step C alone."""
        return apply_offset(states, self.offset)

    def project(self, states: Tensor) -> Tensor:
        """Step P alone."""
        if self.scope == "unknown_only":
            return project_unknown_only(states, self.basis, self.free_masks)
        out = project(states, self.basis)
        if self.scope == "restore_known":
            out = restore_setpoints(out, self.bus_type, states)
        return out

    def __call__(self, states: Tensor) -> Tensor:
        """Both steps: calibrate, then project."""
        return self.project(self.calibrate(states))

    def stages(self, states: Tensor) -> Dict[str, Tensor]:
        """Every intermediate state, keyed as in the paper's tables.

        Returns ``{"raw", "C", "P{k}", "CSP{k}"}``. Useful for reproducing the
        four-row structure of Table 1 from a single forward pass.
        """
        k = self.basis.rank
        calibrated = self.calibrate(states)
        return {"raw": states,
                "C": calibrated,
                f"P{k}": self.project(states),
                f"CSP{k}": self.project(calibrated)}

    # --------------------------------------------------------- serialisation

    @property
    def rank(self) -> int:
        return self.basis.rank

    def state_dict(self) -> dict:
        return {"offset": self.offset.state_dict(),
                "basis": self.basis.state_dict(),
                "scope": self.scope,
                "bus_type": self.bus_type,
                "free_masks": self.free_masks}

    def save(self, path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, map_location=None) -> "CSP":
        d = torch.load(path, map_location=map_location, weights_only=False)
        return cls(Offset.from_state_dict(d["offset"]),
                   Basis.from_state_dict(d["basis"]),
                   d.get("scope", "full"), d.get("bus_type"),
                   d.get("free_masks"))

    def __repr__(self) -> str:
        return (f"CSP(rank={self.rank}, scope={self.scope!r}, "
                f"n_bus={self.basis.n_bus}, "
                f"energy_mag={self.basis.energy_mag:.4f}, "
                f"energy_ang={self.basis.energy_ang:.4f})")
