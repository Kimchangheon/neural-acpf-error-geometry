"""π-model branch-flow unit tests."""
from __future__ import annotations

import math

import pytest
import torch

from gridsfm.model import GridTransformerBackbone
from gridsfm.schema import (
    AC_LINE_BFR_IDX, AC_LINE_BTO_IDX, AC_LINE_R_IDX, AC_LINE_X_IDX,
    TR_BFR_IDX, TR_BTO_IDX, TR_R_IDX, TR_X_IDX, TR_TAP_IDX, TR_SHIFT_IDX,
)


def _ac_attr(*, r=0.01, x=0.1, b_fr=0.0, b_to=0.0, n_cols=9):
    row = [0.0] * n_cols
    row[AC_LINE_R_IDX]   = r
    row[AC_LINE_X_IDX]   = x
    row[AC_LINE_BFR_IDX] = b_fr
    row[AC_LINE_BTO_IDX] = b_to
    return torch.tensor([row], dtype=torch.float32)


def _tr_attr(*, r=0.01, x=0.1, b_fr=0.0, b_to=0.0, tap=1.0, shift=0.0, n_cols=11):
    row = [0.0] * n_cols
    row[TR_R_IDX]     = r
    row[TR_X_IDX]     = x
    row[TR_BFR_IDX]   = b_fr
    row[TR_BTO_IDX]   = b_to
    row[TR_TAP_IDX]   = tap
    row[TR_SHIFT_IDX] = shift
    return torch.tensor([row], dtype=torch.float32)


def test_ac_line_zero_angle_diff_zero_flow():
    ei = torch.tensor([[0], [1]], dtype=torch.long)
    V  = torch.tensor([1.0, 1.0])
    th = torch.tensor([0.0, 0.0])
    Pij, _, Pji, _ = GridTransformerBackbone._pi_model_flows_ac_line(
        ei, _ac_attr(r=0.0, x=0.1), V, th
    )
    assert torch.allclose(Pij, torch.zeros_like(Pij), atol=1e-6)
    assert torch.allclose(Pji, torch.zeros_like(Pji), atol=1e-6)
    Pij_r, _, Pji_r, _ = GridTransformerBackbone._pi_model_flows_ac_line(
        ei, _ac_attr(r=0.01, x=0.0), V, th
    )
    assert torch.allclose(Pij_r, torch.zeros_like(Pij_r), atol=1e-6)
    assert torch.allclose(Pji_r, torch.zeros_like(Pji_r), atol=1e-6)


def test_ac_line_resistive_loss_with_angle_drop():
    ei = torch.tensor([[0], [1]], dtype=torch.long)
    V  = torch.tensor([1.0, 1.0])
    th = torch.tensor([0.05, 0.0])
    Pij, _, Pji, _ = GridTransformerBackbone._pi_model_flows_ac_line(
        ei, _ac_attr(r=0.01, x=0.1), V, th
    )
    loss = (Pij + Pji).item()
    assert loss > 0  # I^2 R loss leaves the wire on both ends


def test_ac_line_p_flow_sign_with_angle_drop():
    ei = torch.tensor([[0], [1]], dtype=torch.long)
    V  = torch.tensor([1.0, 1.0])
    th = torch.tensor([0.05, 0.0])  # θ_i > θ_j
    Pij, _, Pji, _ = GridTransformerBackbone._pi_model_flows_ac_line(
        ei, _ac_attr(r=0.0, x=0.1), V, th
    )
    assert Pij.item() > 0
    assert Pji.item() < 0
    # Lossless → Pij ≈ -Pji
    assert math.isclose(Pij.item(), -Pji.item(), abs_tol=1e-6)


def test_transformer_reduces_to_ac_line_at_unit_tap_zero_shift():
    ei = torch.tensor([[0], [1]], dtype=torch.long)
    V  = torch.tensor([1.05, 0.98])
    th = torch.tensor([0.02, -0.01])
    ac_flows = GridTransformerBackbone._pi_model_flows_ac_line(
        ei, _ac_attr(r=0.01, x=0.1, b_fr=0.001, b_to=0.002), V, th
    )
    tr_flows = GridTransformerBackbone._pi_model_flows_transformer(
        ei, _tr_attr(r=0.01, x=0.1, b_fr=0.001, b_to=0.002, tap=1.0, shift=0.0), V, th
    )
    for a, t in zip(ac_flows, tr_flows):
        torch.testing.assert_close(a, t, atol=1e-6, rtol=1e-6)


def test_transformer_phase_shift_changes_p_flow():
    ei = torch.tensor([[0], [1]], dtype=torch.long)
    V  = torch.tensor([1.0, 1.0])
    th = torch.tensor([0.0, 0.0])
    Pij0, _, _, _ = GridTransformerBackbone._pi_model_flows_transformer(
        ei, _tr_attr(r=0.0, x=0.1, tap=1.0, shift=0.0), V, th
    )
    Pij_s, _, _, _ = GridTransformerBackbone._pi_model_flows_transformer(
        ei, _tr_attr(r=0.0, x=0.1, tap=1.0, shift=0.05), V, th
    )
    # shift > 0 ⇔ effective θ_i shifted negative; same-θ buses now see a drop.
    assert not torch.allclose(Pij0, Pij_s, atol=1e-6)


def test_5_node_radial_balances():
    n_bus = 5
    src = list(range(n_bus - 1))
    dst = list(range(1, n_bus))
    ei = torch.tensor([src, dst], dtype=torch.long)
    n_edges = ei.size(1)
    attr_rows = []
    for _ in range(n_edges):
        row = [0.0] * 9
        row[AC_LINE_R_IDX] = 0.0
        row[AC_LINE_X_IDX] = 0.1
        attr_rows.append(row)
    edge_attr = torch.tensor(attr_rows, dtype=torch.float32)
    V  = torch.ones(n_bus)
    th = torch.linspace(0.04, 0.0, n_bus)  # monotonic angle drop

    Pij, Qij, Pji, Qji = GridTransformerBackbone._pi_model_flows_ac_line(
        ei, edge_attr, V, th
    )
    # Lossless chain: Pij + Pji == 0 per edge.
    assert torch.allclose(Pij + Pji, torch.zeros_like(Pij), atol=1e-6)
    # Monotonic angle drop → Pij > 0 on every edge (power flows i→j).
    assert (Pij > 0).all()
