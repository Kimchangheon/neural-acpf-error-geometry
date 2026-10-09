"""The refactored package must agree with the code that produced the paper.

A clean rewrite that silently differs from the published implementation is
worse than no rewrite. These tests import the archived research code from
``reproduction/code/`` and check the new package against it numerically.

Run with:  pytest tests/ -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reproduction" / "code"))

import csp  # noqa: E402

paper_manifold = pytest.importorskip(
    "manifold_projection",
    reason="reproduction/code/manifold_projection.py is required for the "
           "equivalence tests; they are skipped if the archive is absent.",
)


def _states(n_scen=64, n_bus=14, seed=0):
    g = torch.Generator().manual_seed(seed)
    mag = 1.0 + 0.03 * torch.randn(n_scen, n_bus, generator=g, dtype=torch.float64)
    ang = 0.25 * torch.randn(n_scen, n_bus, generator=g, dtype=torch.float64)
    return torch.stack([mag, ang], dim=-1)


def test_basis_matches_paper_implementation():
    ref = _states()
    k = 6
    mine = csp.fit_basis(ref, k)
    (u_v, u_t, vbar, tbar), ev, et = paper_manifold.fit_basis(
        ref[..., 0], ref[..., 1], k)

    assert torch.allclose(mine.mean_mag, vbar, atol=0, rtol=0)
    assert torch.allclose(mine.mean_ang, tbar, atol=0, rtol=0)
    assert math.isclose(mine.energy_mag, ev, rel_tol=1e-12)
    assert math.isclose(mine.energy_ang, et, rel_tol=1e-12)
    # Singular vectors are only defined up to sign, so compare the projectors.
    assert torch.allclose(mine.u_mag @ mine.u_mag.T, u_v @ u_v.T, atol=1e-12)
    assert torch.allclose(mine.u_ang @ mine.u_ang.T, u_t @ u_t.T, atol=1e-12)


def test_projection_matches_paper_implementation():
    ref, pred = _states(seed=0), _states(seed=1)
    k = 6
    mine = csp.fit_basis(ref, k)
    paper_basis, _, _ = paper_manifold.fit_basis(ref[..., 0], ref[..., 1], k)

    got = csp.project(pred, mine)
    want = paper_manifold.project_packed(pred, paper_basis)
    assert torch.allclose(got, want, atol=1e-12), \
        f"max deviation {float((got - want).abs().max()):.3e}"


def test_calibration_matches_paper_formula():
    """`fit_per_bus_offset` + `calibrated_forward`, written out directly.

    The archived versions take data loaders, so the arithmetic is reproduced
    here rather than imported.
    """
    ref, pred = _states(seed=2), _states(seed=3)
    offset = csp.fit_offset(pred, ref)

    d_mag_paper = (pred[..., 0] - ref[..., 0]).mean(0)
    d_th = torch.atan2(torch.sin(pred[..., 1] - ref[..., 1]),
                       torch.cos(pred[..., 1] - ref[..., 1]))
    d_ang_paper = torch.atan2(torch.sin(d_th).mean(0), torch.cos(d_th).mean(0))
    assert torch.allclose(offset.d_mag, d_mag_paper, atol=1e-14)
    assert torch.allclose(offset.d_ang, d_ang_paper, atol=1e-14)

    got = csp.apply_offset(pred, offset)
    want_mag = pred[..., 0] - d_mag_paper
    want_ang = torch.atan2(torch.sin(pred[..., 1] - d_ang_paper),
                           torch.cos(pred[..., 1] - d_ang_paper))
    assert torch.allclose(got[..., 0], want_mag, atol=1e-14)
    assert torch.allclose(got[..., 1], want_ang, atol=1e-14)


def test_power_balance_matches_paper_formula():
    """`pb_per_scenario`, written out directly on a dense Ybus."""
    torch.manual_seed(4)
    n_scen, n_bus = 8, 14
    states = _states(n_scen, n_bus, seed=5)
    a = torch.randn(n_bus, n_bus, dtype=torch.float64)
    ybus = (a + a.T).to(torch.complex128) + 1j * torch.eye(n_bus, dtype=torch.float64)
    s_spec = torch.randn(n_scen, n_bus, dtype=torch.float64).to(torch.complex128)
    bus_type = torch.tensor([1, 2, 2] + [3] * (n_bus - 3))

    got = csp.power_balance(states, ybus, s_spec, bus_type)

    vc = (states[..., 0] * torch.exp(1j * states[..., 1])).to(torch.complex128)
    sc = vc * (vc @ ybus.T).conj()
    dp, dq = s_spec.real - sc.real, s_spec.imag - sc.imag
    p_mask = (bus_type != 1).to(torch.float64)
    q_mask = ((bus_type != 1) & (bus_type != 2)).to(torch.float64)
    want = torch.sqrt((dp * p_mask) ** 2 + (dq * q_mask) ** 2)
    assert torch.allclose(got, want, atol=1e-12)


def test_fit_offset_streaming_matches_batched():
    ref, pred = _states(n_scen=60, seed=6), _states(n_scen=60, seed=7)
    whole = csp.fit_offset(pred, ref)
    chunks = csp.fit_offset_streaming(
        (pred[i:i + 7], ref[i:i + 7]) for i in range(0, 60, 7))
    assert torch.allclose(whole.d_mag, chunks.d_mag, atol=1e-12)
    assert torch.allclose(whole.d_ang, chunks.d_ang, atol=1e-12)
