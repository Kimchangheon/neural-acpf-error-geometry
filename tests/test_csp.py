"""Properties CSP must have, independent of any particular dataset."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csp  # noqa: E402


def states(n_scen=64, n_bus=14, seed=0, ang_scale=0.25):
    g = torch.Generator().manual_seed(seed)
    mag = 1.0 + 0.03 * torch.randn(n_scen, n_bus, generator=g, dtype=torch.float64)
    ang = ang_scale * torch.randn(n_scen, n_bus, generator=g, dtype=torch.float64)
    return torch.stack([mag, ang], dim=-1)


# ----------------------------------------------------------------- projection

def test_basis_is_orthonormal():
    b = csp.fit_basis(states(), 6)
    for u in (b.u_mag, b.u_ang):
        assert torch.allclose(u.T @ u, torch.eye(6, dtype=u.dtype), atol=1e-12)


def test_projection_is_idempotent():
    b = csp.fit_basis(states(seed=0), 6)
    once = csp.project(states(seed=1), b)
    twice = csp.project(once, b)
    assert torch.allclose(once, twice, atol=1e-11)


def test_full_rank_projection_is_the_identity():
    ref = states(n_scen=40, n_bus=14, seed=2)
    b = csp.fit_basis(ref, rank=14)
    out = csp.project(ref, b)
    assert torch.allclose(out, ref, atol=1e-9)


def test_rank_is_clipped_to_the_data():
    ref = states(n_scen=5, n_bus=14, seed=3)
    assert csp.fit_basis(ref, rank=16).rank == 5


def test_energy_is_monotone_in_rank():
    ref = states(seed=4)
    energies = [csp.fit_basis(ref, k).energy_mag for k in (1, 2, 4, 8)]
    assert all(a <= b + 1e-12 for a, b in zip(energies, energies[1:]))
    assert 0.0 <= energies[0] <= energies[-1] <= 1.0 + 1e-12


def test_angles_near_the_branch_cut_are_handled():
    """A bus sitting at +-pi must not create a spurious 2*pi direction."""
    g = torch.Generator().manual_seed(5)
    n_scen, n_bus = 80, 6
    mag = torch.ones(n_scen, n_bus, dtype=torch.float64)
    ang = math.pi - 0.01 + 0.02 * torch.rand(n_scen, n_bus, generator=g,
                                             dtype=torch.float64)
    ang = csp.wrap_angle(ang)          # straddles the cut
    ref = torch.stack([mag, ang], dim=-1)
    b = csp.fit_basis(ref, rank=3)
    out = csp.project(ref, b)
    # Reconstruction error must be small in the wrapped sense, not the raw one.
    d = csp.angle_difference(out[..., 1], ref[..., 1])
    assert float(d.abs().max()) < 0.05, float(d.abs().max())


def test_leading_dimensions_are_preserved():
    b = csp.fit_basis(states(), 4)
    x = states(n_scen=12).reshape(3, 4, 14, 2)
    assert csp.project(x, b).shape == x.shape


def test_basis_from_another_grid_is_rejected():
    b = csp.fit_basis(states(n_bus=14), 4)
    with pytest.raises(ValueError, match="does not belong to this grid"):
        csp.project(states(n_bus=30), b)


# ---------------------------------------------------------------- calibration

def test_calibration_removes_a_constant_bias_exactly():
    ref = states(seed=6)
    bias_mag = 0.01 * torch.arange(14, dtype=torch.float64)
    pred = ref.clone()
    pred[..., 0] += bias_mag
    offset = csp.fit_offset(pred, ref)
    assert torch.allclose(offset.d_mag, bias_mag, atol=1e-12)
    assert torch.allclose(csp.apply_offset(pred, offset)[..., 0], ref[..., 0],
                          atol=1e-12)


def test_calibration_rejects_mismatched_pairs():
    with pytest.raises(ValueError, match="same scenarios"):
        csp.fit_offset(states(n_scen=10), states(n_scen=11))


# -------------------------------------------------------------------- scopes

def _bus_type(n_bus=14):
    bt = torch.full((n_bus,), 3, dtype=torch.long)
    bt[0] = 1                       # slack
    bt[1:4] = 2                     # PV
    return bt


def test_restore_setpoints_holds_the_prescribed_quantities():
    bt = _bus_type()
    ref, pred = states(seed=7), states(seed=8)
    transform = csp.CSP.fit(pred, ref, rank=6, scope="restore_known", bus_type=bt)
    out = transform(pred)
    hold_mag = (bt == 1) | (bt == 2)
    assert torch.allclose(out[..., hold_mag, 0],
                          transform.calibrate(pred)[..., hold_mag, 0], atol=1e-12)
    assert torch.allclose(out[..., bt == 1, 1],
                          transform.calibrate(pred)[..., bt == 1, 1], atol=1e-12)


def test_unknown_only_leaves_prescribed_coordinates_untouched():
    bt = _bus_type()
    ref, pred = states(seed=9), states(seed=10)
    transform = csp.CSP.fit(pred, ref, rank=4, scope="unknown_only", bus_type=bt)
    cal = transform.calibrate(pred)
    out = transform(pred)
    hold_mag = (bt == 1) | (bt == 2)
    assert torch.allclose(out[..., hold_mag, 0], cal[..., hold_mag, 0], atol=1e-12)


def test_scope_needing_bus_type_says_so():
    with pytest.raises(ValueError, match="bus_type"):
        csp.CSP.fit(states(), states(), rank=4, scope="restore_known")


def test_unknown_scope_is_rejected():
    with pytest.raises(ValueError, match="scope must be one of"):
        csp.CSP.fit(states(), states(), rank=4, scope="nonsense")


# ---------------------------------------------------------------- the object

def test_stages_are_named_as_in_the_paper():
    transform = csp.CSP.fit(states(seed=11), states(seed=12), rank=16)
    got = transform.stages(states(seed=13))
    assert set(got) == {"raw", "C", "P14", "CSP14"}   # rank clipped to 14 buses


def test_csp_is_calibrate_then_project():
    transform = csp.CSP.fit(states(seed=14), states(seed=15), rank=6)
    x = states(seed=16)
    assert torch.allclose(transform(x),
                          transform.project(transform.calibrate(x)), atol=0)


def test_save_and_load_round_trip(tmp_path):
    transform = csp.CSP.fit(states(seed=17), states(seed=18), rank=6)
    p = tmp_path / "csp.pt"
    transform.save(p)
    back = csp.CSP.load(p)
    x = states(seed=19)
    assert torch.allclose(transform(x), back(x), atol=0)
    assert back.rank == transform.rank and back.scope == transform.scope


# -------------------------------------------------------------------- metrics

def test_power_balance_is_zero_at_the_true_solution():
    """If V solves the equations, the specified injection is what it produces."""
    torch.manual_seed(20)
    n_scen, n_bus = 6, 10
    a = torch.randn(n_bus, n_bus, dtype=torch.float64)
    ybus = (a + a.T).to(torch.complex128) + 1j * torch.eye(n_bus, dtype=torch.float64)
    st = states(n_scen, n_bus, seed=21)
    v = (st[..., 0] * torch.exp(1j * st[..., 1])).to(torch.complex128)
    s_spec = v * (v @ ybus.T).conj()          # by construction, exactly consistent
    pb = csp.power_balance(st, ybus, s_spec, _bus_type(n_bus))
    assert float(pb.max()) < 1e-9


def test_power_balance_ignores_unspecified_quantities():
    """P at the slack and Q at PV are solved for, so they are not scored."""
    torch.manual_seed(22)
    n_bus = 8
    ybus = torch.eye(n_bus, dtype=torch.complex128)
    st = states(4, n_bus, seed=23)
    bt = _bus_type(n_bus)
    v = (st[..., 0] * torch.exp(1j * st[..., 1])).to(torch.complex128)
    s_spec = v * (v @ ybus.T).conj()
    perturbed = s_spec.clone()
    perturbed[:, bt == 1] += 5.0            # slack P and Q
    perturbed[:, bt == 2] += 5.0j           # PV Q
    assert float(csp.power_balance(st, ybus, perturbed, bt).max()) < 1e-9


def test_within_bus_r2_ignores_between_bus_spread():
    """A model that only predicts each bus's mean must score zero, not one."""
    ref = states(seed=24)
    pred = ref.clone()
    pred[..., 0] = ref[..., 0].mean(dim=0, keepdim=True).expand_as(ref[..., 0])
    assert csp.within_bus_r2_slope(pred, ref)["within_r2"] < 1e-12


def test_evaluate_reports_pb_only_when_it_can():
    ref, pred = states(seed=25), states(seed=26)
    assert "mean_pb" not in csp.evaluate(pred, ref)
