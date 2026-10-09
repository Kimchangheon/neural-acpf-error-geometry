import numpy as np

from newton_raphson_improved import _compute_mismatch_inf, newtonrapson


def test_newton_trajectory_contains_start_and_every_update():
    # Two-bus slack/PQ system with a modest load and a flat initial voltage.
    y = 10.0 - 20.0j
    Y = np.array([[y, -y], [-y, y]], dtype=np.complex128)
    bus_type = np.array([1, 3], dtype=np.int64)
    S = np.array([0.0 + 0.0j, -0.5 - 0.2j], dtype=np.complex128)
    U0 = np.ones(2, dtype=np.complex128)

    U, _I, _S, diag = newtonrapson(
        bus_type,
        Y,
        S,
        U0,
        K=20,
        diagnose=False,
        return_diagnostics=True,
        return_trajectory=True,
        convergence_mode="misinf",
        mismatch_tol=1e-10,
        verbose=False,
    )

    trajectory = diag["voltage_trajectory"]
    assert diag["converged"]
    assert trajectory.dtype == np.complex128
    assert trajectory.shape == (diag["iterations"] + 1, 2)
    np.testing.assert_array_equal(trajectory[0], U0)
    np.testing.assert_allclose(trajectory[-1], U, rtol=0.0, atol=0.0)
    assert len(diag["misinf_history"]) == diag["iterations"]
    assert len(diag["step_history"]) == diag["iterations"]

    recomputed = [
        _compute_mismatch_inf(bus_type, Y, state, S.real, S.imag)
        for state in trajectory[1:]
    ]
    np.testing.assert_allclose(
        recomputed, diag["misinf_history"], rtol=1e-13, atol=1e-13
    )
