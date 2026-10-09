import numpy as np
import traceback


def _compute_mismatch_inf(bus_typ, Y_admittance, U, P, Q):
    """
    Infinity norm of the NR mismatch vector under the same slack/PV/PQ rules
    used by the solver.

    bus_typ:
      1 = slack
      2 = PV
      3 = PQ
    """
    I = Y_admittance @ U
    S_calc = U * np.conj(I)

    slack = (bus_typ == 1)
    pq = (bus_typ == 3)

    dP = P - S_calc.real
    dQ = Q - S_calc.imag

    mis = np.concatenate([dP[~slack], dQ[pq]])
    if mis.size == 0:
        return 0.0
    return float(np.max(np.abs(mis)))


def _classify_nonconvergence(
    misinf_hist,
    step_hist,
    near_misinf_tol=1e-3,
):
    """
    Heuristic classification for non-converged NR runs.

    Returns one of:
      - divergence
      - stagnation
      - slow_convergence
      - near_convergence_not_passing_step
      - no_progress_data
    """
    if len(misinf_hist) == 0:
        return "no_progress_data"

    first = float(misinf_hist[0])
    last = float(misinf_hist[-1])
    best = float(np.min(misinf_hist))

    eps = 1e-15
    first_safe = max(first, eps)

    # 1) very small mismatch already achieved, but step criterion did not fire
    if (last <= near_misinf_tol) or (best <= near_misinf_tol):
        return "near_convergence_not_passing_step"

    # 2) clear blow-up / oscillatory growth
    grow_ratio = last / first_safe
    inc_count = sum(
        misinf_hist[i] > 1.05 * misinf_hist[i - 1]
        for i in range(1, len(misinf_hist))
    )
    if grow_ratio >= 2.0:
        return "divergence"
    if inc_count >= max(3, len(misinf_hist) // 2) and last > 1.2 * best:
        return "divergence"

    # 3) barely improving
    total_improvement = (first - last) / first_safe
    recent = misinf_hist[-min(5, len(misinf_hist)):]
    recent_drop = (recent[0] - recent[-1]) / max(recent[0], eps)

    if total_improvement < 0.10:
        return "stagnation"
    if recent_drop < 0.02 and total_improvement < 0.50:
        return "stagnation"

    # 4) improving, but not enough within K iterations
    return "slow_convergence"


def VoltageCalculation3p(bus_typ, JMatrix, Y_admittance, U, P, Q):
    """
    General NR voltage update that does NOT assume slack bus is index 0.
    Works with arbitrary PPC bus ordering.

    bus_typ convention:
      1 = slack
      2 = PV
      3 = PQ
    """
    N, _ = Y_admittance.shape

    I = Y_admittance @ U
    Ss = U * np.conj(I)

    slack = (bus_typ == 1)
    pq = (bus_typ == 3)

    # State-variable bus sets
    ang_buses = np.where(~slack)[0]   # angle unknowns for all non-slack buses
    vm_buses = np.where(pq)[0]        # magnitude unknowns for PQ buses only

    # Mismatch
    dP = P - Ss.real
    dQ = Q - Ss.imag
    mismatch = np.concatenate([dP[ang_buses], dQ[vm_buses]])

    # Reduced Jacobian
    keep = np.concatenate([ang_buses, N + vm_buses])
    J_red = JMatrix[np.ix_(keep, keep)]

    try:
        dx = np.linalg.solve(J_red, mismatch)
    except np.linalg.LinAlgError:
        print("Fehler (Jacobian singular)")
        raise

    n_ang = len(ang_buses)
    dth = dx[:n_ang]
    dvm = dx[n_ang:]

    U_new = U.copy()

    # update angles of all non-slack buses
    ang = np.angle(U_new)
    ang[ang_buses] += dth

    # update magnitudes of PQ buses only
    mag = np.abs(U_new)
    mag[vm_buses] += dvm

    U_new = mag * np.exp(1j * ang)

    I_new = Y_admittance @ U_new
    S_new = U_new * np.conj(I_new)

    return U_new, S_new


def JacobianMatrix3p(Y_admittance, U):
    """
    Vectorized Jacobian assembly (no Python loops).
    """
    N, M = Y_admittance.shape
    idx = np.arange(N)

    absU = np.abs(U)
    angU = np.angle(U)

    Ymag = np.abs(Y_admittance)
    Yang = np.angle(Y_admittance)

    theta = angU[:, None] - angU[None, :] - Yang
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    Ymag_d = np.diag(Ymag)
    Yang_d = np.diag(Yang)
    absU_col = absU[:, None]
    absU_row = absU[None, :]

    # dP/d|U|
    J2 = absU_col * Ymag * cos_th
    row_sum_cosu = (Ymag * absU_row * cos_th).sum(axis=1)
    diag_add = absU * Ymag_d * np.cos(Yang_d)
    J2[idx, idx] = row_sum_cosu + diag_add

    # dP/dθ
    base_J1 = (absU_col * absU_row) * Ymag * sin_th
    J1 = base_J1.copy()
    row_sum = base_J1.sum(axis=1)
    diag_v = np.diag(base_J1)
    J1[idx, idx] = -(row_sum - diag_v)

    # dQ/d|U|
    J4 = absU_col * Ymag * sin_th
    row_sum_sinUk = (Ymag * absU_row * sin_th).sum(axis=1)
    J4[idx, idx] = row_sum_sinUk - absU * Ymag_d * np.sin(Yang_d)

    # dQ/dθ
    base_J3 = (absU_col * absU_row) * Ymag * cos_th
    J3 = -base_J3
    row_sum_c = base_J3.sum(axis=1)
    diag_c = np.diag(base_J3)
    J3[idx, idx] = row_sum_c - diag_c

    J = np.block([[J1, J2],
                  [J3, J4]])
    return J, J1, J2, J3, J4


def _scaled_mismatch_tol(S_spec, mismatch_tol, bus_typ=None, min_power_scale=1.0):
    """
    Relative mismatch tolerance: mismatch_tol * (largest specified injection).

    The scale must be taken over the quantities the mismatch vector actually
    tests -- P at non-slack buses and Q at PQ buses -- and nothing else.
    Q at a PV bus is not a constraint (the solver fixes |V| and solves for Q),
    so whatever sits in that slot is arbitrary. CGMES exports routinely park a
    reactive-capability figure there: on LVN heo1 one 380 kV PV bus carries
    Q_spec = -2507 pu against a true maximum of 8.4 pu, which inflated the
    tolerance 297x and let the solver stop far short of its intended accuracy.

    bus_typ is optional for backward compatibility; without it the old
    all-entries behaviour is kept.
    """
    S_spec = np.asarray(S_spec)

    if bus_typ is not None and S_spec.size:
        bus_typ = np.asarray(bus_typ)
        slack = (bus_typ == 1)
        pq = (bus_typ == 3)
        enforced = np.concatenate([S_spec.real[~slack], S_spec.imag[pq]])
    else:
        enforced = S_spec

    power_scale = max(
        min_power_scale,
        float(np.max(np.abs(enforced))) if enforced.size else 0.0,
    )
    return float(mismatch_tol * power_scale), power_scale


def newtonrapson(
    bus_typ,
    Y_system,
    s_L,
    U,
    K=40,
    diagnose=False,
    print_misinf=False,
    return_diagnostics=False,
    near_misinf_tol=1e-3,
    convergence_mode="misinf",
    step_tol=5e-4,
    mismatch_tol=1e-8,
    return_trajectory=False,
    verbose=True,
):
    """
    Newton-Raphson without assuming slack bus = index 0.

    Optional arguments:
      diagnose:
          store mismatch/step histories and classify failed runs
      print_misinf:
          print mismatch infinity norm each iteration
      return_diagnostics:
          return a 4th output dict with detailed diagnostics
      near_misinf_tol:
          threshold used for the
          'near_convergence_not_passing_step' classification
      convergence_mode:
          "two_step" -> original two-step voltage-change criterion
          "misinf"   -> mismatch-infinity-norm criterion
      step_tol:
          threshold for the two-step voltage-change criterion
      mismatch_tol:
          threshold for the mismatch-infinity-norm criterion
      return_trajectory:
          include the initial voltage and every full Newton iterate in
          ``diag['voltage_trajectory']``.  The array has shape [T, N], uses
          complex128, and satisfies T = iterations + 1 for a normal run.
          Histories are recorded whenever this option is enabled, even when
          ``diagnose`` is False.
      verbose:
          retain the historical per-case progress printing.  Dataset
          augmentation can disable it to avoid millions of log characters.

    Default behavior remains the same:
      - default convergence is still the original two-step criterion
      - failed runs are still zeroed
      - return is still (u_final, I, S) unless return_diagnostics=True
    """
    if convergence_mode not in ("two_step", "misinf"):
        raise ValueError(
            f"Unknown convergence_mode={convergence_mode!r}. "
            "Use 'two_step' or 'misinf'."
        )

    diag = {
        "converged": False,
        "classification": None,
        "failure_reason": None,
        "iterations": 0,
        "misinf_history": [],
        "step_history": [],
        "final_misinf": None,
        "best_misinf": None,
        "voltage_trajectory": None,
    }

    try:
        bus_typ = np.asarray(bus_typ)
        Y_system = np.asarray(Y_system, dtype=np.complex128)
        S_spec = np.asarray(s_L, dtype=np.complex128)
        U = np.asarray(U, dtype=np.complex128).copy()
        voltage_trajectory = [U.copy()] if return_trajectory else None

        effective_mismatch_tol, power_scale = _scaled_mismatch_tol(
            S_spec=S_spec,
            mismatch_tol=mismatch_tol,
            bus_typ=bus_typ,
        )

        P = S_spec.real
        Q = S_spec.imag

        N, _ = Y_system.shape
        if verbose:
            print(f"{N}- Bus", end="", flush=True)

        prev2 = None
        prev1 = None
        converged = False
        failure_reason = "max_iterations_reached"

        misinf_hist = []
        step_hist = []

        for it in range(K):
            J, _, _, _, _ = JacobianMatrix3p(Y_system, U)
            if verbose:
                print(".", end="", flush=True)

            try:
                Us, _ = VoltageCalculation3p(bus_typ, J, Y_system, U, P, Q)
            except np.linalg.LinAlgError:
                if verbose:
                    print("|Jacobian singular, abort NR|")
                failure_reason = "jacobian_singular"
                converged = False
                break

            step_inf = float(np.max(np.abs(Us - U)))
            misinf = _compute_mismatch_inf(bus_typ, Y_system, Us, P, Q)

            if diagnose or return_trajectory:
                misinf_hist.append(misinf)
                step_hist.append(step_inf)

            if return_trajectory:
                voltage_trajectory.append(np.asarray(Us, dtype=np.complex128).copy())

            if print_misinf and verbose:
                print(f"[misinf={misinf:.3e}]", end="", flush=True)

            if convergence_mode == "misinf":
                if misinf < effective_mismatch_tol:
                    if verbose:
                        print("|converged successfully (misinf)|")
                    U = Us
                    converged = True
                    failure_reason = None
                    break

            elif convergence_mode == "two_step":
                if prev2 is not None and prev1 is not None:
                    if (
                        np.max(np.abs(Us - prev1)) < step_tol
                        and np.max(np.abs(prev1 - prev2)) < step_tol
                    ):
                        if verbose:
                            print("|converged successfully|")
                        U = Us
                        converged = True
                        failure_reason = None
                        break

            prev2 = prev1
            prev1 = Us
            U = Us
        else:
            if verbose:
                print("|not-converged|")

        I = Y_system @ U
        S = U * np.conj(I)

        # keep zeroing behavior
        u_final = U if converged else (U * 0)

        diag["converged"] = converged
        diag["failure_reason"] = failure_reason
        diag["iterations"] = (
            len(misinf_hist)
            if (diagnose or return_trajectory)
            else (it + 1 if "it" in locals() else 0)
        )

        if return_trajectory:
            diag["voltage_trajectory"] = np.stack(voltage_trajectory, axis=0)

        if diagnose or return_trajectory:
            diag["misinf_history"] = misinf_hist
            diag["step_history"] = step_hist
            diag["final_misinf"] = misinf_hist[-1] if len(misinf_hist) else None
            diag["best_misinf"] = float(np.min(misinf_hist)) if len(misinf_hist) else None
            diag["power_scale"] = power_scale
            diag["effective_mismatch_tol"] = effective_mismatch_tol

            if converged:
                diag["classification"] = "converged"
            elif failure_reason == "jacobian_singular":
                diag["classification"] = "jacobian_singular"
            else:
                diag["classification"] = _classify_nonconvergence(
                    misinf_hist=misinf_hist,
                    step_hist=step_hist,
                    near_misinf_tol=near_misinf_tol,
                )
        else:
            diag["classification"] = "converged" if converged else failure_reason

    except Exception as e:
        if verbose:
            print(f"Fail in newtonrapson: {str(e)}")
            traceback.print_exc()
        u_final = []
        I = []
        S = []

        diag["converged"] = False
        diag["classification"] = "exception"
        diag["failure_reason"] = repr(e)
        diag["iterations"] = 0

    if return_diagnostics:
        return u_final, I, S, diag
    return u_final, I, S
