#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import traceback
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandapower as pp
import pandapower.networks as pn
from pandapower.powerflow import LoadflowNotConverged
from pandapower.pypower.idx_bus import BASE_KV
from pandapower.pypower.idx_gen import GEN_BUS, VG, GEN_STATUS

from case_generator_all_test_cases_pandapower_consider_ppc_branch_row import case_generation_pandapower
from newton_raphson_improved import newtonrapson


CASES = [
    "case1888rte",
    "case6470rte",
    "case6495rte",
    "case6515rte",
]

FORCE_SHUNT_CASES = {
    "case4gs", "case5", "case6ww", "case9", "case30", "case33bw"
}

CASE_KWARGS: Dict[str, Dict[str, Any]] = {
    # "case1888rte": {"ref_bus_idx": 1246},
    # "case6470rte": {"ref_bus_idx": 5988},
    # "case6495rte": {"ref_bus_idx": None},
    # "case6515rte": {"ref_bus_idx": 6171},
}

SCENARIO_CFG = dict(
    jitter_load=0.0,
    jitter_gen=0.0,
    pv_vset_range=(1.0, 1.0),
    rand_u_start=False,
    angle_jitter_deg=0.0,
    mag_jitter_pq=0.0,
)


# ============================================================
# helpers
# ============================================================

def is_converged_solution(u_newton) -> bool:
    a = np.asarray(u_newton)
    if a.size == 0:
        return False
    if np.allclose(a, 0.0):
        return False
    return True


def mismatch_inf_for_your_nr(bus_typ, Ybus, S_spec, U) -> float:
    """
    Infinity norm of mismatch under YOUR NR equation selection:
      - dP on all non-slack buses
      - dQ on PQ buses only
    """
    bus_typ = np.asarray(bus_typ)
    Ybus = np.asarray(Ybus, dtype=np.complex128)
    S_spec = np.asarray(S_spec, dtype=np.complex128)
    U = np.asarray(U, dtype=np.complex128)

    I = Ybus @ U
    S_calc = U * np.conj(I)

    dP = S_spec.real - S_calc.real
    dQ = S_spec.imag - S_calc.imag

    slack = (bus_typ == 1)
    pv = (bus_typ == 2)

    mis = np.concatenate([
        dP[~slack],
        dQ[(~slack) & (~pv)],
    ])

    if mis.size == 0:
        return 0.0
    return float(np.max(np.abs(mis)))


def _build_pp_to_ppc_bus_lookup(net, N: int) -> Dict[int, int]:
    """
    pandapower bus index -> PPC internal bus index
    """
    pp_to_ppc: Dict[int, int] = {}
    pp_bus_ids = net.bus.index.to_numpy(dtype=int)

    raw = None
    if hasattr(net, "_pd2ppc_lookups"):
        raw = net._pd2ppc_lookups.get("bus", None)

    if raw is not None:
        raw = np.asarray(raw).reshape(-1)

        for pp_bus in pp_bus_ids:
            if 0 <= pp_bus < len(raw):
                ppc_bus = int(raw[pp_bus])
                if 0 <= ppc_bus < N:
                    pp_to_ppc[pp_bus] = ppc_bus

        if len(pp_to_ppc) == 0 and len(raw) == len(pp_bus_ids):
            for pp_bus, ppc_bus in zip(pp_bus_ids, raw.astype(int)):
                if 0 <= int(ppc_bus) < N:
                    pp_to_ppc[int(pp_bus)] = int(ppc_bus)

    if len(pp_to_ppc) == 0:
        for k, pp_bus in enumerate(pp_bus_ids[:N]):
            pp_to_ppc[int(pp_bus)] = int(k)

    return pp_to_ppc


def apply_same_scenario_to_net(
    net,
    seed: int,
    scenario_cfg: Dict[str, Any],
    trafo_pfe_kw: Optional[float] = None,
    trafo_i0_percent: Optional[float] = None,
):
    """
    Apply the SAME scenario logic as your case_generator.
    """
    rng = np.random.default_rng(seed)

    if hasattr(net, "trafo") and net.trafo is not None and len(net.trafo):
        if trafo_pfe_kw is not None:
            net.trafo.loc[:, "pfe_kw"] = float(trafo_pfe_kw)
        if trafo_i0_percent is not None:
            net.trafo.loc[:, "i0_percent"] = float(trafo_i0_percent)

    jitter_load = float(scenario_cfg.get("jitter_load", 0.0))
    jitter_gen = float(scenario_cfg.get("jitter_gen", 0.0))
    pv_vset_range = scenario_cfg.get("pv_vset_range", None)

    if jitter_load > 0 and hasattr(net, "load") and len(net.load):
        s = rng.normal(1.0, jitter_load, size=len(net.load))
        net.load["p_mw"] = net.load["p_mw"].to_numpy(float) * s
        net.load["q_mvar"] = net.load["q_mvar"].to_numpy(float) * s

    if jitter_gen > 0 and hasattr(net, "gen") and len(net.gen):
        s = rng.normal(1.0, jitter_gen, size=len(net.gen))
        net.gen["p_mw"] = net.gen["p_mw"].to_numpy(float) * s

    if pv_vset_range is not None and hasattr(net, "gen") and len(net.gen):
        lo, hi = pv_vset_range
        net.gen["vm_pu"] = rng.uniform(lo, hi, size=len(net.gen))

    return net


def compile_internal_ppc(net, init_mode="flat"):
    """
    Compile enough to obtain net._ppc['internal'].
    """
    exc = None
    try:
        pp.runpp(
            net,
            algorithm="nr",
            init=init_mode,
            calculate_voltage_angles=True,
            max_iteration=1,
            enforce_q_lims=False,
            tolerance_mva=1e9,
        )
    except LoadflowNotConverged as e:
        exc = e
    except Exception as e:
        exc = e

    if not hasattr(net, "_ppc") or "internal" not in net._ppc:
        raise RuntimeError(f"Could not obtain internal PPC with init={init_mode}. exception={repr(exc)}")

    return net._ppc["internal"], exc


def build_manual_flat_start_from_ppc_int(ppc_int) -> np.ndarray:
    """
    Rebuild the same style of flat start as your current generator:
      - angle = 0 for all buses
      - magnitude = 1.0 pu
      - overwrite online gen buses with VG
    Returns SI voltage.
    """
    bus_ppc = np.asarray(ppc_int["bus"], dtype=float)
    gen_ppc = np.asarray(ppc_int.get("gen", np.empty((0, 0))), dtype=float)

    if gen_ppc.ndim == 1 and gen_ppc.size == 0:
        gen_ppc = gen_ppc.reshape(0, 0)

    Vbase = bus_ppc[:, BASE_KV].astype(float) * 1e3
    N = len(bus_ppc)

    mag = np.ones(N, dtype=np.float64)
    ang = np.zeros(N, dtype=np.float64)

    if gen_ppc.size > 0 and gen_ppc.shape[1] > max(GEN_BUS, VG, GEN_STATUS):
        on = gen_ppc[:, GEN_STATUS] > 0
        gbus = gen_ppc[on, GEN_BUS].astype(int)
        vg = gen_ppc[on, VG].astype(float)
        for b, vm in zip(gbus, vg):
            if 0 <= b < N and np.isfinite(vm) and vm > 0:
                mag[b] = vm

    return (mag * np.exp(1j * ang)) * Vbase


def build_dc_start_from_net(net) -> Tuple[np.ndarray, Any]:
    """
    Build a DC-style start without relying on ppc_int['V0'].
    Uses:
      - bus angles from pandapower rundcpp()
      - magnitudes from compiled gen VG (else 1.0 pu)
    Returns SI voltage and the internal ppc after DC solve.
    """
    # DC solve
    pp.rundcpp(net)

    if not hasattr(net, "_ppc") or "internal" not in net._ppc:
        raise RuntimeError("Could not obtain internal PPC after rundcpp().")

    ppc_int = net._ppc["internal"]
    bus_ppc = np.asarray(ppc_int["bus"], dtype=float)
    gen_ppc = np.asarray(ppc_int.get("gen", np.empty((0, 0))), dtype=float)

    if gen_ppc.ndim == 1 and gen_ppc.size == 0:
        gen_ppc = gen_ppc.reshape(0, 0)

    N = len(bus_ppc)
    Vbase = bus_ppc[:, BASE_KV].astype(float) * 1e3

    # magnitudes: 1.0 pu, except online gen buses get VG
    mag = np.ones(N, dtype=np.float64)
    if gen_ppc.size > 0 and gen_ppc.shape[1] > max(GEN_BUS, VG, GEN_STATUS):
        on = gen_ppc[:, GEN_STATUS] > 0
        gbus = gen_ppc[on, GEN_BUS].astype(int)
        vg = gen_ppc[on, VG].astype(float)
        for b, vm in zip(gbus, vg):
            if 0 <= b < N and np.isfinite(vm) and vm > 0:
                mag[b] = vm

    # angles from net.res_bus in pandapower order -> map to PPC order
    ang = np.zeros(N, dtype=np.float64)
    pp_to_ppc = _build_pp_to_ppc_bus_lookup(net, N)

    if not hasattr(net, "res_bus") or "va_degree" not in net.res_bus.columns:
        raise RuntimeError("net.res_bus.va_degree not available after rundcpp().")

    for pp_bus, row in net.res_bus.iterrows():
        ppc_bus = pp_to_ppc.get(int(pp_bus), None)
        if ppc_bus is not None and np.isfinite(row["va_degree"]):
            ang[ppc_bus] = np.deg2rad(float(row["va_degree"]))

    U0_pu = mag * np.exp(1j * ang)
    U0_si = U0_pu * Vbase
    return U0_si, ppc_int


def solve_pandapower_fully(net):
    """
    Full pandapower NR solve for reference solved voltage.
    """
    pp.runpp(
        net,
        algorithm="nr",
        init="dc",
        calculate_voltage_angles=True,
        enforce_q_lims=False,
    )

    ppc_int = net._ppc["internal"]
    bus_ppc = np.asarray(ppc_int["bus"], dtype=float)
    Vbase = bus_ppc[:, BASE_KV].astype(float) * 1e3

    # Prefer internal V if available; else fallback to ppc bus VM/VA if exposed;
    # else use net.res_bus mapped to PPC order.
    if "V" in ppc_int:
        U_pu = np.asarray(ppc_int["V"], dtype=np.complex128).reshape(-1)
        U_si = U_pu * Vbase
        return U_si, ppc_int

    # fallback via net.res_bus
    N = len(bus_ppc)
    pp_to_ppc = _build_pp_to_ppc_bus_lookup(net, N)
    mag = np.ones(N, dtype=np.float64)
    ang = np.zeros(N, dtype=np.float64)

    for pp_bus, row in net.res_bus.iterrows():
        ppc_bus = pp_to_ppc.get(int(pp_bus), None)
        if ppc_bus is not None:
            mag[ppc_bus] = float(row["vm_pu"])
            ang[ppc_bus] = np.deg2rad(float(row["va_degree"]))

    U_si = (mag * np.exp(1j * ang)) * Vbase
    return U_si, ppc_int


def run_your_nr(bus_typ, Y_matrix, s_multi, u_start, K=40):
    u_newton, I_newton, S_newton = newtonrapson(
        np.asarray(bus_typ).copy(),
        np.asarray(Y_matrix, dtype=np.complex128).copy(),
        np.asarray(s_multi, dtype=np.complex128).copy(),
        np.asarray(u_start, dtype=np.complex128).copy(),
        K=K,
    )
    return u_newton, I_newton, S_newton


# ============================================================
# main comparison
# ============================================================

def compare_case(case_name: str, seed: int = 1000, K: int = 40):
    print("=" * 120)
    print(f"[{case_name}]")

    fn = getattr(pn, case_name)
    case_kwargs = CASE_KWARGS.get(case_name, {}).copy()

    force = None
    if case_name in FORCE_SHUNT_CASES:
        force = {"g": 0.0, "b": 0.2, "g_asym": 0.0, "b_asym": 0.0}

    # ------------------------------------------------------------
    # A) Your current equations + your current manual_flat from generator
    # ------------------------------------------------------------
    (
        gridtype,
        bus_typ,
        s_multi,
        u_start_manual_flat,
        Y_matrix,
        is_connected,
        Branch_f_bus,
        Branch_t_bus,
        Branch_status,
        Branch_tau,
        Branch_shift_deg,
        Branch_y_series_from,
        Branch_y_series_to,
        Branch_y_series_ft,
        Branch_y_shunt_from,
        Branch_y_shunt_to,
        Y_shunt_bus,
        Is_trafo,
        Branch_hv_is_f,
        Branch_n,
        Y_Lines,
        Y_C_Lines,
        U_base,
        S_base,
        vn_kv,
    ) = case_generation_pandapower(
        case_fn=fn,
        case_kwargs=case_kwargs,
        ybus_mode="ppcY",
        seed=seed,
        force_branch_shunt_pu=force,
        **SCENARIO_CFG,
    )

    if not is_connected:
        print("generator returned not connected")
        return

    bus_typ = np.asarray(bus_typ, dtype=np.int64)
    s_multi = np.asarray(s_multi, dtype=np.complex128)
    Y_matrix = np.asarray(Y_matrix, dtype=np.complex128)
    u_start_manual_flat = np.asarray(u_start_manual_flat, dtype=np.complex128)
    Vbase = np.asarray(vn_kv, dtype=np.float64) * 1e3
    N = len(bus_typ)

    print(f"N={N} | slack={np.sum(bus_typ==1)} pv={np.sum(bus_typ==2)} pq={np.sum(bus_typ==3)}")

    # ------------------------------------------------------------
    # B) "flat compile" equivalent start
    # ------------------------------------------------------------
    net_flat = apply_same_scenario_to_net(
        fn(**case_kwargs),
        seed=seed,
        scenario_cfg=SCENARIO_CFG,
    )
    ppc_int_flat, exc_flat = compile_internal_ppc(net_flat, init_mode="flat")
    u_start_flat_compile_equiv = build_manual_flat_start_from_ppc_int(ppc_int_flat)

    # ------------------------------------------------------------
    # C) DC start
    # ------------------------------------------------------------
    net_dc = apply_same_scenario_to_net(
        fn(**case_kwargs),
        seed=seed,
        scenario_cfg=SCENARIO_CFG,
    )
    u_start_dc_compile, ppc_int_dc = build_dc_start_from_net(net_dc)

    # ------------------------------------------------------------
    # D) Fully solved pandapower voltage for distance reference
    # ------------------------------------------------------------
    net_sol = apply_same_scenario_to_net(
        fn(**case_kwargs),
        seed=seed,
        scenario_cfg=SCENARIO_CFG,
    )
    U_pp_sol_si, ppc_int_sol = solve_pandapower_fully(net_sol)

    starts = {
        "manual_flat": u_start_manual_flat,
        "flat_compile_equiv": u_start_flat_compile_equiv,
        "dc_compile_start": u_start_dc_compile,
    }

    print("\nSide-by-side starts under YOUR SAME fixed equations:")
    print("start_name            | start_misinf        | dist_to_pp_sol_pu_inf | NR_result")
    print("-" * 100)

    for start_name, U0 in starts.items():
        mis0 = mismatch_inf_for_your_nr(bus_typ, Y_matrix, s_multi, U0)
        dist_pu = float(np.max(np.abs((U0 - U_pp_sol_si) / Vbase)))

        try:
            u_nr, _, _ = run_your_nr(
                bus_typ=bus_typ,
                Y_matrix=Y_matrix,
                s_multi=s_multi,
                u_start=U0,
                K=K,
            )
            conv = is_converged_solution(u_nr)

            if conv:
                mis_final = mismatch_inf_for_your_nr(bus_typ, Y_matrix, s_multi, u_nr)
                nr_result = f"converged | final_misinf={mis_final:.12g}"
            else:
                nr_result = "failed"
        except Exception as e:
            nr_result = f"exception | {repr(e)}"

        print(
            f"{start_name:21s} | "
            f"{mis0:<19.12g} | "
            f"{dist_pu:<19.12g} | "
            f"{nr_result}"
        )

    print("\nNotes:")
    print(f"  compile-time runpp(flat) exception = {repr(exc_flat)}")
    print("  manual_flat and flat_compile_equiv should be nearly identical.")
    print("  dc_compile_start is the important extra diagnostic.")
    print("  If dc_compile_start converges while flat does not, the issue is mainly basin of attraction.")


def main():
    print("=" * 120)
    print("Compare 3 starts side by side for the hard RTE cases")
    print("Starts tested:")
    print("  1) manual_flat         = your current case_generator u_start")
    print("  2) flat_compile_equiv  = reconstructed flat compile start from PPC")
    print("  3) dc_compile_start    = DC-angle start from pandapower rundcpp()")
    print("=" * 120)

    for case_name in CASES:
        try:
            compare_case(case_name, seed=1000, K=40)
        except Exception as e:
            print("=" * 120)
            print(f"[{case_name}] FAILED")
            print(repr(e))
            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()