#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# It does the following for each case:
# - builds a compile-time / early internal PPC using your same style of 1-iteration runpp
# - builds a fully converged internal PPC using pandapower NR with init="dc"
# - computes the mismatch of the compile-time equations evaluated at the converged pandapower voltage
# - finds the top mismatch buses
# - compares, at those buses:
#     - BUS_TYPE
#     - whether the bus appears in gen [:, GEN_BUS]
#     - how many on-line generators are connected there
#     - Sbus from compile-time PPC vs converged PPC
#     - initial voltage from compile-time PPC vs solved voltage from converged PPC
#     - prints a global summary of how many buses changed type / generator membership

import copy
import numpy as np
import pandapower as pp
import pandapower.networks as pn

from pandapower.powerflow import LoadflowNotConverged
from pandapower.pypower.makeSbus import makeSbus
from pandapower.pypower.idx_bus import BUS_TYPE, VM, VA
from pandapower.pypower.idx_gen import GEN_BUS, GEN_STATUS


CASES = [
    "case1888rte",
    "case6470rte",
    "case6495rte",
    "case6515rte",
]

# Optional special kwargs if needed
CASE_KWARGS = {
    # "case1888rte": {"ref_bus_idx": 1246},
    # "case6470rte": {"ref_bus_idx": 5988},
    # "case6515rte": {"ref_bus_idx": 6171},
}

# Keep these aligned with what you used in your generator
TRAFO_PFE_KW = None
TRAFO_I0_PERCENT = None


def _dense_ybus(ppc_int):
    Y = ppc_int["Ybus"]
    if hasattr(Y, "toarray"):
        return Y.toarray().astype(np.complex128)
    return np.asarray(Y, dtype=np.complex128)


def _make_sbus_pu(ppc_int):
    baseMVA = float(ppc_int["baseMVA"])
    bus = np.asarray(ppc_int["bus"], dtype=float)
    gen = np.asarray(ppc_int.get("gen", np.empty((0, 0))), dtype=float)
    if gen.ndim == 1 and gen.size == 0:
        gen = gen.reshape(0, 0)
    Sbus = makeSbus(baseMVA, bus, gen)
    return np.asarray(Sbus).reshape(-1).astype(np.complex128)


def _extract_u_pu_from_bus(ppc_int):
    bus = np.asarray(ppc_int["bus"], dtype=float)
    vm = bus[:, VM].astype(float)
    va_deg = bus[:, VA].astype(float)
    return vm * np.exp(1j * np.deg2rad(va_deg))


def _pp_bus_type_name(x: int) -> str:
    mapping = {
        1: "PQ",
        2: "PV",
        3: "REF",
        4: "NONE",
    }
    return mapping.get(int(x), f"UNK({int(x)})")


def _bus_gen_info(ppc_int):
    """
    Returns per-bus generator membership info from compiled internal ppc gen table.
    """
    bus = np.asarray(ppc_int["bus"], dtype=float)
    N = bus.shape[0]

    gen = np.asarray(ppc_int.get("gen", np.empty((0, 0))), dtype=float)
    if gen.ndim == 1 and gen.size == 0:
        gen = gen.reshape(0, 0)

    any_gen_count = np.zeros(N, dtype=int)
    on_gen_count = np.zeros(N, dtype=int)

    if gen.size > 0 and gen.shape[1] > max(GEN_BUS, GEN_STATUS):
        gbus = gen[:, GEN_BUS].astype(int)
        gstat = gen[:, GEN_STATUS].astype(float)

        for b in gbus:
            if 0 <= b < N:
                any_gen_count[b] += 1

        for b, st in zip(gbus, gstat):
            if 0 <= b < N and st > 0:
                on_gen_count[b] += 1

    return {
        "any_gen_count": any_gen_count,
        "on_gen_count": on_gen_count,
        "has_any_gen": any_gen_count > 0,
        "has_on_gen": on_gen_count > 0,
    }


def _power_mismatch_inf_by_bus(Ybus_pu, U_pu, Sbus_pu, ppc_bus_type):
    """
    Mismatch under fixed compiled equations, respecting pypower bus types:
      1=PQ, 2=PV, 3=REF

    Returned arrays are per bus:
      dP used on PV/PQ, zero on REF
      dQ used on PQ only, zero on PV/REF
      mis_bus = max(|dP_used|, |dQ_used|)
    """
    I = Ybus_pu @ U_pu
    S_calc = U_pu * np.conj(I)

    dP = Sbus_pu.real - S_calc.real
    dQ = Sbus_pu.imag - S_calc.imag

    used_dP = dP.copy()
    used_dQ = dQ.copy()

    is_ref = (ppc_bus_type == 3)
    is_pv = (ppc_bus_type == 2)
    is_pq = (ppc_bus_type == 1)

    used_dP[is_ref] = 0.0
    used_dQ[~is_pq] = 0.0

    mis_bus = np.maximum(np.abs(used_dP), np.abs(used_dQ))
    mis_inf = float(np.max(mis_bus)) if mis_bus.size else 0.0

    return mis_inf, mis_bus, used_dP, used_dQ


def _compile_internal_ppc_once(net):
    """
    Build the internal ppc in the same spirit as your generator:
    compile with a 1-iteration runpp attempt.
    """
    runpp_exc = None
    try:
        pp.runpp(
            net,
            algorithm="nr",
            init="flat",
            calculate_voltage_angles=True,
            max_iteration=1,
            enforce_q_lims=False,
            tolerance_mva=1e9,
        )
    except LoadflowNotConverged as e:
        runpp_exc = e
    except Exception as e:
        runpp_exc = e

    if not hasattr(net, "_ppc") or "internal" not in net._ppc:
        raise RuntimeError(f"Failed to build internal PPC in compile step: {repr(runpp_exc)}")

    return net._ppc["internal"], runpp_exc


def _solve_converged_internal_ppc(net):
    """
    Run a real pandapower solve and return the converged internal ppc.
    """
    pp.runpp(
        net,
        algorithm="nr",
        init="dc",
        calculate_voltage_angles=True,
        enforce_q_lims=False,
        max_iteration=50,
        tolerance_mva=1e-8,
    )

    if not hasattr(net, "_ppc") or "internal" not in net._ppc:
        raise RuntimeError("Converged runpp finished but internal PPC is missing.")

    return net._ppc["internal"]


def _apply_optional_trafo_params(net):
    if hasattr(net, "trafo") and net.trafo is not None and len(net.trafo):
        if TRAFO_PFE_KW is not None:
            net.trafo.loc[:, "pfe_kw"] = float(TRAFO_PFE_KW)
        if TRAFO_I0_PERCENT is not None:
            net.trafo.loc[:, "i0_percent"] = float(TRAFO_I0_PERCENT)


def analyze_case(case_name: str, topk: int = 15):
    fn = getattr(pn, case_name)
    case_kwargs = CASE_KWARGS.get(case_name, {}).copy()

    net0 = fn(**case_kwargs)
    _apply_optional_trafo_params(net0)

    net_compile = copy.deepcopy(net0)
    net_conv = copy.deepcopy(net0)

    # 1) compile-time internal ppc
    ppc_compile, compile_exc = _compile_internal_ppc_once(net_compile)

    # 2) fully converged internal ppc
    ppc_conv = _solve_converged_internal_ppc(net_conv)

    # 3) exact quantities from both
    bus_compile = np.asarray(ppc_compile["bus"], dtype=float)
    bus_conv = np.asarray(ppc_conv["bus"], dtype=float)

    N = bus_compile.shape[0]
    if bus_conv.shape[0] != N:
        raise RuntimeError(
            f"{case_name}: bus count differs between compile-time and converged PPC "
            f"({N} vs {bus_conv.shape[0]})."
        )

    Y_compile = _dense_ybus(ppc_compile)
    S_compile = _make_sbus_pu(ppc_compile)
    U_compile0 = _extract_u_pu_from_bus(ppc_compile)

    Y_conv = _dense_ybus(ppc_conv)
    S_conv = _make_sbus_pu(ppc_conv)
    U_conv = _extract_u_pu_from_bus(ppc_conv)

    bt_compile = bus_compile[:, BUS_TYPE].astype(int)
    bt_conv = bus_conv[:, BUS_TYPE].astype(int)

    geninfo_compile = _bus_gen_info(ppc_compile)
    geninfo_conv = _bus_gen_info(ppc_conv)

    # 4) mismatch of compile-time equations at:
    #    (a) compile-time initial voltage
    #    (b) converged pandapower voltage
    start_misinf, start_mis_bus, start_dP, start_dQ = _power_mismatch_inf_by_bus(
        Y_compile, U_compile0, S_compile, bt_compile
    )

    solved_misinf_under_compile, mis_bus, dP_used, dQ_used = _power_mismatch_inf_by_bus(
        Y_compile, U_conv, S_compile, bt_compile
    )

    # 5) mismatch of converged ppc equations at converged voltage
    solved_misinf_under_conv, _, _, _ = _power_mismatch_inf_by_bus(
        Y_conv, U_conv, S_conv, bt_conv
    )

    # 6) global differences
    changed_bus_type = np.where(bt_compile != bt_conv)[0]
    changed_has_any_gen = np.where(geninfo_compile["has_any_gen"] != geninfo_conv["has_any_gen"])[0]
    changed_has_on_gen = np.where(geninfo_compile["has_on_gen"] != geninfo_conv["has_on_gen"])[0]
    changed_sbus = np.where(np.abs(S_compile - S_conv) > 1e-10)[0]

    # 7) top mismatch buses under compile-time equations at converged voltage
    order = np.argsort(-mis_bus)
    top = order[:topk]

    print("=" * 130)
    print(f"[{case_name}]")
    print(
        f"N={N} | "
        f"compile_start_misinf={start_misinf:.12g} | "
        f"compile_eq_at_convU_misinf={solved_misinf_under_compile:.12g} | "
        f"conv_eq_at_convU_misinf={solved_misinf_under_conv:.12g}"
    )
    print(
        f"compile-time runpp exception = {repr(compile_exc)}"
    )
    print(
        f"changed BUS_TYPE buses     : {len(changed_bus_type)}\n"
        f"changed has_any_gen buses  : {len(changed_has_any_gen)}\n"
        f"changed has_on_gen buses   : {len(changed_has_on_gen)}\n"
        f"changed Sbus buses         : {len(changed_sbus)}"
    )

    if len(changed_bus_type) > 0:
        print("First BUS_TYPE-changed buses:", changed_bus_type[:20].tolist())
    if len(changed_has_on_gen) > 0:
        print("First has_on_gen-changed buses:", changed_has_on_gen[:20].tolist())
    if len(changed_sbus) > 0:
        print("First Sbus-changed buses:", changed_sbus[:20].tolist())

    print("\nTop mismatch buses under COMPILE-TIME equations at CONVERGED pandapower voltage:")
    print(
        "bus | mis | dP_used | dQ_used | "
        "BT_compile -> BT_conv | "
        "anyGen_compile -> anyGen_conv | "
        "onGen_compile -> onGen_conv | "
        "Sbus_compile -> Sbus_conv | "
        "V0_compile(pu) -> V_conv(pu)"
    )

    for b in top:
        print(
            f"{b:5d} | "
            f"{mis_bus[b]:.12g} | "
            f"{dP_used[b]:.12g} | "
            f"{dQ_used[b]:.12g} | "
            f"{_pp_bus_type_name(bt_compile[b]):>4s} -> {_pp_bus_type_name(bt_conv[b]):<4s} | "
            f"{int(geninfo_compile['has_any_gen'][b])} -> {int(geninfo_conv['has_any_gen'][b])} | "
            f"{geninfo_compile['on_gen_count'][b]:2d} -> {geninfo_conv['on_gen_count'][b]:2d} | "
            f"{S_compile[b]:.12g} -> {S_conv[b]:.12g} | "
            f"{U_compile0[b]:.12g} -> {U_conv[b]:.12g}"
        )

    print("\nIntersection checks on top mismatch buses:")
    top_set = set(int(x) for x in top.tolist())
    print("top ∩ changed_bus_type    =", sorted(top_set.intersection(set(changed_bus_type.tolist())))[:50])
    print("top ∩ changed_has_on_gen  =", sorted(top_set.intersection(set(changed_has_on_gen.tolist())))[:50])
    print("top ∩ changed_sbus        =", sorted(top_set.intersection(set(changed_sbus.tolist())))[:50])

    print()


def main():
    for case_name in CASES:
        try:
            analyze_case(case_name, topk=15)
        except Exception as e:
            print("=" * 130)
            print(f"[{case_name}] FAILED: {repr(e)}")
            print()


if __name__ == "__main__":
    main()