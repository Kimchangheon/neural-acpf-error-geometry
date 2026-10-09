#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare custom-NR initial mismatch vs pandapower solved-state mismatch
for the 4 failing RTE cases.

What this does:
1) builds the case in pandapower
2) compiles PPC/internal Ybus
3) constructs the SAME kind of custom-NR inputs you use:
     - bus_typ in PPC order
     - S_spec in PPC order
     - flat start U_start = 1∠0 pu in PPC order
4) computes mismatch_inf at:
     - your flat/default start
     - pandapower's converged voltage (if runpp converges)
5) prints a compact diagnosis

This is meant to answer:
"Does pandapower have a good solved state for this case while my custom NR
 diverges from flat start?"

If yes, that points to NR robustness / basin-of-attraction issues.
If no, the case is hard even for pandapower under that solve setup.
"""

import numpy as np
import pandapower as pp
import pandapower.networks as pn
from pandapower.powerflow import LoadflowNotConverged

# ---------------------------------------------------------------------
# cases to inspect
# ---------------------------------------------------------------------

CASES = [
    "case1888rte",
    "case6470rte",
    "case6495rte",
    "case6515rte",
]

# Optional special kwargs if needed for specific networks
CASE_KWARGS = {
    # "case1888rte": {"ref_bus_idx": 1246},
    # "case6470rte": {"ref_bus_idx": 5988},
    # "case6515rte": {"ref_bus_idx": 6171},
}

# Optional trafo magnetizing parameters
TRAFO_PFE_KW = None
TRAFO_I0_PERCENT = None

TOPK = 10


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _build_pp_to_ppc_bus_lookup(net, N: int):
    """
    Build mapping:
        pandapower net bus index -> PPC internal bus index
    """
    pp_to_ppc = {}
    pp_bus_ids = net.bus.index.to_numpy(dtype=int)

    raw = None
    if hasattr(net, "_pd2ppc_lookups"):
        raw = net._pd2ppc_lookups.get("bus", None)

    if raw is not None:
        raw = np.asarray(raw).reshape(-1)

        # common case: raw indexed by actual pandapower bus index
        for pp_bus in pp_bus_ids:
            if 0 <= pp_bus < len(raw):
                ppc_bus = int(raw[pp_bus])
                if 0 <= ppc_bus < N:
                    pp_to_ppc[pp_bus] = ppc_bus

        # fallback: raw length equals len(net.bus)
        if len(pp_to_ppc) == 0 and len(raw) == len(pp_bus_ids):
            for pp_bus, ppc_bus in zip(pp_bus_ids, raw.astype(int)):
                if 0 <= int(ppc_bus) < N:
                    pp_to_ppc[int(pp_bus)] = int(ppc_bus)

    # final fallback
    if len(pp_to_ppc) == 0:
        for k, pp_bus in enumerate(pp_bus_ids[:N]):
            pp_to_ppc[int(pp_bus)] = int(k)

    return pp_to_ppc


def _apply_optional_trafo_params(net):
    if hasattr(net, "trafo") and net.trafo is not None and len(net.trafo):
        if TRAFO_PFE_KW is not None:
            net.trafo.loc[:, "pfe_kw"] = float(TRAFO_PFE_KW)
        if TRAFO_I0_PERCENT is not None:
            net.trafo.loc[:, "i0_percent"] = float(TRAFO_I0_PERCENT)


def _compile_internal_ppc(net):
    """
    Compile internal PPC / Ybus even if full PF does not converge.
    """
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
    except LoadflowNotConverged:
        pass

    if not hasattr(net, "_ppc") or "internal" not in net._ppc:
        raise RuntimeError("Could not build net._ppc['internal'].")

    return net._ppc["internal"]


def _build_custom_nr_inputs_from_net(net):
    """
    Build the custom-NR inputs in PPC order, in per-unit:
      - Ybus_pu
      - bus_typ
      - S_spec_pu
      - U_start_pu = flat start 1∠0
    """
    ppc_int = _compile_internal_ppc(net)

    bus_ppc = np.asarray(ppc_int["bus"], dtype=float)
    Ybus_pu = ppc_int["Ybus"].toarray().astype(np.complex128)
    baseMVA = float(ppc_int["baseMVA"])

    N = bus_ppc.shape[0]
    pp_to_ppc = _build_pp_to_ppc_bus_lookup(net, N)

    # bus types in PPC order
    # 1 = slack, 2 = PV, 3 = PQ
    bus_typ = np.full(N, 3, dtype=np.int64)

    if hasattr(net, "ext_grid") and len(net.ext_grid):
        for _, r in net.ext_grid.iterrows():
            pp_bus = int(r.bus)
            ppc_bus = pp_to_ppc.get(pp_bus, None)
            if ppc_bus is not None:
                bus_typ[ppc_bus] = 1

    if hasattr(net, "gen") and len(net.gen):
        for _, r in net.gen.iterrows():
            pp_bus = int(r.bus)
            ppc_bus = pp_to_ppc.get(pp_bus, None)
            if ppc_bus is not None and bus_typ[ppc_bus] != 1:
                bus_typ[ppc_bus] = 2

    # specified complex power in pu, PPC order
    S_spec_pu = np.zeros(N, dtype=np.complex128)

    if hasattr(net, "load") and len(net.load):
        for _, r in net.load.iterrows():
            pp_bus = int(r.bus)
            ppc_bus = pp_to_ppc.get(pp_bus, None)
            if ppc_bus is None:
                continue
            p_pu = float(r.p_mw) / baseMVA
            q_pu = float(r.q_mvar) / baseMVA
            S_spec_pu[ppc_bus] -= (p_pu + 1j * q_pu)

    if hasattr(net, "sgen") and len(net.sgen):
        for _, r in net.sgen.iterrows():
            pp_bus = int(r.bus)
            ppc_bus = pp_to_ppc.get(pp_bus, None)
            if ppc_bus is None:
                continue
            p_pu = float(r.p_mw) / baseMVA
            q_pu = float(r.get("q_mvar", 0.0)) / baseMVA
            S_spec_pu[ppc_bus] += (p_pu + 1j * q_pu)

    if hasattr(net, "gen") and len(net.gen):
        for _, r in net.gen.iterrows():
            pp_bus = int(r.bus)
            ppc_bus = pp_to_ppc.get(pp_bus, None)
            if ppc_bus is None:
                continue
            p_pu = float(r.p_mw) / baseMVA
            S_spec_pu[ppc_bus] += p_pu

    # mirror your "no_change" generator logic:
    # flat start at 1 pu, 0 angle on every bus
    U_start_pu = np.ones(N, dtype=np.complex128)

    return {
        "ppc_int": ppc_int,
        "Ybus_pu": Ybus_pu,
        "bus_typ": bus_typ,
        "S_spec_pu": S_spec_pu,
        "U_start_pu": U_start_pu,
        "pp_to_ppc": pp_to_ppc,
    }


def _try_pandapower_solve(net):
    """
    Try a few pandapower solve settings.
    Returns a dict with success/failure info.
    """
    attempts = [
        dict(algorithm="nr", init="flat"),
        dict(algorithm="nr", init="dc"),
        dict(algorithm="iwamoto_nr", init="flat"),
    ]

    last_err = None

    for a in attempts:
        try:
            pp.runpp(
                net,
                algorithm=a["algorithm"],
                init=a["init"],
                calculate_voltage_angles=True,
                enforce_q_lims=False,
                max_iteration=40,
                tolerance_mva=1e-8,
            )
            return {
                "success": True,
                "algorithm": a["algorithm"],
                "init": a["init"],
                "error": None,
            }
        except Exception as e:
            last_err = repr(e)

    return {
        "success": False,
        "algorithm": None,
        "init": None,
        "error": last_err,
    }


def _build_U_from_res_bus_in_ppc_order(net, pp_to_ppc, N):
    """
    Build converged complex voltage in PPC order, in pu.
    """
    U_pp = np.zeros(N, dtype=np.complex128)

    for pp_bus, row in net.res_bus.iterrows():
        ppc_bus = pp_to_ppc.get(int(pp_bus), None)
        if ppc_bus is None:
            continue
        vm = float(row.vm_pu)
        va = np.deg2rad(float(row.va_degree))
        U_pp[ppc_bus] = vm * np.exp(1j * va)

    return U_pp


def calc_mismatch(bus_typ, Ybus, S_spec, U):
    """
    Same mismatch masking logic as your custom solver family:
      - dP for non-slack
      - dQ for PQ only
    """
    I = Ybus @ U
    S_calc = U * np.conj(I)

    dP = (S_spec.real - S_calc.real).copy()
    dQ = (S_spec.imag - S_calc.imag).copy()

    slack = (bus_typ == 1)
    pv = (bus_typ == 2)

    dP[slack] = 0.0
    dQ[slack | pv] = 0.0

    misinf = float(max(np.max(np.abs(dP)), np.max(np.abs(dQ))))
    per_bus = np.maximum(np.abs(dP), np.abs(dQ))

    return {
        "dP": dP,
        "dQ": dQ,
        "misinf": misinf,
        "per_bus": per_bus,
        "S_calc": S_calc,
    }


def top_mismatch_buses(info, k=10):
    idx = np.argsort(-info["per_bus"])[:k]
    out = []
    for i in idx:
        out.append({
            "bus": int(i),
            "mis": float(info["per_bus"][i]),
            "dP": float(info["dP"][i]),
            "dQ": float(info["dQ"][i]),
        })
    return out


def analyze_case(case_name):
    fn = getattr(pn, case_name)
    case_kwargs = CASE_KWARGS.get(case_name, {}).copy()

    net = fn(**case_kwargs)
    _apply_optional_trafo_params(net)

    custom = _build_custom_nr_inputs_from_net(net)

    Ybus_pu = custom["Ybus_pu"]
    bus_typ = custom["bus_typ"]
    S_spec_pu = custom["S_spec_pu"]
    U_start_pu = custom["U_start_pu"]
    pp_to_ppc = custom["pp_to_ppc"]

    N = len(bus_typ)

    start_info = calc_mismatch(bus_typ, Ybus_pu, S_spec_pu, U_start_pu)

    # solve pandapower on the same net
    solve_info = _try_pandapower_solve(net)

    result = {
        "case": case_name,
        "N": N,
        "n_slack": int(np.sum(bus_typ == 1)),
        "n_pv": int(np.sum(bus_typ == 2)),
        "n_pq": int(np.sum(bus_typ == 3)),
        "start_misinf": start_info["misinf"],
        "start_top": top_mismatch_buses(start_info, k=TOPK),
        "pp_success": solve_info["success"],
        "pp_algorithm": solve_info["algorithm"],
        "pp_init": solve_info["init"],
        "pp_error": solve_info["error"],
        "solved_misinf": None,
        "solved_top": None,
    }

    if solve_info["success"]:
        # after successful runpp, refresh PPC order lookup just in case
        ppc_int_solved = net._ppc["internal"]
        N2 = ppc_int_solved["bus"].shape[0]
        pp_to_ppc_solved = _build_pp_to_ppc_bus_lookup(net, N2)

        U_pp_solved = _build_U_from_res_bus_in_ppc_order(net, pp_to_ppc_solved, N2)

        solved_info = calc_mismatch(bus_typ, Ybus_pu, S_spec_pu, U_pp_solved)
        result["solved_misinf"] = solved_info["misinf"]
        result["solved_top"] = top_mismatch_buses(solved_info, k=TOPK)

    return result


def print_case_result(r):
    print("=" * 110)
    print(f"[{r['case']}]")
    print(f"N={r['N']} | slack={r['n_slack']} pv={r['n_pv']} pq={r['n_pq']}")
    print(f"custom flat start mismatch_inf = {r['start_misinf']:.12g}")

    print("\nTop mismatch buses at custom flat start:")
    for x in r["start_top"]:
        print(
            f"  bus {x['bus']:5d} | mis={x['mis']:.12g} | "
            f"dP={x['dP']:.12g} | dQ={x['dQ']:.12g}"
        )

    print("\nPandapower solve:")
    if r["pp_success"]:
        print(f"  converged = True  | algorithm={r['pp_algorithm']} | init={r['pp_init']}")
        print(f"  mismatch_inf at pandapower solved U = {r['solved_misinf']:.12g}")

        print("\nTop mismatch buses at pandapower solved U:")
        for x in r["solved_top"]:
            print(
                f"  bus {x['bus']:5d} | mis={x['mis']:.12g} | "
                f"dP={x['dP']:.12g} | dQ={x['dQ']:.12g}"
            )

        print("\nInterpretation:")
        if r["solved_misinf"] is not None and r["solved_misinf"] < 1e-6:
            print("  pandapower solved state is also a near-solution of your custom equations.")
            print("  This points mainly to NR robustness / basin-of-attraction from flat start.")
        else:
            print("  pandapower converged, but its solved voltage is NOT a tiny-mismatch solution")
            print("  under your custom fixed (bus_typ, Ybus, S_spec) equations.")
            print("  That points to a model mismatch, not only step robustness.")
    else:
        print("  converged = False")
        print(f"  error     = {r['pp_error']}")
        print("\nInterpretation:")
        print("  pandapower did not produce a solved state under these attempts,")
        print("  so this case is hard even before comparing to your custom NR.")

    print()


def main():
    all_results = []

    for case_name in CASES:
        try:
            r = analyze_case(case_name)
            all_results.append(r)
            print_case_result(r)
        except Exception as e:
            print("=" * 110)
            print(f"[{case_name}]")
            print(f"FAILED during analysis: {repr(e)}")
            print()

    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)

    for r in all_results:
        pp_status = "yes" if r["pp_success"] else "no"
        solved_mis = "None" if r["solved_misinf"] is None else f"{r['solved_misinf']:.6g}"
        print(
            f"{r['case']:12s} | "
            f"N={r['N']:5d} | "
            f"start_misinf={r['start_misinf']:.6g} | "
            f"pp_converged={pp_status:3s} | "
            f"pp_algo={str(r['pp_algorithm']):11s} | "
            f"pp_init={str(r['pp_init']):4s} | "
            f"solved_misinf={solved_mis}"
        )


if __name__ == "__main__":
    main()