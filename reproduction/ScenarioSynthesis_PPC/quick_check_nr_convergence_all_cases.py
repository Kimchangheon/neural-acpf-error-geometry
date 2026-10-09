#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import multiprocessing as mp
from collections import defaultdict, Counter

import numpy as np
import pandapower.networks as pn

from case_generator_all_test_cases_pandapower_consider_ppc_branch_row import case_generation_pandapower
from newton_raphson_improved import newtonrapson


CASES = [
    "case4gs",
    "case5",
    "case6ww",
    "case9",
    "case14",
    "case24_ieee_rts",
    "case30",
    "case_ieee30",
    "case33bw",
    "case39",
    "case57",
    "case89pegase",
    "case118",
    "case145",
    "case_illinois200",
    "case300",
    "case1354pegase",
    "case1888rte",
    "case2848rte",
    "case2869pegase",
    "case3120sp",
    "case6470rte",
    "case6495rte",
    "case6515rte",
    "case9241pegase",
    "GBnetwork",
    "GBreducednetwork",
    "iceland",
]

# Use only for cases without net.trafo, if you want a nonzero compiled branch shunt
FORCE_SHUNT_CASES = {
    "case4gs", "case5", "case6ww", "case9", "case30", "case33bw"
}

# Optional special kwargs for some huge RTE cases if needed
CASE_KWARGS = {
    # "case1888rte": {"ref_bus_idx": 1246},
    # "case2848rte": {"ref_bus_idx": 271},
    # "case6470rte": {"ref_bus_idx": 5988},
    # "case6495rte": {"ref_bus_idx": None},
    # "case6515rte": {"ref_bus_idx": 6171},
}

SCENARIO_PRESETS = {
    "easy": dict(
        jitter_load=0.02,
        jitter_gen=0.01,
        pv_vset_range=(0.995, 1.005),
        rand_u_start=False,
        angle_jitter_deg=0.5,
        mag_jitter_pq=0.002,
    ),
    "no_change": dict(
        jitter_load=0.0,
        jitter_gen=0.0,
        pv_vset_range=(1.0, 1.0),
        rand_u_start=False,
        angle_jitter_deg=0.0,
        mag_jitter_pq=0.0,
    ),
    "A": dict(
        jitter_load=0.05,
        jitter_gen=0.03,
        pv_vset_range=(0.99, 1.02),
        rand_u_start=True,
        angle_jitter_deg=3.0,
        mag_jitter_pq=0.01,
    ),
    "B": dict(
        jitter_load=0.10,
        jitter_gen=0.05,
        pv_vset_range=(0.98, 1.03),
        rand_u_start=True,
        angle_jitter_deg=5.0,
        mag_jitter_pq=0.02,
    ),
    "C": dict(
        jitter_load=0.15,
        jitter_gen=0.08,
        pv_vset_range=(0.97, 1.04),
        rand_u_start=True,
        angle_jitter_deg=7.0,
        mag_jitter_pq=0.03,
    ),
}


# ============================================================
# per-unit conversion helpers for NR
# ============================================================

def ybus_si_to_pu(Y_si: np.ndarray, Vbase_bus: np.ndarray, S_base: float) -> np.ndarray:
    """
    Entrywise Ybus SI -> pu with local bus bases:
        Y_pu[i,j] = Y_SI[i,j] * V_i * V_j / S_base
    """
    scale = np.outer(Vbase_bus, Vbase_bus) / float(S_base)
    return np.asarray(Y_si, dtype=np.complex128) * scale.astype(np.float64, copy=False)


def u_si_to_pu_per_bus(u_si: np.ndarray, Vbase_bus: np.ndarray) -> np.ndarray:
    """
    Voltage SI -> pu per bus:
        U_pu[i] = U_SI[i] / V_i
    """
    return np.asarray(u_si, dtype=np.complex128) / np.asarray(Vbase_bus, dtype=np.float64)


def s_si_to_pu(S_si: np.ndarray, S_base: float) -> np.ndarray:
    """
    Power SI -> pu:
        S_pu[i] = S_SI[i] / S_base
    """
    return np.asarray(S_si, dtype=np.complex128) / float(S_base)


def convert_nr_inputs_to_pu(
    Y_matrix_si: np.ndarray,
    s_multi_si: np.ndarray,
    u_start_si: np.ndarray,
    vn_kv: np.ndarray,
    S_base: float,
):
    """
    Convert NR inputs from SI to per-unit using per-bus voltage bases.
    """
    Vbase_bus = np.asarray(vn_kv, dtype=np.float64) * 1e3

    Y_pu = ybus_si_to_pu(np.asarray(Y_matrix_si, dtype=np.complex128), Vbase_bus, S_base)
    s_pu = s_si_to_pu(np.asarray(s_multi_si, dtype=np.complex128), S_base)
    u_pu = u_si_to_pu_per_bus(np.asarray(u_start_si, dtype=np.complex128), Vbase_bus)

    return Y_pu, s_pu, u_pu, Vbase_bus


# ============================================================
# misc helpers
# ============================================================

def is_converged_solution(u_newton) -> bool:
    a = np.asarray(u_newton)
    if a.size == 0:
        return False
    if np.allclose(a, 0.0):
        return False
    return True


def _worker_init():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["BLIS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _compact_diag(diag: dict):
    """
    Keep only compact diagnostic info in results to avoid bloating memory.
    """
    if diag is None:
        return None

    out = {
        "converged": diag.get("converged"),
        "classification": diag.get("classification"),
        "failure_reason": diag.get("failure_reason"),
        "iterations": diag.get("iterations"),
        "final_misinf": diag.get("final_misinf"),
        "best_misinf": diag.get("best_misinf"),
    }

    if "misinf_history" in diag:
        hist = diag.get("misinf_history", [])
        out["misinf_history_tail"] = hist[-5:] if hist else []

    if "step_history" in diag:
        hist = diag.get("step_history", [])
        out["step_history_tail"] = hist[-5:] if hist else []

    return out


# ============================================================
# worker
# ============================================================

def run_one_case_one_seed(task):
    """
    task = (
        case_name,
        seed,
        scenario_cfg,
        K,
        use_force_shunt_when_no_trafo,
        trafo_pfe_kw,
        trafo_i0_percent,
        pu_nr,
        diagnose_nr,
        print_misinf,
        near_misinf_tol,
        start_mode,
    )
    """
    (
        case_name,
        seed,
        scenario_cfg,
        K,
        use_force_shunt_when_no_trafo,
        trafo_pfe_kw,
        trafo_i0_percent,
        pu_nr,
        diagnose_nr,
        print_misinf,
        near_misinf_tol,
        start_mode,
    ) = task

    try:
        fn = getattr(pn, case_name)
        case_kwargs = CASE_KWARGS.get(case_name, {}).copy()

        force = None
        if use_force_shunt_when_no_trafo and case_name in FORCE_SHUNT_CASES:
            force = {"g": 0.0, "b": 0.2, "g_asym": 0.0, "b_asym": 0.0}

        gen_kwargs = dict(
            case_fn=fn,
            case_kwargs=case_kwargs,
            ybus_mode="ppcY",
            seed=seed,
            trafo_pfe_kw=trafo_pfe_kw,
            trafo_i0_percent=trafo_i0_percent,
            force_branch_shunt_pu=force,
            start_mode=start_mode,
        )
        gen_kwargs.update(scenario_cfg)

        (
            gridtype,
            bus_typ,
            s_multi,
            u_start,
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
        ) = case_generation_pandapower(**gen_kwargs)

        N = len(bus_typ)

        if not is_connected:
            return {
                "case": case_name,
                "seed": seed,
                "N": N,
                "gridtype": gridtype,
                "converged": False,
                "reason": "generator_returned_not_connected",
                "nr_diag": None,
            }

        bus_typ_arr = np.asarray(bus_typ, dtype=np.int64).copy()

        if pu_nr:
            Y_for_nr, S_for_nr, U_for_nr, _Vbase_bus = convert_nr_inputs_to_pu(
                Y_matrix_si=np.asarray(Y_matrix, dtype=np.complex128),
                s_multi_si=np.asarray(s_multi, dtype=np.complex128),
                u_start_si=np.asarray(u_start, dtype=np.complex128),
                vn_kv=np.asarray(vn_kv, dtype=np.float64),
                S_base=float(S_base),
            )
        else:
            Y_for_nr = np.asarray(Y_matrix, dtype=np.complex128).copy()
            S_for_nr = np.asarray(s_multi, dtype=np.complex128).copy()
            U_for_nr = np.asarray(u_start, dtype=np.complex128).copy()

        # NOTE:
        # print_misinf=True with multiprocessing will interleave logs across workers.
        # Use workers=1 when you want readable per-iteration traces.
        u_newton, _I_unused, S_newton, nr_diag = newtonrapson(
            bus_typ_arr,
            Y_for_nr,
            S_for_nr,
            U_for_nr,
            K=K,
            diagnose=diagnose_nr,
            print_misinf=print_misinf,
            return_diagnostics=True,
            near_misinf_tol=near_misinf_tol,
        )

        conv = is_converged_solution(u_newton)

        if conv:
            reason = "ok"
        else:
            if nr_diag is not None and nr_diag.get("classification") is not None:
                reason = str(nr_diag["classification"])
            else:
                reason = "all_zero_solution_or_empty"

        return {
            "case": case_name,
            "seed": seed,
            "N": N,
            "gridtype": gridtype,
            "converged": conv,
            "reason": reason,
            "nr_diag": _compact_diag(nr_diag),
        }

    except Exception as e:
        return {
            "case": case_name,
            "seed": seed,
            "N": None,
            "gridtype": None,
            "converged": False,
            "reason": f"NR_exception: {repr(e)}",
            "nr_diag": None,
        }


# ============================================================
# task building
# ============================================================

def build_tasks(
    scenarios_per_case=10,
    scenario_level="A",
    K=40,
    use_force_shunt_when_no_trafo=True,
    trafo_pfe_kw=50.0,
    trafo_i0_percent=2.0,
    pu_nr=False,
    diagnose_nr=True,
    print_misinf=False,
    near_misinf_tol=1e-3,
    start_mode="auto",
):
    if scenario_level not in SCENARIO_PRESETS:
        raise ValueError(f"Unknown scenario_level={scenario_level}. Use one of {list(SCENARIO_PRESETS.keys())}")

    scenario_cfg = SCENARIO_PRESETS[scenario_level]
    tasks = []

    for case_name in CASES:
        for s in range(scenarios_per_case):
            seed = 1000 + s
            tasks.append((
                case_name,
                seed,
                scenario_cfg,
                K,
                use_force_shunt_when_no_trafo,
                trafo_pfe_kw,
                trafo_i0_percent,
                pu_nr,
                diagnose_nr,
                print_misinf,
                near_misinf_tol,
                start_mode,
            ))
    return tasks


# ============================================================
# summary
# ============================================================

def summarize_results(results, scenarios_per_case):
    by_case = defaultdict(list)
    for r in results:
        by_case[r["case"]].append(r)

    summary = []
    zero_conv_cases = []
    all_fail_details = {}
    all_fail_diag_samples = {}

    for case_name in CASES:
        rows = by_case.get(case_name, [])
        rows = sorted(rows, key=lambda x: x["seed"])

        n_conv = sum(1 for r in rows if r["converged"])
        n_fail = len(rows) - n_conv
        N = None
        gridtype = None

        fail_reasons = []
        fail_diag_samples = []

        for r in rows:
            if N is None and r["N"] is not None:
                N = r["N"]
            if gridtype is None and r["gridtype"] is not None:
                gridtype = r["gridtype"]
            if not r["converged"]:
                fail_reasons.append(r["reason"])
                if r.get("nr_diag") is not None:
                    fail_diag_samples.append(r["nr_diag"])

        at_least_once = (n_conv > 0)

        summary.append({
            "case": case_name,
            "N": N,
            "gridtype": gridtype,
            "converged": n_conv,
            "failed": n_fail,
            "at_least_once": at_least_once,
            "reason_counts": dict(Counter(fail_reasons)),
        })

        if not at_least_once:
            zero_conv_cases.append(case_name)
            all_fail_details[case_name] = fail_reasons
            all_fail_diag_samples[case_name] = fail_diag_samples

    return summary, zero_conv_cases, all_fail_details, all_fail_diag_samples


# ============================================================
# main runner
# ============================================================

def quick_screen_all_cases_mp(
    scenarios_per_case=10,
    scenario_level="A",
    K=40,
    workers=0,
    chunksize=1,
    use_force_shunt_when_no_trafo=True,
    trafo_pfe_kw=None,
    trafo_i0_percent=None,
    pu_nr=False,
    diagnose_nr=True,
    print_misinf=False,
    near_misinf_tol=1e-3,
    start_mode="auto",
):
    tasks = build_tasks(
        scenarios_per_case=scenarios_per_case,
        scenario_level=scenario_level,
        K=K,
        use_force_shunt_when_no_trafo=use_force_shunt_when_no_trafo,
        trafo_pfe_kw=trafo_pfe_kw,
        trafo_i0_percent=trafo_i0_percent,
        pu_nr=pu_nr,
        diagnose_nr=diagnose_nr,
        print_misinf=print_misinf,
        near_misinf_tol=near_misinf_tol,
        start_mode=start_mode,
    )

    if workers <= 0:
        workers = os.cpu_count() or 1

    print("=" * 100)
    print("Quick NR convergence screening (multiprocessing)")
    print(f"tasks              = {len(tasks)}")
    print(f"cases              = {len(CASES)}")
    print(f"scenarios_per_case = {scenarios_per_case}")
    print(f"scenario_level     = {scenario_level}")
    print(f"scenario_cfg       = {SCENARIO_PRESETS[scenario_level]}")
    print(f"K                  = {K}")
    print(f"workers            = {workers}")
    print(f"chunksize          = {chunksize}")
    print(f"pu_nr              = {pu_nr}")
    print(f"diagnose_nr        = {diagnose_nr}")
    print(f"print_misinf       = {print_misinf}")
    print(f"near_misinf_tol    = {near_misinf_tol}")
    print(f"start_mode         = {start_mode}")
    print("=" * 100)

    if print_misinf and workers != 1:
        print("WARNING: print_misinf=True with workers>1 will produce interleaved logs.\n")

    if os.name == "nt":
        ctx = mp.get_context("spawn")
    else:
        ctx = mp.get_context("fork")

    results = []
    with ctx.Pool(processes=workers, initializer=_worker_init, maxtasksperchild=200) as pool:
        for idx, res in enumerate(pool.imap_unordered(run_one_case_one_seed, tasks, chunksize=chunksize), 1):
            results.append(res)

            status = "converged" if res["converged"] else f"failed ({res['reason']})"

            extra = ""
            nr_diag = res.get("nr_diag")
            if nr_diag is not None and not res["converged"]:
                final_misinf = nr_diag.get("final_misinf")
                best_misinf = nr_diag.get("best_misinf")
                iterations = nr_diag.get("iterations")
                extra = (
                    f" | it={iterations}"
                    f" | best_misinf={best_misinf if best_misinf is not None else 'None'}"
                    f" | final_misinf={final_misinf if final_misinf is not None else 'None'}"
                )

            print(f"[{idx:4d}/{len(tasks)}] {res['case']} seed={res['seed']} -> {status}{extra}")

    summary, zero_conv_cases, all_fail_details, all_fail_diag_samples = summarize_results(results, scenarios_per_case)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    summary_sorted = sorted(summary, key=lambda x: (x["converged"], x["N"] if x["N"] is not None else 10**9))
    for row in summary_sorted:
        print(
            f"{row['case']:16s} | "
            f"N={str(row['N']):>5s} | "
            f"conv={row['converged']:2d}/{scenarios_per_case} | "
            f"at_least_once={row['at_least_once']} | "
            f"fail_reasons={row['reason_counts']}"
        )

    print("\nCases with ZERO convergence in the tested scenarios:")
    if zero_conv_cases:
        for c in zero_conv_cases:
            print(f"  - {c}")
    else:
        print("  None")

    print("\nFailure reason samples for zero-convergence cases:")
    if zero_conv_cases:
        for c in zero_conv_cases:
            reasons = all_fail_details[c]
            uniq = []
            for r in reasons:
                if r not in uniq:
                    uniq.append(r)
            print(f"\n[{c}]")
            for r in uniq[:10]:
                print(f"  {r}")
    else:
        print("  None")

    print("\nDiagnostic samples for zero-convergence cases:")
    if zero_conv_cases:
        for c in zero_conv_cases:
            samples = all_fail_diag_samples[c]
            print(f"\n[{c}]")
            if not samples:
                print("  no nr_diag stored")
                continue
            for i, d in enumerate(samples[:3], 1):
                print(
                    f"  sample {i}: "
                    f"class={d.get('classification')} | "
                    f"it={d.get('iterations')} | "
                    f"best_misinf={d.get('best_misinf')} | "
                    f"final_misinf={d.get('final_misinf')} | "
                    f"misinf_tail={d.get('misinf_history_tail')}"
                )
    else:
        print("  None")

    return results, summary, zero_conv_cases, all_fail_details, all_fail_diag_samples


if __name__ == "__main__":
    quick_screen_all_cases_mp(
        scenarios_per_case=1,
        scenario_level="A", # no_change, easy, A, B, C
        K=40,
        workers=0,   # 0 => all CPU cores
        chunksize=1,
        use_force_shunt_when_no_trafo=True,
        pu_nr=False,          # per-unit NR
        diagnose_nr=True,    # store and classify non-convergence
        print_misinf=False,  # set True only when workers=1 for readable logs
        near_misinf_tol=1e-3,
        start_mode="dc_compile",   # "auto", "manual_flat", "ppc_v0", "dc_compile"
        # trafo_pfe_kw=50.0,
        # trafo_i0_percent=2.0,
    )