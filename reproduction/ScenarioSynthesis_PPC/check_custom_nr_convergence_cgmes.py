#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check custom Newton-Raphson convergence diagnostics for one or more CGMES files.

This script is intended for CGMES cases where pandapower runpp converges but the
in-repo custom NR solver may fail. It reports:
  - branch asymmetry counts
  - branch-row Ybus reconstruction error
  - pandapower solution residual under the exported Ybus/Sbus equations
  - custom NR convergence diagnostics from the chosen start mode
  - start-to-pandapower-solution distance

Example:
  python check_custom_nr_convergence_cgmes.py \
    --cgmes_path ./CGMES/ENTSO-E-RealGridTest.zip \
    --scenario_level A \
    --start_mode dc_compile manual_flat \
    --K 5 \
    --output_csv ./out/cgmes_custom_nr_check.csv
"""

import argparse
import csv
import glob
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandapower as pp

from case_generator_all_test_cases_pandapower_consider_ppc_branch_row import (
    _build_net_from_source,
    _optional_branch_indices,
    case_generation_pandapower,
    reconstruct_Y_pandapower_branchrows_direct_SI,
)
from main_datagen_multiproc_improved import (
    SCENARIO_PRESETS,
    convert_nr_inputs_to_pu,
    s_pu_to_si,
    u_pu_to_si_per_bus,
)
from newton_raphson_improved import _compute_mismatch_inf, newtonrapson


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose custom NR convergence for CGMES zip files."
    )
    parser.add_argument(
        "--cgmes_path",
        nargs="+",
        required=True,
        help="One or more CGMES zip files, directories, or glob patterns.",
    )
    parser.add_argument("--recursive", action="store_true", help="Search directories recursively for *.zip.")
    parser.add_argument("--cgmes_version", default="2.4.15")
    parser.add_argument("--no_cgmes_ignore_errors", dest="cgmes_ignore_errors", action="store_false")
    parser.set_defaults(cgmes_ignore_errors=True)
    parser.add_argument("--scenario_level", default="no_change", choices=list(SCENARIO_PRESETS.keys()))
    parser.add_argument(
        "--start_mode",
        nargs="+",
        default=["dc_compile"],
        choices=["auto", "manual_flat", "ppc_v0", "dc_compile"],
    )
    parser.add_argument("--ybus_mode", default="ppcY", choices=["ppcY", "stamped"])
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--pu_nr", action="store_true", help="Run custom NR in per-unit coordinates.")
    parser.add_argument("--diagnose_nr", action="store_true", default=True)
    parser.add_argument("--print_misinf", action="store_true")
    parser.add_argument("--mismatch_tol", type=float, default=1e-8)
    parser.add_argument("--step_tol", type=float, default=5e-4)
    parser.add_argument("--near_misinf_tol", type=float, default=1e-3)
    parser.add_argument("--convergence_mode", default="misinf", choices=["misinf", "two_step"])
    parser.add_argument("--output_csv", default="")
    return parser.parse_args()


def expand_sources(patterns: Iterable[str], recursive: bool) -> List[Path]:
    out: List[Path] = []
    for item in patterns:
        expanded = Path(os.path.expanduser(item))
        matches = sorted(glob.glob(str(expanded))) if any(ch in str(expanded) for ch in "*?[]") else []
        if matches:
            out.extend(Path(p) for p in matches if Path(p).is_file() and Path(p).suffix.lower() == ".zip")
            continue

        if expanded.is_file():
            out.append(expanded)
        elif expanded.is_dir():
            globber = expanded.rglob if recursive else expanded.glob
            out.extend(sorted(p for p in globber("*.zip") if p.is_file()))
        else:
            raise FileNotFoundError(f"CGMES path does not exist: {expanded}")

    unique = []
    seen = set()
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            unique.append(rp)
            seen.add(rp)
    return unique


def make_case_cfg(path: Path, case_name: str, cgmes_version: str, ignore_errors: bool) -> Dict:
    return {
        "cgmes_files": str(path),
        "case_name": case_name,
        "converter_kwargs": {
            "cgmes_version": str(cgmes_version).strip(),
            "ignore_errors": bool(ignore_errors),
        },
    }


def branch_asymmetry_summary(path: Path, cgmes_version: str, ignore_errors: bool) -> Dict[str, float]:
    cfg = make_case_cfg(path, path.stem, cgmes_version, ignore_errors)
    net, _ = _build_net_from_source(cfg, {})
    try:
        pp.runpp(
            net,
            init="flat",
            calculate_voltage_angles=True,
            max_iteration=1,
            enforce_q_lims=False,
            tolerance_mva=1e9,
        )
    except Exception:
        pass

    if not hasattr(net, "_ppc") or "internal" not in net._ppc:
        return {
            "asym_r_nz": -1,
            "asym_x_nz": -1,
            "asym_b_nz": -1,
            "asym_g_nz": -1,
            "asym_max_abs": float("nan"),
        }

    branch = np.asarray(net._ppc["internal"]["branch"], dtype=float)
    _br_g, br_r_asym, br_x_asym, br_b_asym, br_g_asym = _optional_branch_indices()

    summary = {}
    max_abs = 0.0
    for name, idx in [
        ("asym_r", br_r_asym),
        ("asym_x", br_x_asym),
        ("asym_b", br_b_asym),
        ("asym_g", br_g_asym),
    ]:
        if idx is None or idx >= branch.shape[1]:
            vals = np.zeros(branch.shape[0], dtype=float)
        else:
            vals = branch[:, idx]
        summary[f"{name}_nz"] = int(np.count_nonzero(np.abs(vals) > 1e-14))
        max_abs = max(max_abs, float(np.max(np.abs(vals))) if vals.size else 0.0)
    summary["asym_max_abs"] = max_abs
    return summary


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values * values)))


def classify_reason(row: Dict) -> str:
    if not row.get("pp_converged", False):
        return "pandapower_label_not_converged"

    if row["pp_label_misinf"] > max(1e-2, 10.0 * row["effective_mismatch_tol"]):
        return "exported_ybus_or_sbus_mismatch"

    if row["y_rec_max_abs_diff"] > 1e-8 and row["asym_total_nz"] > 0:
        return "branch_asymmetry_or_reconstruction_mismatch"

    if row["custom_converged"]:
        return "custom_nr_converged"

    if row["custom_failure_reason"] == "jacobian_singular":
        return "custom_nr_jacobian_singular_after_bad_steps"

    if row["start_angle_rmse_deg"] > 5.0 or row["start_mag_rmse_pu"] > 0.05:
        return "custom_nr_start_outside_basin_undamped_newton"

    return "custom_nr_failed_despite_consistent_equations"


def run_one(path: Path, args, start_mode: str, sample_idx: int, asym_summary: Dict) -> Dict:
    t0 = time.time()
    case_name = f"{path.stem}_sample{sample_idx}_{start_mode}"
    scenario_cfg = SCENARIO_PRESETS[args.scenario_level]
    seed = int(args.seed) + int(sample_idx)
    cfg = make_case_cfg(path, case_name, args.cgmes_version, args.cgmes_ignore_errors)

    out = case_generation_pandapower(
        case_fn=cfg,
        case_kwargs={},
        ybus_mode=args.ybus_mode,
        seed=seed,
        cgmes_model_a_cleanup=False,
        jitter_load=float(scenario_cfg["jitter_load"]),
        jitter_load_q=float(scenario_cfg["jitter_load_q"]),
        jitter_gen=float(scenario_cfg["jitter_gen"]),
        pv_vset_range=scenario_cfg["pv_vset_range"],
        rand_u_start=bool(scenario_cfg["rand_u_start"]),
        angle_jitter_deg=float(scenario_cfg["angle_jitter_deg"]),
        mag_jitter_pq=float(scenario_cfg["mag_jitter_pq"]),
        start_mode=start_mode,
        load_scale_range=scenario_cfg.get("load_scale_range"),
        scale_gen_with_load=bool(scenario_cfg.get("scale_gen_with_load", True)),
        line_outage_prob=float(scenario_cfg.get("line_outage_prob", 0.0)),
        return_pp_solution=True,
    )

    base_out = out[:25]
    u_pp_si, S_pp_si, pp_converged = out[25:]
    (
        gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,
        branch_f, branch_t, branch_status, branch_tau, branch_shift,
        branch_y_from, branch_y_to, branch_y_ft,
        branch_ysh_from, branch_ysh_to, y_shunt_bus,
        is_trafo, _branch_hv_is_f, _branch_n, _y_lines, _y_c_lines,
        U_base, S_base, vn_kv,
    ) = base_out

    bus_typ = np.asarray(bus_typ, dtype=np.int64)
    Y_matrix = np.asarray(Y_matrix, dtype=np.complex128)
    s_multi = np.asarray(s_multi, dtype=np.complex128)
    u_start = np.asarray(u_start, dtype=np.complex128)
    u_pp_si = np.asarray(u_pp_si, dtype=np.complex128)
    Vbase_bus = np.asarray(vn_kv, dtype=np.float64) * 1e3

    y_rec = reconstruct_Y_pandapower_branchrows_direct_SI(
        len(bus_typ),
        branch_f,
        branch_t,
        branch_status,
        branch_tau,
        branch_shift,
        branch_y_from,
        branch_y_to,
        branch_y_ft,
        branch_ysh_from,
        branch_ysh_to,
        y_shunt_bus,
        Vbase_bus=Vbase_bus,
    )
    y_diff = np.abs(y_rec - Y_matrix)

    start_misinf_si = _compute_mismatch_inf(bus_typ, Y_matrix, u_start, s_multi.real, s_multi.imag)
    pp_label_misinf = _compute_mismatch_inf(bus_typ, Y_matrix, u_pp_si, s_multi.real, s_multi.imag)

    if args.pu_nr:
        Y_for_nr, S_for_nr, U_for_nr, vbase_for_nr = convert_nr_inputs_to_pu(
            Y_matrix_si=Y_matrix,
            s_multi_si=s_multi,
            u_start_si=u_start,
            vn_kv=np.asarray(vn_kv, dtype=np.float64),
            S_base=float(S_base),
        )
    else:
        Y_for_nr = Y_matrix
        S_for_nr = s_multi
        U_for_nr = u_start
        vbase_for_nr = None

    initial_misinf = _compute_mismatch_inf(
        bus_typ, Y_for_nr, U_for_nr, S_for_nr.real, S_for_nr.imag
    )

    nr_out = newtonrapson(
        bus_typ,
        Y_for_nr,
        S_for_nr,
        U_for_nr,
        K=int(args.K),
        diagnose=True,
        print_misinf=bool(args.print_misinf),
        return_diagnostics=True,
        near_misinf_tol=float(args.near_misinf_tol),
        convergence_mode=str(args.convergence_mode),
        step_tol=float(args.step_tol),
        mismatch_tol=float(args.mismatch_tol),
    )
    u_custom_raw, _i_unused, S_custom_raw, nr_diag = nr_out

    if args.pu_nr and np.asarray(u_custom_raw).size:
        u_custom_si = u_pu_to_si_per_bus(np.asarray(u_custom_raw), vbase_for_nr)
        S_custom_si = s_pu_to_si(np.asarray(S_custom_raw), float(S_base))
    else:
        u_custom_si = np.asarray(u_custom_raw, dtype=np.complex128)
        S_custom_si = np.asarray(S_custom_raw, dtype=np.complex128)

    custom_label_misinf = _compute_mismatch_inf(
        bus_typ, Y_matrix, u_custom_si, s_multi.real, s_multi.imag
    ) if u_custom_si.size else float("nan")

    non_slack = bus_typ != 1
    start_mag_pu = np.abs(u_start) / Vbase_bus
    pp_mag_pu = np.abs(u_pp_si) / Vbase_bus
    angle_delta_deg = np.rad2deg(np.angle(u_start / np.where(u_pp_si == 0, 1.0 + 0j, u_pp_si)))

    row = {
        "path": str(path),
        "case_name": case_name,
        "scenario_level": args.scenario_level,
        "start_mode": start_mode,
        "seed": seed,
        "pu_nr": bool(args.pu_nr),
        "N": int(len(bus_typ)),
        "N_branch": int(len(branch_f)),
        "N_slack": int(np.count_nonzero(bus_typ == 1)),
        "N_pv": int(np.count_nonzero(bus_typ == 2)),
        "N_pq": int(np.count_nonzero(bus_typ == 3)),
        "S_base": float(S_base),
        "U_base": float(U_base),
        "asym_r_nz": int(asym_summary["asym_r_nz"]),
        "asym_x_nz": int(asym_summary["asym_x_nz"]),
        "asym_b_nz": int(asym_summary["asym_b_nz"]),
        "asym_g_nz": int(asym_summary["asym_g_nz"]),
        "asym_total_nz": int(
            asym_summary["asym_r_nz"]
            + asym_summary["asym_x_nz"]
            + asym_summary["asym_b_nz"]
            + asym_summary["asym_g_nz"]
        ),
        "asym_max_abs": float(asym_summary["asym_max_abs"]),
        "y_rec_max_abs_diff": float(np.max(y_diff)) if y_diff.size else 0.0,
        "y_rec_fro_diff": float(np.linalg.norm(y_diff)) if y_diff.size else 0.0,
        "is_connected": bool(is_connected),
        "pp_converged": bool(pp_converged),
        "pp_label_misinf": float(pp_label_misinf),
        "start_misinf_si": float(start_misinf_si),
        "initial_misinf_solver_units": float(initial_misinf),
        "custom_converged": bool(nr_diag.get("converged", False)),
        "custom_failure_reason": str(nr_diag.get("failure_reason")),
        "custom_classification": str(nr_diag.get("classification")),
        "custom_iterations": int(nr_diag.get("iterations", 0)),
        "custom_final_misinf_solver_units": (
            float(nr_diag["final_misinf"]) if nr_diag.get("final_misinf") is not None else float("nan")
        ),
        "custom_best_misinf_solver_units": (
            float(nr_diag["best_misinf"]) if nr_diag.get("best_misinf") is not None else float("nan")
        ),
        "custom_label_misinf_si": float(custom_label_misinf),
        "effective_mismatch_tol": float(nr_diag.get("effective_mismatch_tol") or float("nan")),
        "start_mag_rmse_pu": rmse(start_mag_pu - pp_mag_pu),
        "start_angle_rmse_deg": rmse(angle_delta_deg[non_slack]),
        "elapsed_s": float(time.time() - t0),
    }
    row["likely_reason"] = classify_reason(row)
    return row


def print_row(row: Dict) -> None:
    print("\n=== CGMES custom NR check ===")
    print(f"path              : {row['path']}")
    print(f"scenario/start    : {row['scenario_level']} / {row['start_mode']} / pu_nr={row['pu_nr']}")
    print(f"N, branches       : {row['N']} buses, {row['N_branch']} branches")
    print(f"bus types         : slack={row['N_slack']}, PV={row['N_pv']}, PQ={row['N_pq']}")
    print(
        "asym branches     : "
        f"R={row['asym_r_nz']}, X={row['asym_x_nz']}, "
        f"B={row['asym_b_nz']}, G={row['asym_g_nz']}"
    )
    print(f"Yrec max diff     : {row['y_rec_max_abs_diff']:.6e}")
    print(f"pandapower label  : converged={row['pp_converged']}, misinf={row['pp_label_misinf']:.6e}")
    print(
        "start vs PP label : "
        f"|V| RMSE={row['start_mag_rmse_pu']:.6e} pu, "
        f"angle RMSE={row['start_angle_rmse_deg']:.6f} deg"
    )
    print(
        "custom NR         : "
        f"converged={row['custom_converged']}, "
        f"iters={row['custom_iterations']}, "
        f"final={row['custom_final_misinf_solver_units']:.6e}, "
        f"best={row['custom_best_misinf_solver_units']:.6e}, "
        f"class={row['custom_classification']}"
    )
    print(f"likely reason     : {row['likely_reason']}")
    print(f"elapsed           : {row['elapsed_s']:.1f}s")


def main():
    args = parse_args()
    sources = expand_sources(args.cgmes_path, recursive=bool(args.recursive))
    if not sources:
        raise FileNotFoundError("No CGMES zip files found.")

    rows = []
    asym_cache: Dict[Path, Dict] = {}

    for path in sources:
        print(f"[INFO] Preparing branch asymmetry summary for {path}")
        asym_cache[path] = branch_asymmetry_summary(
            path, args.cgmes_version, bool(args.cgmes_ignore_errors)
        )
        for sample_idx in range(int(args.samples)):
            for start_mode in args.start_mode:
                row = run_one(path, args, start_mode, sample_idx, asym_cache[path])
                rows.append(row)
                print_row(row)

    if args.output_csv:
        out_path = Path(os.path.expanduser(args.output_csv))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[INFO] Wrote CSV: {out_path}")


if __name__ == "__main__":
    main()
