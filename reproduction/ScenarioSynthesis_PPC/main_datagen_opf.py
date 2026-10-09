#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AC-OPF dataset generator for the PPC benchmark grids.

Mirrors main_datagen_multiproc_improved.py but replaces the Newton-Raphson
power-flow label with an optimal-power-flow solution from pandapower.

The structural columns (Y-bus, branch rows, bus types) are produced by the
same case_generation_pandapower() call the power-flow pipeline uses, so an
OPF row and a PF row describe the grid identically. Only the solution block
differs:

    PF  : u_start -> u_newton      (dispatch is an INPUT)
    OPF : u_start -> u_opf         (dispatch is an OUTPUT, plus LMPs)

Because generator dispatch is decided by the optimiser, generator jitter and
PV setpoint jitter are meaningless here and are forced off. The perturbation
is load-side only.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from case_generator_all_test_cases_pandapower_consider_ppc_branch_row import (
    case_generation_pandapower,
)

# --------------------------------------------------------------------------
# OPF scenario presets — load-side perturbation only.
# --------------------------------------------------------------------------
# Reference (slack) bus code in THIS pipeline's convention, which remaps the
# raw PPC codes to 1=slack, 2=PV, 3=PQ (see bus_typ construction in the case
# generator). Note this inverts PPC's own encoding, where 3 means REF.
REF_BUS_TYPE = 1

# Marks the training-loader-aligned schema in output filenames.
SCHEMA_TAG = "OPFv2aligned"

OPF_PRESETS = {
    # Phase-0 OPF backbone: broad demand coverage, fixed topology.
    "opf_backbone": dict(
        load_scale_range=(0.60, 1.40),
        jitter_load=0.10,
        jitter_load_q=0.15,
    ),
    # Narrower band for grids whose OPF feasible region is tight.
    "opf_narrow": dict(
        load_scale_range=(0.80, 1.20),
        jitter_load=0.05,
        jitter_load_q=0.08,
    ),
}

_CFG = {}


def ndarray_to_npy_bytes(arr: np.ndarray) -> bytes:
    import io
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _one_row(seed: int):
    cfg = _CFG
    out = case_generation_pandapower(
        cfg["preset"],
        ybus_mode="ppcY",
        seed=seed,
        # Load-side perturbation only: the optimiser sets generation.
        jitter_load=cfg["jitter_load"],
        jitter_load_q=cfg["jitter_load_q"],
        jitter_gen=0.0,
        pv_vset_range=None,
        scale_gen_with_load=False,
        load_scale_range=cfg["load_scale_range"],
        rand_u_start=False,
        start_mode=cfg["start_mode"],
        return_opf_solution=True,
        opf_vm_limits=cfg["vm_limits"],
        opf_line_max_loading=cfg["line_max_loading"],
        opf_dc=cfg["opf_dc"],
    )

    base = out[:25]
    (
        u_opf, S_opf, lam_p, lam_q, gen_p, gen_q, cost, conv,
        p_disp, q_disp, p_min, p_max, q_min, q_max,
        cost_c2, cost_c1, cost_c0, is_ctrl,
    ) = out[25:]

    (
        gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,
        Branch_f_bus, Branch_t_bus, Branch_status,
        Branch_tau, Branch_shift_deg,
        Branch_y_series_from, Branch_y_series_to, Branch_y_series_ft,
        Branch_y_shunt_from, Branch_y_shunt_to,
        Y_shunt_bus, Is_trafo, Branch_hv_is_f, Branch_n,
        Y_Lines, Y_C_Lines, U_base, S_base, vn_kv,
    ) = base

    if cfg["drop_nonconverged"] and not conv:
        return None

    u_start = np.asarray(u_start, dtype=np.complex128)
    u_opf = np.asarray(u_opf, dtype=np.complex128)

    # --- Angle-datum (gauge) normalisation -------------------------------
    # Rotate every voltage so the reference bus sits at 0 rad. This is a pure
    # gauge choice: S = V * conj(Y V) is exactly invariant under a global phase
    # rotation, so S_demand / S_opf are untouched and the physics is identical.
    # It matters because some benchmarks fix a non-zero datum -- pandapower's
    # case118 puts its slack at +30 deg, leaving no bus within 15 deg of zero --
    # and models that carry no bus-voltage input cannot represent that offset,
    # so it shows up as a large apparent angle error that is not a solve error.
    if cfg["normalize_angle_datum"]:
        bus_typ_arr = np.asarray(bus_typ, dtype=np.int64)
        ref = np.flatnonzero(bus_typ_arr == REF_BUS_TYPE)
        if ref.size and abs(u_opf[ref[0]]) > 0.0:
            phase = np.exp(-1j * np.angle(u_opf[ref[0]]))
            u_opf = u_opf * phase
            u_start = u_start * phase

    b = ndarray_to_npy_bytes
    return {
        "bus_number": np.int32(len(bus_typ)),
        "branch_number": np.int32(len(Branch_f_bus)),
        "gridtype": str(gridtype),
        "U_base": float(U_base),
        "S_base": float(S_base),
        "bus_typ": b(np.asarray(bus_typ, dtype=np.int64)),
        "vn_kv": b(np.asarray(vn_kv, dtype=np.float64)),
        "Y_shunt_bus": b(np.asarray(Y_shunt_bus, dtype=np.complex128)),
        "Branch_f_bus": b(np.asarray(Branch_f_bus, dtype=np.int64)),
        "Branch_t_bus": b(np.asarray(Branch_t_bus, dtype=np.int64)),
        "Branch_status": b(np.asarray(Branch_status, dtype=np.int64)),
        "Branch_tau": b(np.asarray(Branch_tau, dtype=np.float64)),
        "Branch_shift_deg": b(np.asarray(Branch_shift_deg, dtype=np.float64)),
        "Branch_y_series_from": b(np.asarray(Branch_y_series_from, dtype=np.complex128)),
        "Branch_y_series_to": b(np.asarray(Branch_y_series_to, dtype=np.complex128)),
        "Branch_y_series_ft": b(np.asarray(Branch_y_series_ft, dtype=np.complex128)),
        "Branch_y_shunt_from": b(np.asarray(Branch_y_shunt_from, dtype=np.complex128)),
        "Branch_y_shunt_to": b(np.asarray(Branch_y_shunt_to, dtype=np.complex128)),
        "Is_trafo": b(np.asarray(Is_trafo, dtype=np.int64)),
        "Branch_hv_is_f": b(np.asarray(Branch_hv_is_f, dtype=np.int64)),
        "Branch_n": b(np.asarray(Branch_n, dtype=np.float64)),
        "Y_Lines": b(np.asarray(Y_Lines, dtype=np.complex128)),
        "Y_C_Lines": b(np.asarray(Y_C_Lines, dtype=np.complex128)),
        # --- solution block ---------------------------------------------
        # Canonical names match the training loader's required columns
        # (u_start / u_newton / S_start / S_newton) so these files load with
        # no conversion step. The OPF reading of each is:
        #   S_start  = signed injection BEFORE optimisation: perturbed load
        #              everywhere, plus each unit's pre-existing scheduled
        #              dispatch at controllable buses. It is a warm-start
        #              operating point, NOT the optimum -- verified: at
        #              non-controllable buses S_start == S_newton to ~1e-6 MW,
        #              while at controllable buses it differs from p_disp_bus.
        #   u_newton = OPF voltage solution      S_newton = V * conj(Y V)
        # OPF-specific columns keep their own names and ride along via the
        # loader's extra_binary_columns passthrough.
        "u_start": b(u_start),
        "S_start": b(np.asarray(s_multi, dtype=np.complex128)),
        "u_newton": b(u_opf),
        "S_newton": b(np.asarray(S_opf, dtype=np.complex128)),
        # Duplicated under OPF names so the file is self-describing.
        "u_opf": b(u_opf),
        "S_opf": b(np.asarray(S_opf, dtype=np.complex128)),
        "S_demand": b(np.asarray(s_multi, dtype=np.complex128)),
        # --- decision space (bus-aligned, includes the slack) -------------
        "p_disp_bus": b(np.asarray(p_disp, dtype=np.float64)),
        "q_disp_bus": b(np.asarray(q_disp, dtype=np.float64)),
        "p_min_bus": b(np.asarray(p_min, dtype=np.float64)),
        "p_max_bus": b(np.asarray(p_max, dtype=np.float64)),
        "q_min_bus": b(np.asarray(q_min, dtype=np.float64)),
        "q_max_bus": b(np.asarray(q_max, dtype=np.float64)),
        "cost_c2": b(np.asarray(cost_c2, dtype=np.float64)),
        "cost_c1": b(np.asarray(cost_c1, dtype=np.float64)),
        "cost_c0": b(np.asarray(cost_c0, dtype=np.float64)),
        "is_controllable": b(np.asarray(is_ctrl, dtype=np.int8)),
        "lam_p": b(np.asarray(lam_p, dtype=np.float64)),
        "lam_q": b(np.asarray(lam_q, dtype=np.float64)),
        "gen_p_opt": b(np.asarray(gen_p, dtype=np.float64)),
        "gen_q_opt": b(np.asarray(gen_q, dtype=np.float64)),
        "opf_cost": float(cost),
        "opf_converged": np.int8(1 if conv else 0),
    }


def _schema() -> pa.Schema:
    f = [
        pa.field("bus_number", pa.int32()),
        pa.field("branch_number", pa.int32()),
        pa.field("gridtype", pa.string()),
        pa.field("U_base", pa.float64()),
        pa.field("S_base", pa.float64()),
        pa.field("bus_typ", pa.binary()),
        pa.field("vn_kv", pa.binary()),
        pa.field("Y_shunt_bus", pa.binary()),
        pa.field("Branch_f_bus", pa.binary()),
        pa.field("Branch_t_bus", pa.binary()),
        pa.field("Branch_status", pa.binary()),
        pa.field("Branch_tau", pa.binary()),
        pa.field("Branch_shift_deg", pa.binary()),
        pa.field("Branch_y_series_from", pa.binary()),
        pa.field("Branch_y_series_to", pa.binary()),
        pa.field("Branch_y_series_ft", pa.binary()),
        pa.field("Branch_y_shunt_from", pa.binary()),
        pa.field("Branch_y_shunt_to", pa.binary()),
        pa.field("Is_trafo", pa.binary()),
        pa.field("Branch_hv_is_f", pa.binary()),
        pa.field("Branch_n", pa.binary()),
        pa.field("Y_Lines", pa.binary()),
        pa.field("Y_C_Lines", pa.binary()),
        pa.field("u_start", pa.binary()),
        pa.field("S_start", pa.binary()),
        pa.field("u_newton", pa.binary()),
        pa.field("S_newton", pa.binary()),
        pa.field("u_opf", pa.binary()),
        pa.field("S_opf", pa.binary()),
        pa.field("S_demand", pa.binary()),
        pa.field("p_disp_bus", pa.binary()),
        pa.field("q_disp_bus", pa.binary()),
        pa.field("p_min_bus", pa.binary()),
        pa.field("p_max_bus", pa.binary()),
        pa.field("q_min_bus", pa.binary()),
        pa.field("q_max_bus", pa.binary()),
        pa.field("cost_c2", pa.binary()),
        pa.field("cost_c1", pa.binary()),
        pa.field("cost_c0", pa.binary()),
        pa.field("is_controllable", pa.binary()),
        pa.field("lam_p", pa.binary()),
        pa.field("lam_q", pa.binary()),
        pa.field("gen_p_opt", pa.binary()),
        pa.field("gen_q_opt", pa.binary()),
        pa.field("opf_cost", pa.float64()),
        pa.field("opf_converged", pa.int8()),
    ]
    return pa.schema(f)


def _init_worker(cfg):
    global _CFG
    _CFG = cfg
    import warnings
    warnings.filterwarnings("ignore")


def build_output_filename(args) -> str:
    lo, hi = OPF_PRESETS[args.scenario_level]["load_scale_range"]
    if args.load_scale_lo is not None:
        lo = args.load_scale_lo
    if args.load_scale_hi is not None:
        hi = args.load_scale_hi
    kind = "DCOPF" if args.opf_dc else "ACOPF"
    vm = f"vm{args.vm_min:.2f}-{args.vm_max:.2f}"
    # SCHEMA_TAG distinguishes the loader-aligned schema (canonical
    # u_start/u_newton/S_start/S_newton names, bus-aligned decision space
    # including the slack, gauge-normalised angles, small row groups) from the
    # original OPF-only files, whose derived artefacts must not be mixed in.
    name = (
        f"{args.preset}_ppcY_{args.scenario_level}_{args.start_mode}_"
        f"{kind}_{vm}_ls{lo:.2f}-{hi:.2f}_"
        f"{args.runs}_{SCHEMA_TAG}_branchrows_directSI.parquet"
    )
    return os.path.join(args.save_path, name)


def parse_args():
    p = argparse.ArgumentParser(description="AC/DC-OPF dataset generator.")
    p.add_argument("--preset", required=True)
    p.add_argument("--runs", type=int, default=36000)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    p.add_argument("--chunksize", type=int, default=8)
    p.add_argument("--scenario_level", default="opf_backbone",
                   choices=list(OPF_PRESETS.keys()))
    p.add_argument("--load_scale_lo", type=float, default=None)
    p.add_argument("--load_scale_hi", type=float, default=None)
    p.add_argument("--vm_min", type=float, default=0.90)
    p.add_argument("--vm_max", type=float, default=1.10)
    p.add_argument("--line_max_loading", type=float, default=None)
    p.add_argument("--opf_dc", action="store_true")
    p.add_argument("--start_mode", default="dc_compile",
                   choices=["auto", "manual_flat", "ppc_v0", "dc_compile"])
    p.add_argument("--drop_nonconverged", action="store_true")
    p.add_argument("--save_path", default="./out")
    p.add_argument("--save_steps", type=int, default=2000,
                   help="Rows buffered in memory before a write.")
    # Row groups are the unit of random-access I/O in the training loader:
    # reading one sample decodes its whole row group, so 2000-row groups make
    # shuffled access read ~100x more than it needs. Keep the write buffer
    # large (throughput) but the row group small (read amplification).
    p.add_argument("--row_group_size", type=int, default=20,
                   help="Rows per Parquet row group. Small values keep "
                        "shuffled reads in the training loader cheap.")
    p.add_argument("--no_normalize_angle_datum", action="store_true",
                   help="Keep the case's native slack angle instead of "
                        "rotating the reference bus to 0 degrees.")
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    preset_cfg = OPF_PRESETS[args.scenario_level]
    lo, hi = preset_cfg["load_scale_range"]
    if args.load_scale_lo is not None:
        lo = args.load_scale_lo
    if args.load_scale_hi is not None:
        hi = args.load_scale_hi

    cfg = dict(
        preset=args.preset,
        load_scale_range=(lo, hi),
        jitter_load=preset_cfg["jitter_load"],
        jitter_load_q=preset_cfg["jitter_load_q"],
        start_mode=args.start_mode,
        vm_limits=(args.vm_min, args.vm_max),
        line_max_loading=args.line_max_loading,
        opf_dc=bool(args.opf_dc),
        drop_nonconverged=bool(args.drop_nonconverged),
        normalize_angle_datum=not args.no_normalize_angle_datum,
    )

    path = build_output_filename(args)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if args.overwrite and os.path.exists(path):
        os.remove(path)

    print(f"[OPF] preset          = {args.preset}")
    print(f"[OPF] scenario        = {args.scenario_level}  load scale {lo:.2f}-{hi:.2f}")
    print(f"[OPF] formulation     = {'DC-OPF' if args.opf_dc else 'AC-OPF'}")
    print(f"[OPF] bus vm limits   = [{args.vm_min}, {args.vm_max}]")
    print(f"[OPF] runs            = {args.runs}  workers = {args.workers}")
    print(f"[OPF] output          = {path}")

    schema = _schema()
    writer = pq.ParquetWriter(path, schema, compression="zstd", use_dictionary=True)

    t0 = time.time()
    written = 0
    attempted = 0
    buf = []
    try:
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker, initargs=(cfg,)
        ) as ex:
            seeds = range(args.seed0, args.seed0 + args.runs)
            for row in ex.map(_one_row, seeds, chunksize=args.chunksize):
                attempted += 1
                if row is not None:
                    buf.append(row)
                if len(buf) >= args.save_steps:
                    writer.write_table(pa.Table.from_pylist(buf, schema=schema),
                                   row_group_size=args.row_group_size)
                    written += len(buf)
                    buf = []
                    el = time.time() - t0
                    print(f"[OPF] {written:,} written / {attempted:,} attempted "
                          f"({written/max(el,1e-9):.1f} rows/s)", flush=True)
        if buf:
            writer.write_table(pa.Table.from_pylist(buf, schema=schema),
                                   row_group_size=args.row_group_size)
            written += len(buf)
    finally:
        writer.close()

    # The name carries the row count, but --runs is only a request: dropped
    # non-converged samples make the two differ. Rename to what the file
    # actually holds so the name never overstates the dataset.
    if written != args.runs:
        final = path.replace(f"_{args.runs}_{SCHEMA_TAG}_", f"_{written}_{SCHEMA_TAG}_")
        if final != path and not os.path.exists(final):
            os.replace(path, final)
            print(f"[INFO] Renamed to actual row count: {os.path.basename(final)}")
            path = final

    el = time.time() - t0
    rate = written / el if el > 0 else 0.0
    conv = 100.0 * written / max(attempted, 1)
    print(f"[DONE] Wrote {written:,} rows in {el:,.1f}s ({rate:.0f} rows/s)")
    print(f"[DONE] OPF convergence: {conv:.1f}% ({written:,}/{attempted:,})")
    print(f"[INFO] File size: {os.path.getsize(path)/1e9:.3f} GB -> {path}")


if __name__ == "__main__":
    main()
