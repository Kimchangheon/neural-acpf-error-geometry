#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import io
import math
import multiprocessing as mp
import os.path
import sys
import time
import traceback
from typing import Any, Dict, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from case_generator_lvn_snapshot_envelope import (
    DEFAULT_LVN_CGMES,
    DEFAULT_SNAPSHOT_ROOT,
    align_bounds_to_base_net,
    case_generation_lvn_snapshot_envelope,
    get_or_build_snapshot_load_bounds,
    prepare_lvn_base_net,
)
from newton_raphson_improved import newtonrapson


def ndarray_to_npy_bytes(x: Any) -> bytes:
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    buf = io.BytesIO()
    np.save(buf, x, allow_pickle=False)
    return buf.getvalue()


def ybus_si_to_pu(Y_si: np.ndarray, Vbase_bus: np.ndarray, S_base: float) -> np.ndarray:
    scale = np.outer(Vbase_bus, Vbase_bus) / float(S_base)
    return np.asarray(Y_si, dtype=np.complex128) * scale.astype(np.float64, copy=False)


def u_si_to_pu_per_bus(u_si: np.ndarray, Vbase_bus: np.ndarray) -> np.ndarray:
    return np.asarray(u_si, dtype=np.complex128) / np.asarray(Vbase_bus, dtype=np.float64)


def s_si_to_pu(S_si: np.ndarray, S_base: float) -> np.ndarray:
    return np.asarray(S_si, dtype=np.complex128) / float(S_base)


def u_pu_to_si_per_bus(u_pu: np.ndarray, Vbase_bus: np.ndarray) -> np.ndarray:
    return np.asarray(u_pu, dtype=np.complex128) * np.asarray(Vbase_bus, dtype=np.float64)


def s_pu_to_si(S_pu: np.ndarray, S_base: float) -> np.ndarray:
    return np.asarray(S_pu, dtype=np.complex128) * float(S_base)


def convert_nr_inputs_to_pu(
    Y_matrix_si: np.ndarray,
    s_multi_si: np.ndarray,
    u_start_si: np.ndarray,
    vn_kv: np.ndarray,
    S_base: float,
):
    Vbase_bus = np.asarray(vn_kv, dtype=np.float64) * 1e3

    Y_pu = ybus_si_to_pu(np.asarray(Y_matrix_si, dtype=np.complex128), Vbase_bus, S_base)
    s_pu = s_si_to_pu(np.asarray(s_multi_si, dtype=np.complex128), S_base)
    u_pu = u_si_to_pu_per_bus(np.asarray(u_start_si, dtype=np.complex128), Vbase_bus)

    return Y_pu, s_pu, u_pu, Vbase_bus


class ParquetAppendWriter:
    def __init__(
        self,
        path: str,
        compression: str = "zstd",
        overwrite: bool = True,
        save_y_matrix: bool = True,
    ):
        self.path = path
        self.save_y_matrix = bool(save_y_matrix)

        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        if overwrite and os.path.exists(path):
            os.remove(path)

        fields = [
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
        ]

        if self.save_y_matrix:
            fields.append(pa.field("Y_matrix", pa.binary()))

        fields.extend([
            pa.field("u_start", pa.binary()),
            pa.field("u_newton", pa.binary()),
            pa.field("S_start", pa.binary()),
            pa.field("S_newton", pa.binary()),
        ])

        self._schema = pa.schema(fields)
        self._writer = pq.ParquetWriter(
            where=path,
            schema=self._schema,
            compression=compression,
            use_dictionary=True,
        )

    def write_records(self, records: List[Dict[str, Any]]):
        if not records:
            return

        cols = {name: [] for name in self._schema.names}
        for record in records:
            for key in cols.keys():
                cols[key].append(record[key])

        arrays = []
        for name, field in zip(self._schema.names, self._schema):
            if field.type == pa.binary():
                arrays.append(pa.array(cols[name], type=pa.binary()))
            elif field.type == pa.int32():
                arrays.append(pa.array(cols[name], type=pa.int32()))
            elif field.type == pa.float64():
                arrays.append(pa.array(cols[name], type=pa.float64()))
            elif pa.types.is_string(field.type):
                arrays.append(pa.array(cols[name], type=pa.string()))
            else:
                arrays.append(pa.array(cols[name]))

        table = pa.Table.from_arrays(arrays, schema=self._schema)
        self._writer.write_table(table)

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None


_CFG: Dict[str, Any] = {}
_RNG = None
_BASE_NET = None
_ALIGNED_BOUNDS = None
DBG = os.environ.get("GEN_DEBUG", "0") == "1"


def _init_worker(cfg: dict, seed_base: int):
    global _CFG, _RNG, _BASE_NET, _ALIGNED_BOUNDS

    _CFG = cfg
    pid = os.getpid()
    ss = np.random.SeedSequence([seed_base & 0xFFFFFFFF, pid & 0xFFFFFFFF])
    _RNG = np.random.default_rng(ss)
    np.random.seed(int(ss.generate_state(1, dtype=np.uint32)[0]))

    _BASE_NET = prepare_lvn_base_net(
        _CFG["base_cgmes_path"],
        case_name=str(_CFG["case_name"]).strip(),
        cgmes_version=str(_CFG["cgmes_version"]).strip(),
        ignore_errors=bool(_CFG["cgmes_ignore_errors"]),
        apply_model_a_cleanup=bool(_CFG["cgmes_model_a_cleanup"]),
        base_sn_mva=_CFG.get("base_sn_mva"),
    )
    _ALIGNED_BOUNDS = align_bounds_to_base_net(
        _BASE_NET,
        _CFG["load_bounds_df"],
    )


def _generate_one_record_serialized() -> Dict[str, Any]:
    global _CFG, _RNG, _BASE_NET, _ALIGNED_BOUNDS

    K = int(_CFG["K"])
    save_y_matrix = bool(_CFG["save_y_matrix"])
    pu_nr = bool(_CFG["pu_nr"])

    sample_seed = int(_RNG.integers(0, 2**32 - 1, dtype=np.uint32))

    out = case_generation_lvn_snapshot_envelope(
        base_net=_BASE_NET,
        aligned_bounds=_ALIGNED_BOUNDS,
        seed=sample_seed,
        rng=_RNG,
        case_name=str(_CFG["case_name"]).strip(),
        sample_mode=str(_CFG["sample_mode"]).strip(),
        ybus_mode=str(_CFG["ybus_mode"]).strip(),
        start_mode=str(_CFG["start_mode"]).strip(),
        v2_load_scale_range=_CFG.get("v2_load_scale_range"),
        v2_jitter_load=float(_CFG.get("v2_jitter_load", 0.0)),
        v2_jitter_load_q=float(_CFG.get("v2_jitter_load_q", 0.0)),
        v2_jitter_gen=float(_CFG.get("v2_jitter_gen", 0.0)),
        v2_perturb_sgen=bool(_CFG.get("v2_perturb_sgen", False)),
        v2_pv_q_from_vstart=bool(_CFG.get("v2_pv_q_from_vstart", False)),
    )

    (
        gridtype_out, bus_typ, s_multi, u_start, Y_matrix, is_connected,
        Branch_f_bus, Branch_t_bus, Branch_status,
        Branch_tau, Branch_shift_deg,
        Branch_y_series_from, Branch_y_series_to, Branch_y_series_ft,
        Branch_y_shunt_from, Branch_y_shunt_to,
        Y_shunt_bus,
        Is_trafo, Branch_hv_is_f, Branch_n,
        Y_Lines, Y_C_Lines,
        U_base, S_base, vn_kv
    ) = out

    bus_number = int(len(bus_typ))
    branch_number = int(len(Branch_f_bus))

    ymat_bytes = ndarray_to_npy_bytes(np.asarray(Y_matrix, dtype=np.complex128).copy()) if save_y_matrix else None

    if not is_connected:
        u_newton_si = np.zeros_like(u_start, dtype=np.complex128)
        S_newton_si = np.zeros_like(s_multi, dtype=np.complex128)
    else:
        bus_typ_arr = np.asarray(bus_typ, dtype=np.int64).copy()

        if pu_nr:
            Y_for_nr, S_for_nr, U_for_nr, Vbase_bus = convert_nr_inputs_to_pu(
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
            Vbase_bus = None

        nr_out = newtonrapson(
            bus_typ_arr,
            Y_for_nr,
            S_for_nr,
            U_for_nr,
            K=K,
            diagnose=bool(_CFG["diagnose_nr"]),
            print_misinf=bool(_CFG["print_misinf"]),
            return_diagnostics=True,
            near_misinf_tol=float(_CFG["near_misinf_tol"]),
            convergence_mode=str(_CFG["convergence_mode"]),
            step_tol=float(_CFG["step_tol"]),
            mismatch_tol=float(_CFG["mismatch_tol"]),
        )

        u_newton_raw, _I_unused, S_newton_raw, _nr_diag = nr_out

        if pu_nr:
            u_newton_arr = np.asarray(u_newton_raw)
            s_newton_arr = np.asarray(S_newton_raw)

            if u_newton_arr.size == 0:
                u_newton_si = np.asarray(u_newton_arr, dtype=np.complex128)
            else:
                u_newton_si = u_pu_to_si_per_bus(u_newton_arr, Vbase_bus)

            if s_newton_arr.size == 0:
                S_newton_si = np.asarray(s_newton_arr, dtype=np.complex128)
            else:
                S_newton_si = s_pu_to_si(s_newton_arr, float(S_base))
        else:
            u_newton_si = np.asarray(u_newton_raw, dtype=np.complex128)
            S_newton_si = np.asarray(S_newton_raw, dtype=np.complex128)

    record = {
        "bus_number": bus_number,
        "branch_number": branch_number,
        "gridtype": gridtype_out,
        "U_base": float(U_base),
        "S_base": float(S_base),
        "bus_typ": ndarray_to_npy_bytes(np.asarray(bus_typ, dtype=np.int32)),
        "vn_kv": ndarray_to_npy_bytes(np.asarray(vn_kv, dtype=np.float64)),
        "Y_shunt_bus": ndarray_to_npy_bytes(np.asarray(Y_shunt_bus, dtype=np.complex128)),
        "Branch_f_bus": ndarray_to_npy_bytes(np.asarray(Branch_f_bus, dtype=np.int32)),
        "Branch_t_bus": ndarray_to_npy_bytes(np.asarray(Branch_t_bus, dtype=np.int32)),
        "Branch_status": ndarray_to_npy_bytes(np.asarray(Branch_status, dtype=np.int8)),
        "Branch_tau": ndarray_to_npy_bytes(np.asarray(Branch_tau, dtype=np.float64)),
        "Branch_shift_deg": ndarray_to_npy_bytes(np.asarray(Branch_shift_deg, dtype=np.float64)),
        "Branch_y_series_from": ndarray_to_npy_bytes(np.asarray(Branch_y_series_from, dtype=np.complex128)),
        "Branch_y_series_to": ndarray_to_npy_bytes(np.asarray(Branch_y_series_to, dtype=np.complex128)),
        "Branch_y_series_ft": ndarray_to_npy_bytes(np.asarray(Branch_y_series_ft, dtype=np.complex128)),
        "Branch_y_shunt_from": ndarray_to_npy_bytes(np.asarray(Branch_y_shunt_from, dtype=np.complex128)),
        "Branch_y_shunt_to": ndarray_to_npy_bytes(np.asarray(Branch_y_shunt_to, dtype=np.complex128)),
        "Is_trafo": ndarray_to_npy_bytes(np.asarray(Is_trafo, dtype=np.int8)),
        "Branch_hv_is_f": ndarray_to_npy_bytes(np.asarray(Branch_hv_is_f, dtype=np.int8)),
        "Branch_n": ndarray_to_npy_bytes(np.asarray(Branch_n, dtype=np.float64)),
        "Y_Lines": ndarray_to_npy_bytes(np.asarray(Y_Lines, dtype=np.complex128)),
        "Y_C_Lines": ndarray_to_npy_bytes(np.asarray(Y_C_Lines, dtype=np.float64)),
        "u_start": ndarray_to_npy_bytes(np.asarray(u_start, dtype=np.complex128)),
        "u_newton": ndarray_to_npy_bytes(np.asarray(u_newton_si, dtype=np.complex128)),
        "S_start": ndarray_to_npy_bytes(np.asarray(s_multi, dtype=np.complex128)),
        "S_newton": ndarray_to_npy_bytes(np.asarray(S_newton_si, dtype=np.complex128)),
    }

    if save_y_matrix:
        record["Y_matrix"] = ymat_bytes

    return record


def _generate_batch(n_rows: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    err_shown = 0

    for _ in range(n_rows):
        try:
            out.append(_generate_one_record_serialized())
        except Exception as exc:
            if DBG and err_shown < 3:
                print("[WORKER ERROR]", repr(exc))
                traceback.print_exc()
                err_shown += 1
            continue

    return out


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Parallel dataset generation for LVN using per-load snapshot min/max envelopes."
    )

    parser.add_argument("--base_cgmes_path", type=str, default=DEFAULT_LVN_CGMES)
    parser.add_argument("--snapshot_root", type=str, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--case_name", type=str, default="LVN_snapshot_envelope")
    parser.add_argument("--cgmes_version", type=str, default="2.4.15")
    parser.add_argument(
        "--base_sn_mva",
        type=float,
        default=0.0,
        help="Override pandapower net.sn_mva before ppc/Ybus construction. Use <=0 to keep CGMES import default.",
    )

    parser.add_argument(
        "--cgmes_model_a_cleanup",
        dest="cgmes_model_a_cleanup",
        action="store_true",
        help="Apply LVN Model A cleanup to the base LVN net (default).",
    )
    parser.add_argument(
        "--no_cgmes_model_a_cleanup",
        dest="cgmes_model_a_cleanup",
        action="store_false",
        help="Do not apply LVN Model A cleanup to the base LVN net.",
    )
    parser.set_defaults(cgmes_model_a_cleanup=True)

    parser.add_argument(
        "--cgmes_ignore_errors",
        dest="cgmes_ignore_errors",
        action="store_true",
        help="Ignore CGMES converter errors (default).",
    )
    parser.add_argument(
        "--no_cgmes_ignore_errors",
        dest="cgmes_ignore_errors",
        action="store_false",
        help="Do not ignore CGMES converter errors.",
    )
    parser.set_defaults(cgmes_ignore_errors=True)

    parser.add_argument(
        "--bounds_cache_path",
        type=str,
        default="",
        help="Optional .npz path for cached snapshot load min/max bounds.",
    )
    parser.add_argument(
        "--sample_mode",
        type=str,
        default="coupled",
        choices=["coupled", "independent"],
        help="How to sample P/Q inside each load's daily min/max envelope.",
    )
    parser.add_argument(
        "--ybus_mode",
        type=str,
        default="ppcY",
        choices=["ppcY", "stamped"],
    )

    parser.add_argument("--K", type=int, default=40)
    parser.add_argument("--runs", type=int, default=10000)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--rows_per_task", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--save_y_matrix", dest="save_y_matrix", action="store_true")
    group.add_argument("--no_save_y_matrix", dest="save_y_matrix", action="store_false")
    parser.set_defaults(save_y_matrix=True)

    parser.add_argument("--pu_nr", action="store_true")
    parser.add_argument("--diagnose_nr", action="store_true")
    parser.add_argument("--print_misinf", action="store_true")
    parser.add_argument("--near_misinf_tol", type=float, default=1e-3)
    parser.add_argument(
        "--start_mode",
        type=str,
        default="dc_compile",
        choices=["auto", "manual_flat", "ppc_v0", "dc_compile"],
    )
    parser.add_argument(
        "--convergence_mode",
        type=str,
        default="misinf",
        choices=["two_step", "misinf"],
    )
    parser.add_argument("--step_tol", type=float, default=5e-4)
    parser.add_argument("--mismatch_tol", type=float, default=1e-8)
    parser.add_argument("--v2_perturb", action="store_true",
                        help="Apply the v2 global scale/jitter law after envelope sampling.")
    parser.add_argument("--v2_load_scale_lo", type=float, default=0.60)
    parser.add_argument("--v2_load_scale_hi", type=float, default=1.40)
    parser.add_argument("--v2_jitter_load", type=float, default=0.10)
    parser.add_argument("--v2_jitter_load_q", type=float, default=0.15)
    parser.add_argument("--v2_jitter_gen", type=float, default=0.05)
    parser.add_argument("--v2_perturb_sgen", action="store_true")
    parser.add_argument("--v2_pv_q_from_vstart", action="store_true")

    return parser.parse_args()


def build_output_filename(args) -> str:
    nr_unit = "puNR" if args.pu_nr else "siNR"
    name = (
        f"{args.case_name}_{args.sample_mode}_{args.ybus_mode}_{args.start_mode}_"
        f"{nr_unit}_{args.runs}_NR_branchrows_directSI.parquet"
    )
    return os.path.join(args.save_path, name)


def get_parquet_file_size(path: str) -> int:
    return os.path.getsize(path) if os.path.exists(path) else 0


def main():
    args = parse_args()

    print("[INFO] Building or loading LVN snapshot load bounds")
    load_bounds_df = get_or_build_snapshot_load_bounds(
        args.snapshot_root,
        cgmes_version=args.cgmes_version,
        ignore_errors=bool(args.cgmes_ignore_errors),
        cache_path=args.bounds_cache_path,
        progress_every=10,
    )

    workers = args.workers if args.workers and args.workers > 0 else (os.cpu_count() or 1)
    runs = int(args.runs)
    rows_per_task = max(int(args.rows_per_task), 1)
    save_steps = max(int(args.save_steps), 1)
    save_y_matrix = bool(args.save_y_matrix)

    filename = build_output_filename(args)
    print(f"[INFO] Output file: {filename}")

    cfg = dict(
        base_cgmes_path=str(args.base_cgmes_path).strip(),
        snapshot_root=str(args.snapshot_root).strip(),
        case_name=str(args.case_name).strip(),
        cgmes_version=str(args.cgmes_version).strip(),
        cgmes_model_a_cleanup=bool(args.cgmes_model_a_cleanup),
        cgmes_ignore_errors=bool(args.cgmes_ignore_errors),
        base_sn_mva=float(args.base_sn_mva) if float(args.base_sn_mva) > 0 else None,
        bounds_cache_path=str(args.bounds_cache_path).strip(),
        sample_mode=str(args.sample_mode).strip(),
        ybus_mode=str(args.ybus_mode).strip(),
        K=int(args.K),
        save_y_matrix=save_y_matrix,
        pu_nr=bool(args.pu_nr),
        diagnose_nr=bool(args.diagnose_nr),
        print_misinf=bool(args.print_misinf),
        near_misinf_tol=float(args.near_misinf_tol),
        start_mode=str(args.start_mode).strip(),
        convergence_mode=str(args.convergence_mode).strip(),
        step_tol=float(args.step_tol),
        mismatch_tol=float(args.mismatch_tol),
        v2_load_scale_range=(float(args.v2_load_scale_lo), float(args.v2_load_scale_hi)) if args.v2_perturb else None,
        v2_jitter_load=float(args.v2_jitter_load) if args.v2_perturb else 0.0,
        v2_jitter_load_q=float(args.v2_jitter_load_q) if args.v2_perturb else 0.0,
        v2_jitter_gen=float(args.v2_jitter_gen) if args.v2_perturb else 0.0,
        v2_perturb_sgen=bool(args.v2_perturb_sgen or args.v2_perturb),
        v2_pv_q_from_vstart=bool(args.v2_pv_q_from_vstart or args.v2_perturb),
        load_bounds_df=load_bounds_df,
    )

    num_tasks = math.ceil(runs / rows_per_task)
    task_sizes = [rows_per_task] * (num_tasks - 1) + [runs - rows_per_task * (num_tasks - 1)]

    if sys.platform.startswith("win"):
        ctx = mp.get_context("spawn")
    else:
        ctx = mp.get_context("fork")

    total_written = 0
    start = time.time()
    writer = ParquetAppendWriter(
        filename,
        compression="zstd",
        overwrite=args.overwrite,
        save_y_matrix=save_y_matrix,
    )

    print("[INFO] Configuration")
    print(f"  base_cgmes_path           = {args.base_cgmes_path}")
    print(f"  snapshot_root             = {args.snapshot_root}")
    print(f"  case_name                 = {args.case_name}")
    print(f"  cgmes_version             = {args.cgmes_version}")
    print(f"  cgmes_model_a_cleanup     = {args.cgmes_model_a_cleanup}")
    print(f"  cgmes_ignore_errors       = {args.cgmes_ignore_errors}")
    print(f"  base_sn_mva               = {args.base_sn_mva if args.base_sn_mva > 0 else 'CGMES default'}")
    print(f"  bounds_cache_path         = {args.bounds_cache_path}")
    print(f"  sample_mode               = {args.sample_mode}")
    print(f"  bounds_n_loads            = {len(load_bounds_df)}")
    print(f"  ybus_mode                 = {args.ybus_mode}")
    print(f"  K                         = {args.K}")
    print(f"  pu_nr                     = {args.pu_nr}")
    print(f"  start_mode                = {args.start_mode}")
    print(f"  diagnose_nr               = {args.diagnose_nr}")
    print(f"  convergence_mode          = {args.convergence_mode}")
    print(f"  step_tol                  = {args.step_tol}")
    print(f"  mismatch_tol              = {args.mismatch_tol}")
    print(f"  v2_perturb                = {args.v2_perturb}")
    print(f"  workers                   = {workers}")
    print(f"  rows_per_task             = {rows_per_task}")
    print(f"  save_steps                = {save_steps}")
    print(f"  save_y_matrix             = {save_y_matrix}")

    if args.print_misinf and workers != 1:
        print("[WARN] print_misinf=True with workers>1 will produce interleaved logs.")

    buffer: List[Dict[str, Any]] = []

    try:
        with ctx.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(cfg, int(time.time())),
            maxtasksperchild=1000,
        ) as pool:
            for batch in pool.imap_unordered(_generate_batch, task_sizes, chunksize=1):
                if batch:
                    buffer.extend(batch)

                if len(buffer) >= save_steps:
                    writer.write_records(buffer)
                    total_written += len(buffer)
                    buffer.clear()

                    elapsed = time.time() - start
                    rate = total_written / max(elapsed, 1e-6)
                    print(f"[INFO] {total_written:,}/{runs:,} rows written ({rate:,.0f} rows/s)")

            if buffer:
                writer.write_records(buffer)
                total_written += len(buffer)
                buffer.clear()

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user. Flushing remaining buffer...")
        if buffer:
            writer.write_records(buffer)
            total_written += len(buffer)
    finally:
        writer.close()

    elapsed = time.time() - start
    rate = total_written / max(elapsed, 1e-6)
    size_gb = get_parquet_file_size(filename) / 1e9
    print(f"[DONE] Wrote {total_written:,} rows in {elapsed:,.1f}s ({rate:,.0f} rows/s)")
    print(f"[INFO] File size: {size_gb:.3f} GB -> {filename}")


if __name__ == "__main__":
    main()
