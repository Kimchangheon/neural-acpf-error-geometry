#!/usr/bin/env python3
"""Add custom-Newton trajectories to an existing PF parquet.

This is the preferred way to build the LVN_heo1 trajectory corpus: it keeps
the exact scenarios, ordering, pandapower final labels, and train/test split of
the existing v2 dataset, while adding a deterministic complex128 Newton path
from each stored ``u_start``.  The ordinary generator can also emit the same
columns for newly sampled custom-NR datasets via ``--save_nr_trajectory``.
"""

import argparse
import io
import math
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import multiprocessing as mp

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from case_generator_all_test_cases_pandapower_consider_ppc_branch_row import (
    reconstruct_Y_pandapower_branchrows_direct_SI,
)
from newton_raphson_improved import newtonrapson


TRAJECTORY_FIELDS = (
    pa.field("u_nr_trajectory", pa.binary()),
    pa.field("nr_misinf_history", pa.binary()),
    pa.field("nr_step_history", pa.binary()),
    pa.field("nr_trajectory_length", pa.int32()),
    pa.field("nr_trajectory_converged", pa.int8()),
    pa.field("nr_trajectory_final_misinf", pa.float64()),
    pa.field("nr_trajectory_final_vs_label_max_pu", pa.float64()),
)

_WORKER = {}


def npy_load(cell):
    with io.BytesIO(cell) as buf:
        return np.load(buf, allow_pickle=False)


def npy_bytes(array):
    with io.BytesIO() as buf:
        np.save(buf, np.asarray(array), allow_pickle=False)
        return buf.getvalue()


def _cell(table, name, row=0):
    return npy_load(table[name][row].as_py())


def _init_worker(Y_pu, bus_type, vbase_bus, K, mismatch_tol, require_converged):
    global _WORKER
    _WORKER = {
        "Y": np.asarray(Y_pu, dtype=np.complex128),
        "bus_type": np.asarray(bus_type, dtype=np.int64),
        "vbase": np.asarray(vbase_bus, dtype=np.float64),
        "K": int(K),
        "mismatch_tol": float(mismatch_tol),
        "require_converged": bool(require_converged),
    }


def _solve_one(task):
    row_id, u_start_cell, s_start_cell, u_label_cell, s_base = task
    vbase = _WORKER["vbase"]
    u0_pu = np.asarray(npy_load(u_start_cell), dtype=np.complex128) / vbase
    s_pu = np.asarray(npy_load(s_start_cell), dtype=np.complex128) / float(s_base)
    label_pu = np.asarray(npy_load(u_label_cell), dtype=np.complex128) / vbase

    _u, _i, _s, diag = newtonrapson(
        _WORKER["bus_type"],
        _WORKER["Y"],
        s_pu,
        u0_pu,
        K=_WORKER["K"],
        diagnose=True,
        return_diagnostics=True,
        return_trajectory=True,
        convergence_mode="misinf",
        mismatch_tol=_WORKER["mismatch_tol"],
        verbose=False,
    )
    converged = bool(diag.get("converged", False))
    if _WORKER["require_converged"] and not converged:
        raise RuntimeError(
            f"custom NR did not converge for source row {row_id}: "
            f"{diag.get('classification')} / {diag.get('failure_reason')}"
        )

    trajectory_pu = np.asarray(diag["voltage_trajectory"], dtype=np.complex128)
    if trajectory_pu.ndim != 2 or trajectory_pu.shape[1] != vbase.size:
        raise RuntimeError(
            f"bad trajectory shape for source row {row_id}: {trajectory_pu.shape}"
        )
    trajectory_si = trajectory_pu * vbase.reshape(1, -1)
    final_vs_label = float(np.max(np.abs(trajectory_pu[-1] - label_pu)))

    return (
        npy_bytes(trajectory_si.astype(np.complex128, copy=False)),
        npy_bytes(np.asarray(diag.get("misinf_history", []), dtype=np.float64)),
        npy_bytes(np.asarray(diag.get("step_history", []), dtype=np.float64)),
        int(trajectory_pu.shape[0]),
        int(converged),
        float(diag.get("final_misinf", np.nan)),
        final_vs_label,
    )


def _grid_from_first_row(pf):
    needed = [
        "bus_number", "S_base", "bus_typ", "vn_kv", "Y_shunt_bus",
        "Branch_f_bus", "Branch_t_bus", "Branch_status", "Branch_tau",
        "Branch_shift_deg", "Branch_y_series_from", "Branch_y_series_to",
        "Branch_y_series_ft", "Branch_y_shunt_from", "Branch_y_shunt_to",
    ]
    if "Y_matrix" in pf.schema_arrow.names:
        needed.append("Y_matrix")
    first = pf.read_row_group(0, columns=needed).slice(0, 1)
    n = int(first["bus_number"][0].as_py())
    s_base = float(first["S_base"][0].as_py())
    bus_type = _cell(first, "bus_typ").astype(np.int64, copy=False)
    vbase = _cell(first, "vn_kv").astype(np.float64, copy=False) * 1e3

    if "Y_matrix" in first.column_names:
        Y_si = _cell(first, "Y_matrix").astype(np.complex128, copy=False)
    else:
        Y_si = reconstruct_Y_pandapower_branchrows_direct_SI(
            N=n,
            Branch_f_bus=_cell(first, "Branch_f_bus"),
            Branch_t_bus=_cell(first, "Branch_t_bus"),
            Branch_status=_cell(first, "Branch_status"),
            Branch_tau=_cell(first, "Branch_tau"),
            Branch_shift_deg=_cell(first, "Branch_shift_deg"),
            Branch_y_series_from=_cell(first, "Branch_y_series_from"),
            Branch_y_series_to=_cell(first, "Branch_y_series_to"),
            Branch_y_series_ft=_cell(first, "Branch_y_series_ft"),
            Branch_y_shunt_from=_cell(first, "Branch_y_shunt_from"),
            Branch_y_shunt_to=_cell(first, "Branch_y_shunt_to"),
            Y_shunt_bus=_cell(first, "Y_shunt_bus"),
            Vbase_bus=vbase,
        )
    Y_pu = Y_si * np.outer(vbase, vbase) / s_base
    return Y_pu, bus_type, vbase, s_base


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--K", type=int, default=40)
    p.add_argument("--mismatch_tol", type=float, default=1e-8)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--allow_nonconverged", action="store_true",
        help="Write failed custom-NR paths instead of aborting the shard.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    src_path = Path(args.input).expanduser().resolve()
    dst_path = Path(args.output).expanduser().resolve()
    if not src_path.is_file():
        raise SystemExit(f"input parquet does not exist: {src_path}")
    if not (args.num_shards > 0 and 0 <= args.shard_index < args.num_shards):
        raise SystemExit("require num_shards > 0 and 0 <= shard_index < num_shards")
    if src_path == dst_path:
        raise SystemExit("input and output must differ")
    if dst_path.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {dst_path}; pass --overwrite")

    pf = pq.ParquetFile(src_path)
    overlap = set(pf.schema_arrow.names) & {f.name for f in TRAJECTORY_FIELDS}
    if overlap:
        raise SystemExit(f"input already has trajectory columns: {sorted(overlap)}")
    Y_pu, bus_type, vbase, first_sbase = _grid_from_first_row(pf)

    rg_start = args.shard_index * pf.num_row_groups // args.num_shards
    rg_end = (args.shard_index + 1) * pf.num_row_groups // args.num_shards
    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        dst_path.unlink()

    output_schema = pa.schema(
        list(pf.schema_arrow) + list(TRAJECTORY_FIELDS),
        metadata=pf.schema_arrow.metadata,
    )
    context = mp.get_context("fork")
    total = 0
    lengths = []
    final_misinf = []
    label_gap = []
    source_row_offset = sum(
        pf.metadata.row_group(i).num_rows for i in range(rg_start)
    )

    print(
        f"[trajectory] source={src_path} rows={pf.metadata.num_rows} "
        f"row_groups={pf.num_row_groups} grid={vbase.size} buses "
        f"S_base={first_sbase/1e6:g} MVA |Ypu|max={np.abs(Y_pu).max():.6e}"
    )
    print(
        f"[trajectory] shard={args.shard_index}/{args.num_shards} "
        f"row_groups=[{rg_start},{rg_end}) workers={workers} output={dst_path}"
    )

    writer = pq.ParquetWriter(
        dst_path, output_schema, compression="zstd", use_dictionary=True
    )
    try:
        with context.Pool(
            workers,
            initializer=_init_worker,
            initargs=(
                Y_pu, bus_type, vbase, args.K, args.mismatch_tol,
                not args.allow_nonconverged,
            ),
            maxtasksperchild=200,
        ) as pool:
            for rg in range(rg_start, rg_end):
                table = pf.read_row_group(rg)
                if args.max_rows > 0:
                    remaining = args.max_rows - total
                    if remaining <= 0:
                        break
                    table = table.slice(0, min(table.num_rows, remaining))
                tasks = [
                    (
                        source_row_offset + i,
                        table["u_start"][i].as_py(),
                        table["S_start"][i].as_py(),
                        table["u_newton"][i].as_py(),
                        float(table["S_base"][i].as_py()),
                    )
                    for i in range(table.num_rows)
                ]
                solved = pool.map(_solve_one, tasks, chunksize=1)
                columns = list(zip(*solved))
                augmented = table
                for field, values in zip(TRAJECTORY_FIELDS, columns):
                    augmented = augmented.append_column(
                        field, pa.array(values, type=field.type)
                    )
                writer.write_table(augmented, row_group_size=table.num_rows)

                lengths.extend(columns[3])
                final_misinf.extend(columns[5])
                label_gap.extend(columns[6])
                total += table.num_rows
                source_row_offset += table.num_rows
                print(
                    f"[trajectory] rows={total:,} length="
                    f"{min(lengths)}..{max(lengths)} "
                    f"misinf_max={max(final_misinf):.3e} "
                    f"label_gap_max={max(label_gap):.3e}",
                    flush=True,
                )
    finally:
        writer.close()

    check = pq.ParquetFile(dst_path)
    print(
        f"[done] rows={check.metadata.num_rows:,} row_groups={check.num_row_groups} "
        f"trajectory_length_mean={np.mean(lengths):.3f} "
        f"final_misinf_max={max(final_misinf):.6e} "
        f"final_vs_label_max_pu={max(label_gap):.6e} output={dst_path}"
    )


if __name__ == "__main__":
    main()
