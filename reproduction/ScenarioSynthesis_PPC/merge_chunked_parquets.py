#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

import pyarrow.parquet as pq


RUNS_SUFFIX_RE = re.compile(r"_(\d+)_NR_branchrows_directSI\.parquet$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge parquet chunks produced by Slurm array jobs."
    )
    parser.add_argument(
        "--case",
        required=True,
        help="Case name, for example: case6470rte or case9241pegase",
    )
    parser.add_argument(
        "--input-root",
        default="./out",
        help="Root directory that contains the chunk directories.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output parquet path. Defaults to <input-root>/<merged-name>.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output parquet if it already exists.",
    )
    parser.add_argument(
        "--chunk-prefix",
        default="",
        help=(
            "Chunk-directory infix to merge, e.g. 'ppnr' or 'cnrclean'. Selects "
            "<case>_<prefix>_chunk_* exclusively. Strongly preferred when several "
            "run families coexist, since auto-detection picks only one family and "
            "would otherwise choose by fixed priority rather than by intent."
        ),
    )
    return parser.parse_args()


def chunk_sort_key(path: Path):
    try:
        return int(path.name.rsplit("_", 1)[-1])
    except ValueError:
        return path.name


def find_chunk_files(case_name: str, input_root: Path, chunk_prefix: str = ""):
    # Only ONE family is ever merged. Mixing families (e.g. stale custom-NR
    # "_backbone_chunk_" dirs with fresh "_ppnr_chunk_" dirs) would silently
    # blend runs that used different solvers or start points.
    if chunk_prefix:
        prefixes = (f"{case_name}_{chunk_prefix.strip('_')}_chunk_",)
    else:
        prefixes = (
            f"{case_name}_cnrclean_chunk_",
            f"{case_name}_ppnr_chunk_",
            f"{case_name}_cnr_chunk_",
            f"{case_name}_backbone_chunk_",
            f"{case_name}_A_chunk_",
            f"{case_name}_chunk_",
        )

    chunk_dirs = []
    for prefix in prefixes:
        matched = [
            path
            for path in input_root.iterdir()
            if path.is_dir() and path.name.startswith(prefix)
        ]
        if matched:
            chunk_dirs = sorted(matched, key=chunk_sort_key)
            print(f"[MERGE] Using chunk family '{prefix}*' ({len(chunk_dirs)} dirs)")
            break

    parquet_files = []
    for chunk_dir in chunk_dirs:
        parquet_files.extend(sorted(chunk_dir.glob("*.parquet")))
    return parquet_files


def build_output_path(files, total_rows: int, output_arg: str, input_root: Path, case_name: str):
    if output_arg:
        return Path(output_arg)

    first_name = files[0].name
    merged_name = RUNS_SUFFIX_RE.sub(
        f"_{total_rows}_NR_branchrows_directSI.parquet",
        first_name,
    )
    if merged_name == first_name:
        merged_name = f"{case_name}_merged_{total_rows}.parquet"
    return input_root / merged_name


def main():
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()

    if not input_root.exists():
        raise SystemExit(f"[ERROR] Input root does not exist: {input_root}")

    files = find_chunk_files(args.case, input_root, args.chunk_prefix)
    if not files:
        where = f" (chunk-prefix={args.chunk_prefix!r})" if args.chunk_prefix else ""
        raise SystemExit(
            f"[ERROR] No chunk parquet files found for case '{args.case}'{where} "
            f"under {input_root}"
        )

    schema = None
    total_rows = 0
    row_counts = []

    for path in files:
        parquet_file = pq.ParquetFile(path)
        file_schema = parquet_file.schema_arrow
        if schema is None:
            schema = file_schema
        elif schema != file_schema:
            raise SystemExit(f"[ERROR] Schema mismatch in {path}")

        rows = parquet_file.metadata.num_rows
        row_counts.append((path, rows))
        total_rows += rows

    output_path = build_output_path(files, total_rows, args.output, input_root, args.case)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        raise SystemExit(
            f"[ERROR] Output already exists: {output_path}\n"
            f"Use --overwrite or pass --output to a new path."
        )

    if output_path.exists():
        output_path.unlink()

    writer = pq.ParquetWriter(
        where=str(output_path),
        schema=schema,
        compression="zstd",
        use_dictionary=True,
    )

    try:
        for path, rows in row_counts:
            print(f"[MERGE] {path} ({rows:,} rows)")
            parquet_file = pq.ParquetFile(path)
            for row_group_idx in range(parquet_file.num_row_groups):
                writer.write_table(parquet_file.read_row_group(row_group_idx))
    finally:
        writer.close()

    print(f"[DONE] Merged {len(files)} files")
    print(f"[DONE] Total rows: {total_rows:,}")
    print(f"[DONE] Output: {output_path}")


if __name__ == "__main__":
    main()
