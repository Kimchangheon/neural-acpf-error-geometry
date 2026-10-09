#!/usr/bin/env python3
"""Rewrite the v2 corpus with uniform 20-row row groups.

Row groups are the unit of random-access I/O: a shuffled training loader must
read a whole group to get one row. The v2 files were written with one row group
per --save_steps flush, so group size varies from 100 to 4000 across the corpus
and read amplification is both large and non-uniform.

This is a re-encode, not a regeneration: rows, schema and values are untouched,
only the row-group layout changes. Streams batch by batch so peak memory stays
bounded regardless of file size (the largest file here is ~19 GB).

Writes to a temporary file in the same directory and os.replace()s it into
position, so an interrupted run can never leave a half-written dataset under the
real name.
"""
import argparse
import glob
import os
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

OUT = "/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out"
TAG = "u0clean_sgenpert_pvqvstart"


def reencode(path, target=20, batch_rows=2000):
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    n_before = pf.metadata.num_rows
    rg_before = pf.metadata.num_row_groups
    comp = pf.metadata.row_group(0).column(0).compression.lower()
    if comp == "uncompressed":
        comp = "none"

    tmp = path + ".reencode.tmp"
    written = 0
    w = pq.ParquetWriter(tmp, schema, compression=comp, use_dictionary=True)
    try:
        for batch in pf.iter_batches(batch_size=batch_rows):
            t = pa.Table.from_batches([batch], schema=schema)
            w.write_table(t, row_group_size=target)
            written += t.num_rows
    finally:
        w.close()

    # Verify before replacing: row count and schema must match exactly.
    chk = pq.ParquetFile(tmp)
    if chk.metadata.num_rows != n_before:
        os.remove(tmp)
        raise RuntimeError(f"row count changed {n_before} -> {chk.metadata.num_rows}")
    if chk.schema_arrow != schema:
        os.remove(tmp)
        raise RuntimeError("schema changed")
    rg_after = chk.metadata.num_row_groups
    sizes = {chk.metadata.row_group(i).num_rows for i in range(min(rg_after, 50))}
    if max(sizes) > target:
        os.remove(tmp)
        raise RuntimeError(f"row groups still too large: {sorted(sizes)[-3:]}")

    os.replace(tmp, path)
    return n_before, rg_before, rg_after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=20)
    ap.add_argument("--pattern", type=str, default=f"*{TAG}*directSI.parquet")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(OUT, a.pattern)))
    print(f"{len(files)} files matching {a.pattern}\n")
    if not a.apply:
        print(f"{'file':<52}{'rows':>9}{'groups':>8}{'rows/group':>12}")
        for f in files:
            m = pq.ParquetFile(f).metadata
            rg = [m.row_group(i).num_rows for i in range(min(m.num_row_groups, 200))]
            print(f"{os.path.basename(f).split('_ppcY_')[0]:<52}{m.num_rows:>9,}"
                  f"{m.num_row_groups:>8,}{f'{min(rg)}..{max(rg)}':>12}")
        print("\nDry run. Re-run with --apply to rewrite in place.")
        return

    t0 = time.time()
    for i, f in enumerate(files, 1):
        g = os.path.basename(f).split("_ppcY_")[0]
        s = time.time()
        try:
            n, rb, ra = reencode(f, a.target)
        except Exception as exc:
            print(f"[{i:2d}/{len(files)}] {g:<24} FAILED: {exc}", flush=True)
            sys.exit(1)
        print(f"[{i:2d}/{len(files)}] {g:<24} {n:>8,} rows  "
              f"{rb:>6,} -> {ra:>7,} groups  {time.time()-s:6.0f}s", flush=True)
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
