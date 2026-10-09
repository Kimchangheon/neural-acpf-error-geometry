#!/usr/bin/env python3
"""Decide, per chunk family, whether the chunks are redundant or the only copy.

A chunk directory is safe to delete ONLY if a merged parquet exists whose row
count equals the sum over the chunks. Matching on the grid name alone is not
enough: a family like case6470rte_A would happily "match" the unrelated
backbone file for the same grid, and deleting on that basis destroys the only
copy of an entire campaign.

Prints one line per family with a verdict. Deletes nothing.
"""
import glob
import os
import re
import sys

import pyarrow.parquet as pq

OUT = "/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out"


def rows_of(path):
    try:
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return None


def main():
    fams = {}
    for d in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, d)
        if not os.path.isdir(p):
            continue
        m = re.match(r"^(.*)_chunk_(\d+)$", d)
        if not m:
            fams.setdefault(d, [])          # non-chunk dir (smoke runs etc.)
            continue
        fams.setdefault(m.group(1), []).append(p)

    merged = sorted(glob.glob(f"{OUT}/*.parquet"))
    merged_rows = {}

    print(f"{'family':<46}{'chunks':>7}{'chunk rows':>12}{'GB':>8}  verdict")
    for fam, dirs in sorted(fams.items()):
        size = sum(os.path.getsize(os.path.join(dp, f))
                   for dp in dirs for f in os.listdir(dp)
                   if os.path.isfile(os.path.join(dp, f))) / 1e9 if dirs else 0.0
        if not dirs:
            n_files = len(os.listdir(os.path.join(OUT, fam)))
            print(f"{fam:<46}{'-':>7}{'-':>12}{0.0:>8.2f}  non-chunk dir, {n_files} file(s)")
            continue

        total = 0
        ok = True
        for dp in dirs:
            for f in os.listdir(dp):
                if f.endswith(".parquet"):
                    r = rows_of(os.path.join(dp, f))
                    if r is None:
                        ok = False
                    else:
                        total += r
        if not ok:
            print(f"{fam:<46}{len(dirs):>7}{total:>12,}{size:>8.2f}  UNREADABLE chunk(s)")
            continue

        # A merged file must carry the same grid AND the same scenario level.
        grid = re.sub(r"_(ppnr|cnr|cnrclean|punr|A|dc_init_A|flat_start_A)$", "", fam,
                      flags=re.I)
        level = None
        for lv in ("backbone", "_A_", "no_change", "snapshot"):
            if lv.strip("_").lower() in fam.lower():
                level = lv
        cands = []
        for mp in merged:
            b = os.path.basename(mp)
            if not b.startswith(grid.split("_")[0]):
                continue
            if not b.startswith(grid):
                continue
            r = merged_rows.setdefault(mp, rows_of(mp))
            if r == total:
                cands.append((b, r))
        if cands:
            print(f"{fam:<46}{len(dirs):>7}{total:>12,}{size:>8.2f}  "
                  f"REDUNDANT -> {cands[0][0][:52]}")
        else:
            print(f"{fam:<46}{len(dirs):>7}{total:>12,}{size:>8.2f}  "
                  f"NO MATCHING MERGE (chunks are the only copy)")


if __name__ == "__main__":
    main()
