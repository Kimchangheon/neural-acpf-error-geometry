#!/usr/bin/env python3
"""
Build per-unit static-generator envelopes from the CGMES snapshot set.

`case_generator_lvn_snapshot_envelope.build_snapshot_load_bounds` scans the
snapshots for load bounds only, so every dataset built so far varies the loads
and holds generation fixed at whatever the base snapshot happened to contain.
For a power flow that is harmless. For an AC-OPF it is not: the SimBench base
export is a maximum-wind snapshot (3,716 MW), so every scenario ends up at the
same corner of the feasible set, curtailing ~97% of a constant availability,
and the OPF targets barely move between scenarios.

Across the full snapshot set the available renewable power actually spans
12 MW to 3,716 MW (94% relative standard deviation), so sampling it alongside
the loads restores a meaningful decision space.

This script mirrors the load-bounds builder for `net.sgen`, parallelised over
snapshots because the scan converts ~39k CGMES archives.

    python build_snapshot_sgen_bounds.py \
        --snapshot_root .../SimBenchSnapshots \
        --out .../SimBench_snapshot_sgen_bounds_cache.npz \
        --workers 72
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from case_generator_lvn_snapshot_envelope import (
    _build_net_from_source,
    _snapshot_case_config,
    list_snapshot_sources,
    normalize_load_description,
)

_CFG: Dict[str, Any] = {}


def sgen_row_key(row: pd.Series) -> str:
    """Stable identity for a static generator across snapshots."""
    for col in ("name", "description"):
        value = row.get(col, None)
        if value is not None and str(value).strip():
            return normalize_load_description(str(value))
    return f"bus{int(row.get('bus', -1))}"


def extract_snapshot_sgen_frame(snapshot_source, *, cgmes_version: str,
                                ignore_errors: bool) -> Optional[pd.DataFrame]:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net, _name = _build_net_from_source(
            _snapshot_case_config(snapshot_source, cgmes_version=cgmes_version,
                                  ignore_errors=ignore_errors),
            case_kwargs={},
        )
    if not hasattr(net, "sgen") or len(net.sgen) == 0:
        return None

    cols = [c for c in ("name", "description", "bus", "p_mw", "q_mvar") if c in net.sgen.columns]
    df = net.sgen[cols].copy()
    df["sgen_key"] = df.apply(sgen_row_key, axis=1)
    if df["sgen_key"].duplicated().any():
        df["sgen_key"] = df["sgen_key"] + "#" + df.groupby("sgen_key").cumcount().astype(str)
    return df[["sgen_key", "p_mw", "q_mvar"]].sort_values("sgen_key").reset_index(drop=True)


def _init_worker(cfg: Dict[str, Any]):
    global _CFG
    _CFG = cfg


def _scan_one(path_str: str):
    try:
        df = extract_snapshot_sgen_frame(
            Path(path_str),
            cgmes_version=_CFG["cgmes_version"],
            ignore_errors=_CFG["ignore_errors"],
        )
    except Exception:
        return None
    if df is None:
        return None
    return (df["sgen_key"].tolist(),
            df["p_mw"].to_numpy(dtype=float),
            df["q_mvar"].to_numpy(dtype=float))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot_root", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--cgmes_version", type=str, default="2.4.15")
    p.add_argument("--ignore_errors", action="store_true", default=True)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="Scan only the first N snapshots (debug).")
    p.add_argument("--stride", type=int, default=1, help="Scan every Nth snapshot.")
    return p.parse_args()


def main():
    import multiprocessing as mp

    args = parse_args()
    sources = list_snapshot_sources(args.snapshot_root)
    if args.stride > 1:
        sources = sources[:: int(args.stride)]
    if args.limit:
        sources = sources[: int(args.limit)]
    print(f"[sgen-bounds] scanning {len(sources)} snapshots from {args.snapshot_root}")

    cfg = {"cgmes_version": args.cgmes_version, "ignore_errors": bool(args.ignore_errors)}
    workers = int(args.workers) or max(1, (os.cpu_count() or 2) - 1)

    ref_keys: Optional[List[str]] = None
    p_min = p_max = q_min = q_max = None
    ok = skipped = 0
    t0 = time.time()

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for i, res in enumerate(pool.imap_unordered(_scan_one, [str(s) for s in sources], chunksize=4), 1):
            if res is None:
                skipped += 1
            else:
                keys, p_vals, q_vals = res
                if ref_keys is None:
                    ref_keys = keys
                    p_min, p_max = p_vals.copy(), p_vals.copy()
                    q_min, q_max = q_vals.copy(), q_vals.copy()
                elif keys == ref_keys:
                    p_min = np.minimum(p_min, p_vals); p_max = np.maximum(p_max, p_vals)
                    q_min = np.minimum(q_min, q_vals); q_max = np.maximum(q_max, q_vals)
                else:
                    skipped += 1
                    continue
                ok += 1
            if i % 2000 == 0 or i == len(sources):
                print(f"[sgen-bounds] {i}/{len(sources)} ok={ok} skipped={skipped} "
                      f"{time.time()-t0:.0f}s", flush=True)

    if ref_keys is None:
        raise SystemExit("No snapshot yielded static generators.")

    bounds = pd.DataFrame({
        "sgen_key": ref_keys,
        "p_min_mw": np.minimum(p_min, p_max),
        "p_max_mw": np.maximum(p_min, p_max),
        "q_min_mvar": np.minimum(q_min, q_max),
        "q_max_mvar": np.maximum(q_min, q_max),
    }).sort_values("sgen_key").reset_index(drop=True)

    out = Path(os.path.abspath(os.path.expanduser(args.out)))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        sgen_key=bounds["sgen_key"].to_numpy(dtype=str),
        p_min_mw=bounds["p_min_mw"].to_numpy(dtype=float),
        p_max_mw=bounds["p_max_mw"].to_numpy(dtype=float),
        q_min_mvar=bounds["q_min_mvar"].to_numpy(dtype=float),
        q_max_mvar=bounds["q_max_mvar"].to_numpy(dtype=float),
    )
    print(f"[sgen-bounds] wrote {out} ({len(bounds)} units, {ok} snapshots, {skipped} skipped)")
    print(f"[sgen-bounds] total available P: min={bounds['p_min_mw'].sum():.1f} "
          f"max={bounds['p_max_mw'].sum():.1f} MW")


def load_snapshot_sgen_bounds_npz(path: str) -> pd.DataFrame:
    data = np.load(os.path.abspath(os.path.expanduser(str(path))), allow_pickle=False)
    return pd.DataFrame({
        "sgen_key": data["sgen_key"].astype(str),
        "p_min_mw": data["p_min_mw"].astype(float),
        "p_max_mw": data["p_max_mw"].astype(float),
        "q_min_mvar": data["q_min_mvar"].astype(float),
        "q_max_mvar": data["q_max_mvar"].astype(float),
    }).sort_values("sgen_key").reset_index(drop=True)


def align_sgen_bounds_to_net(net, bounds: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Align an sgen envelope onto a base net's sgen table order."""
    if not hasattr(net, "sgen") or len(net.sgen) == 0:
        raise RuntimeError("Base net has no static generators.")
    df = net.sgen.copy()
    df["sgen_key"] = df.apply(sgen_row_key, axis=1)
    if df["sgen_key"].duplicated().any():
        df["sgen_key"] = df["sgen_key"] + "#" + df.groupby("sgen_key").cumcount().astype(str)
    merged = df[["sgen_key"]].merge(bounds, on="sgen_key", how="left")
    missing = int(merged["p_min_mw"].isna().sum())
    if missing:
        # Units absent from the envelope keep their base-net value as a fixed point.
        base_p = np.asarray(net.sgen["p_mw"], dtype=float)
        base_q = np.asarray(net.sgen.get("q_mvar", pd.Series(np.zeros(len(net.sgen)))), dtype=float)
        for col, fallback in (("p_min_mw", base_p), ("p_max_mw", base_p),
                              ("q_min_mvar", base_q), ("q_max_mvar", base_q)):
            merged[col] = merged[col].fillna(pd.Series(fallback, index=merged.index))
        print(f"[sgen-bounds] {missing} sgens not found in envelope; pinned to base-net values")
    return {
        "p_min_mw": merged["p_min_mw"].to_numpy(dtype=float),
        "p_max_mw": merged["p_max_mw"].to_numpy(dtype=float),
        "q_min_mvar": merged["q_min_mvar"].to_numpy(dtype=float),
        "q_max_mvar": merged["q_max_mvar"].to_numpy(dtype=float),
        "n_missing": np.array([missing]),
    }


if __name__ == "__main__":
    main()
