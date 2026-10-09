#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from case_generator_lvn_snapshot_envelope import (
    align_bounds_to_base_net,
    get_or_build_snapshot_load_bounds,
    list_snapshot_sources,
    prepare_lvn_base_net,
    save_snapshot_load_bounds_npz,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Explore load P/Q envelopes from many CGMES snapshots."
    )
    parser.add_argument("--snapshot_root", required=True)
    parser.add_argument("--base_cgmes_path", default="")
    parser.add_argument("--case_name", default="CGMES_snapshot_envelope")
    parser.add_argument("--cgmes_version", default="2.4.15")
    parser.add_argument(
        "--base_sn_mva",
        type=float,
        default=0.0,
        help="Override pandapower net.sn_mva for the base-network summary. Use <=0 to keep CGMES import default.",
    )
    parser.add_argument("--cgmes_ignore_errors", action="store_true")
    parser.add_argument("--no_cgmes_ignore_errors", dest="cgmes_ignore_errors", action="store_false")
    parser.set_defaults(cgmes_ignore_errors=True)
    parser.add_argument("--cgmes_model_a_cleanup", action="store_true")
    parser.add_argument("--bounds_cache_path", default="")
    parser.add_argument(
        "--fast_xml_bounds",
        action="store_true",
        help="Build bounds by directly parsing EQ/SSH EnergyConsumer p/q from CGMES zip snapshots.",
    )
    parser.add_argument("--progress_every", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=20)
    return parser.parse_args()


def _rdf_id(value: str) -> str:
    text = str(value or "")
    if text.startswith("#"):
        return text[1:]
    return text.rsplit("#", 1)[-1]


def _first_member_name(zf: zipfile.ZipFile, pattern: str) -> str:
    matches = [name for name in zf.namelist() if pattern in Path(name).name and name.lower().endswith(".xml")]
    if not matches:
        raise FileNotFoundError(f"No {pattern} XML member found in {zf.filename}")
    return sorted(matches)[0]


def _parse_energyconsumer_names_from_eq(eq_bytes: bytes) -> dict:
    ns = {
        "cim": "http://iec.ch/TC57/2013/CIM-schema-cim16#",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    }
    root = ET.fromstring(eq_bytes)
    out = {}
    load_tags = {"EnergyConsumer", "ConformLoad", "NonConformLoad"}
    for elem in root.iter():
        local_tag = elem.tag.rsplit("}", 1)[-1]
        if local_tag not in load_tags:
            continue
        rdf_about = (
            elem.attrib.get(f"{{{ns['rdf']}}}about", "")
            or elem.attrib.get(f"{{{ns['rdf']}}}ID", "")
        )
        name_elem = elem.find("cim:IdentifiedObject.name", ns)
        if rdf_about and name_elem is not None and name_elem.text:
            out[_rdf_id(rdf_about)] = str(name_elem.text).strip()
    return out


def _parse_energyconsumer_pq_from_ssh(ssh_bytes: bytes) -> dict:
    ns = {
        "cim": "http://iec.ch/TC57/2013/CIM-schema-cim16#",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    }
    root = ET.fromstring(ssh_bytes)
    out = {}
    for elem in root.iter():
        rdf_about = (
            elem.attrib.get(f"{{{ns['rdf']}}}about", "")
            or elem.attrib.get(f"{{{ns['rdf']}}}ID", "")
        )
        p_elem = elem.find("cim:EnergyConsumer.p", ns)
        q_elem = elem.find("cim:EnergyConsumer.q", ns)
        if not rdf_about or p_elem is None or q_elem is None:
            continue
        out[_rdf_id(rdf_about)] = (float(p_elem.text), float(q_elem.text))
    return out


def _read_eq_ssh_from_snapshot_zip(path: Path):
    with zipfile.ZipFile(path) as zf:
        eq_name = _first_member_name(zf, "__EQ_")
        ssh_name = _first_member_name(zf, "__SSH_")
        return zf.read(eq_name), zf.read(ssh_name)


def build_snapshot_load_bounds_fast_xml(snapshot_sources, *, progress_every: int = 500) -> pd.DataFrame:
    if not snapshot_sources:
        raise ValueError("No snapshot sources provided.")
    if any(not Path(src).is_file() for src in snapshot_sources):
        raise ValueError("--fast_xml_bounds expects snapshot sources to be CGMES zip files.")

    eq_bytes, ssh_bytes = _read_eq_ssh_from_snapshot_zip(Path(snapshot_sources[0]))
    id_to_name = _parse_energyconsumer_names_from_eq(eq_bytes)
    first_pq = _parse_energyconsumer_pq_from_ssh(ssh_bytes)

    ids = [eid for eid in id_to_name.keys() if eid in first_pq]
    if not ids:
        raise RuntimeError("No EnergyConsumer p/q entries could be matched between EQ and SSH.")

    p_vals = np.asarray([first_pq[eid][0] for eid in ids], dtype=float)
    q_vals = np.asarray([first_pq[eid][1] for eid in ids], dtype=float)
    p_min = p_vals.copy()
    p_max = p_vals.copy()
    q_min = q_vals.copy()
    q_max = q_vals.copy()

    missing_snapshots = 0
    for idx, source in enumerate(snapshot_sources[1:], start=2):
        _eq_unused, ssh_bytes = _read_eq_ssh_from_snapshot_zip(Path(source))
        pq = _parse_energyconsumer_pq_from_ssh(ssh_bytes)

        missing = [eid for eid in ids if eid not in pq]
        if missing:
            missing_snapshots += 1
            if missing_snapshots <= 3:
                print(f"[WARN] {Path(source).name} missing {len(missing)} EnergyConsumer ids; examples={missing[:5]}")
            continue

        p_vals = np.asarray([pq[eid][0] for eid in ids], dtype=float)
        q_vals = np.asarray([pq[eid][1] for eid in ids], dtype=float)
        p_min = np.minimum(p_min, p_vals)
        p_max = np.maximum(p_max, p_vals)
        q_min = np.minimum(q_min, q_vals)
        q_max = np.maximum(q_max, q_vals)

        if progress_every > 0 and (idx == 2 or idx % progress_every == 0 or idx == len(snapshot_sources)):
            print(f"[INFO] Fast XML bounds: {idx:,}/{len(snapshot_sources):,} processed ({Path(source).name})")

    if missing_snapshots:
        print(f"[WARN] Skipped/ignored {missing_snapshots:,} snapshots with missing EnergyConsumer ids.")

    return pd.DataFrame(
        {
            "load_key": [id_to_name[eid] for eid in ids],
            "p_min_mw": np.minimum(p_min, p_max),
            "p_max_mw": np.maximum(p_min, p_max),
            "q_min_mvar": np.minimum(q_min, q_max),
            "q_max_mvar": np.maximum(q_min, q_max),
        }
    ).sort_values("load_key").reset_index(drop=True)


def describe_vector(name: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        print(f"{name:24s}: empty")
        return

    qs = np.nanpercentile(values, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    print(
        f"{name:24s}: min={qs[0]: .6g} p01={qs[1]: .6g} p05={qs[2]: .6g} "
        f"p25={qs[3]: .6g} med={qs[4]: .6g} p75={qs[5]: .6g} "
        f"p95={qs[6]: .6g} p99={qs[7]: .6g} max={qs[8]: .6g}"
    )


def main():
    args = parse_args()

    snapshot_sources = list_snapshot_sources(args.snapshot_root)
    print("[INFO] Snapshot source summary")
    print(f"  root        = {Path(args.snapshot_root).expanduser().resolve()}")
    print(f"  n_snapshots = {len(snapshot_sources):,}")
    print(f"  first       = {snapshot_sources[0]}")
    print(f"  last        = {snapshot_sources[-1]}")

    base_cgmes_path = args.base_cgmes_path.strip() or str(snapshot_sources[0])
    print("[INFO] Loading base network")
    print(f"  base_cgmes_path       = {base_cgmes_path}")
    print(f"  cgmes_model_a_cleanup = {args.cgmes_model_a_cleanup}")
    print(f"  base_sn_mva           = {args.base_sn_mva if args.base_sn_mva > 0 else 'CGMES default'}")
    base_net = prepare_lvn_base_net(
        base_cgmes_path,
        case_name=args.case_name,
        cgmes_version=args.cgmes_version,
        ignore_errors=bool(args.cgmes_ignore_errors),
        apply_model_a_cleanup=bool(args.cgmes_model_a_cleanup),
        base_sn_mva=float(args.base_sn_mva) if float(args.base_sn_mva) > 0 else None,
    )
    print(
        "  net size              = "
        f"bus={len(base_net.bus):,}, load={len(base_net.load):,}, "
        f"line={len(base_net.line):,}, trafo={len(base_net.trafo):,}, "
        f"trafo3w={len(base_net.trafo3w):,}, switch={len(base_net.switch):,}, "
        f"gen={len(base_net.gen):,}, sgen={len(base_net.sgen):,}, ext_grid={len(base_net.ext_grid):,}"
    )

    print("[INFO] Building/loading load min/max bounds")
    cache_path = str(args.bounds_cache_path or "").strip()
    if args.fast_xml_bounds and cache_path and os.path.exists(os.path.expanduser(cache_path)):
        bounds = get_or_build_snapshot_load_bounds(
            args.snapshot_root,
            cgmes_version=args.cgmes_version,
            ignore_errors=bool(args.cgmes_ignore_errors),
            cache_path=args.bounds_cache_path,
            progress_every=args.progress_every,
        )
    elif args.fast_xml_bounds:
        bounds = build_snapshot_load_bounds_fast_xml(
            snapshot_sources,
            progress_every=args.progress_every,
        )
        if cache_path:
            save_snapshot_load_bounds_npz(bounds, cache_path)
            print(f"[INFO] Wrote snapshot load bounds cache: {Path(cache_path).expanduser().resolve()}")
    else:
        bounds = get_or_build_snapshot_load_bounds(
            args.snapshot_root,
            cgmes_version=args.cgmes_version,
            ignore_errors=bool(args.cgmes_ignore_errors),
            cache_path=args.bounds_cache_path,
            progress_every=args.progress_every,
        )
    aligned = align_bounds_to_base_net(base_net, bounds)

    p_min = np.asarray(aligned["p_min_mw"], dtype=float)
    p_max = np.asarray(aligned["p_max_mw"], dtype=float)
    q_min = np.asarray(aligned["q_min_mvar"], dtype=float)
    q_max = np.asarray(aligned["q_max_mvar"], dtype=float)
    p_width = p_max - p_min
    q_width = q_max - q_min

    print("[INFO] Load envelope summary")
    print(f"  n_loads               = {len(p_min):,}")
    print(f"  zero_width_p_loads    = {int(np.sum(np.isclose(p_width, 0.0))):,}")
    print(f"  zero_width_q_loads    = {int(np.sum(np.isclose(q_width, 0.0))):,}")
    print(f"  negative_p_min_loads  = {int(np.sum(p_min < 0.0)):,}")
    print(f"  negative_q_min_loads  = {int(np.sum(q_min < 0.0)):,}")
    print(f"  positive_q_max_loads  = {int(np.sum(q_max > 0.0)):,}")

    describe_vector("p_min_mw", p_min)
    describe_vector("p_max_mw", p_max)
    describe_vector("p_width_mw", p_width)
    describe_vector("q_min_mvar", q_min)
    describe_vector("q_max_mvar", q_max)
    describe_vector("q_width_mvar", q_width)
    describe_vector("abs_q_over_abs_p_max", np.abs(q_max) / np.maximum(np.abs(p_max), 1e-12))

    print("[INFO] Aggregate envelope, if all loads move to same side")
    print(f"  sum_p_min_mw          = {float(np.sum(p_min)):.12g}")
    print(f"  sum_p_max_mw          = {float(np.sum(p_max)):.12g}")
    print(f"  sum_q_min_mvar        = {float(np.sum(q_min)):.12g}")
    print(f"  sum_q_max_mvar        = {float(np.sum(q_max)):.12g}")

    top_k = max(int(args.top_k), 0)
    if top_k:
        score = np.hypot(p_width, q_width)
        order = np.argsort(score)[::-1][:top_k]
        print(f"[INFO] Top {len(order)} widest load envelopes")
        for rank, idx in enumerate(order, start=1):
            print(
                f"  {rank:02d} idx={idx:5d} key={aligned['load_key'][idx]} "
                f"P=[{p_min[idx]:.8g}, {p_max[idx]:.8g}] MW "
                f"Q=[{q_min[idx]:.8g}, {q_max[idx]:.8g}] Mvar "
                f"width_norm={score[idx]:.8g}"
            )


if __name__ == "__main__":
    main()
