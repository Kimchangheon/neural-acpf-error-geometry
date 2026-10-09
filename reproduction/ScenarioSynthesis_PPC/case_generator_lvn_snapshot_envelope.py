#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from case_generator_all_test_cases_pandapower_consider_ppc_branch_row import (
    _apply_model_a_cgmes_cleanup,
    _build_net_from_source,
    case_generation_pandapower,
    reconstruct_Y_pandapower_branchrows_direct_SI,
)


DEFAULT_LVN_CGMES = (
    "/Users/changhunkim/PycharmProjects/PIGNN-Attn-LS/"
    "CGMES_to_PandaPower_clean/LVN_PowerFactory_fixed.zip"
)
DEFAULT_SNAPSHOT_ROOT = (
    "/Users/changhunkim/PycharmProjects/PIGNN-Attn-LS/"
    "CGMES_to_PandaPower_clean/SNAPSHOTS"
)

_DESC_SUFFIX_RE = re.compile(r"\s*\(R0BDY[^)]*\)$")


def normalize_load_description(value: Any) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    return _DESC_SUFFIX_RE.sub("", text).strip()


def load_row_key(row: pd.Series) -> str:
    description_key = normalize_load_description(row.get("description", ""))
    if description_key:
        return description_key

    name = row.get("name", "")
    if name is not None and not pd.isna(name) and str(name).strip():
        return str(name).strip()

    return str(row.name)


def _snapshot_sort_key(path: Path):
    match = re.search(r"(\d+)(?=\.zip$|$)", path.name)
    numeric = int(match.group(1)) if match else -1
    return (str(path.parent), numeric, path.name)


def list_snapshot_sources(snapshot_root: str) -> List[Path]:
    root = Path(os.path.abspath(os.path.expanduser(str(snapshot_root))))
    if not root.is_dir():
        raise FileNotFoundError(f"Snapshot root does not exist: {root}")

    dirs = [
        p for p in root.iterdir()
        if p.is_dir() and p.name.startswith("SNAPSHOT_")
    ]
    zip_files = [
        p for p in root.rglob("*.zip")
        if p.is_file() and p.name.startswith("CIM_GridAssist_")
    ]

    sources = sorted(dirs, key=_snapshot_sort_key) + sorted(zip_files, key=_snapshot_sort_key)
    if not sources:
        raise FileNotFoundError(
            f"No SNAPSHOT_* directories or CIM_GridAssist_*.zip snapshots found under: {root}"
        )
    return sources


def list_snapshot_dirs(snapshot_root: str) -> List[Path]:
    # Backward-compatible alias for older helper scripts.
    return list_snapshot_sources(snapshot_root)


def _snapshot_case_config(
    snapshot_source: Path,
    *,
    cgmes_version: str,
    ignore_errors: bool,
) -> Dict[str, Any]:
    if snapshot_source.is_dir():
        cgmes_files = sorted(str(p) for p in snapshot_source.glob("*.xml"))
        if not cgmes_files:
            raise FileNotFoundError(f"No CGMES XML files found in snapshot: {snapshot_source}")
    else:
        cgmes_files = str(snapshot_source)

    return {
        "cgmes_files": cgmes_files,
        "case_name": snapshot_source.stem if snapshot_source.is_file() else snapshot_source.name,
        "converter_kwargs": {
            "cgmes_version": str(cgmes_version).strip(),
            "ignore_errors": bool(ignore_errors),
        },
    }


def _extract_snapshot_load_frame(
    snapshot_source: Path,
    *,
    cgmes_version: str,
    ignore_errors: bool,
) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net, _case_name = _build_net_from_source(
            _snapshot_case_config(
                snapshot_source,
                cgmes_version=cgmes_version,
                ignore_errors=ignore_errors,
            ),
            case_kwargs={},
        )

    if not hasattr(net, "load") or len(net.load) == 0:
        raise RuntimeError(f"No loads found in snapshot: {snapshot_source}")

    name_col = "name" if "name" in net.load.columns else None
    cols = ["description", "p_mw", "q_mvar"]
    if name_col:
        cols.insert(0, name_col)
    df = net.load[cols].copy()
    df["load_key"] = df.apply(load_row_key, axis=1)

    dup = df["load_key"].duplicated(keep=False)
    if dup.any():
        dup_keys = sorted(df.loc[dup, "load_key"].unique().tolist())
        raise ValueError(
            f"Snapshot {snapshot_source.name} has non-unique normalized load keys: "
            f"{dup_keys[:10]}"
        )

    return df[["load_key", "p_mw", "q_mvar"]].sort_values("load_key").reset_index(drop=True)


def build_snapshot_load_bounds(
    snapshot_root: str,
    *,
    cgmes_version: str = "2.4.15",
    ignore_errors: bool = True,
    progress_every: int = 10,
) -> pd.DataFrame:
    snapshot_sources = list_snapshot_sources(snapshot_root)

    ref_keys: Optional[List[str]] = None
    p_min = p_max = q_min = q_max = None

    for idx, snapshot_source in enumerate(snapshot_sources, start=1):
        df = _extract_snapshot_load_frame(
            snapshot_source,
            cgmes_version=cgmes_version,
            ignore_errors=ignore_errors,
        )

        keys = df["load_key"].tolist()
        if ref_keys is None:
            ref_keys = keys
            p_vals = df["p_mw"].to_numpy(dtype=float)
            q_vals = df["q_mvar"].to_numpy(dtype=float)
            p_min = p_vals.copy()
            p_max = p_vals.copy()
            q_min = q_vals.copy()
            q_max = q_vals.copy()
        else:
            if keys != ref_keys:
                missing = sorted(set(ref_keys) - set(keys))
                extra = sorted(set(keys) - set(ref_keys))
                raise ValueError(
                    f"Snapshot {snapshot_source.name} does not match reference load key set. "
                    f"missing={missing[:5]}, extra={extra[:5]}"
                )

            p_vals = df["p_mw"].to_numpy(dtype=float)
            q_vals = df["q_mvar"].to_numpy(dtype=float)

            p_min = np.minimum(p_min, p_vals)
            p_max = np.maximum(p_max, p_vals)
            q_min = np.minimum(q_min, q_vals)
            q_max = np.maximum(q_max, q_vals)

        if progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == len(snapshot_sources)):
            print(f"[INFO] Snapshot bounds: {idx}/{len(snapshot_sources)} processed ({snapshot_source.name})")

    bounds = pd.DataFrame(
        {
            "load_key": ref_keys,
            "p_min_mw": np.minimum(p_min, p_max),
            "p_max_mw": np.maximum(p_min, p_max),
            "q_min_mvar": np.minimum(q_min, q_max),
            "q_max_mvar": np.maximum(q_min, q_max),
        }
    ).sort_values("load_key").reset_index(drop=True)

    return bounds


def save_snapshot_load_bounds_npz(bounds: pd.DataFrame, path: str) -> None:
    path_obj = Path(os.path.abspath(os.path.expanduser(str(path))))
    if path_obj.parent:
        path_obj.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path_obj,
        load_key=bounds["load_key"].to_numpy(dtype=str),
        p_min_mw=bounds["p_min_mw"].to_numpy(dtype=float),
        p_max_mw=bounds["p_max_mw"].to_numpy(dtype=float),
        q_min_mvar=bounds["q_min_mvar"].to_numpy(dtype=float),
        q_max_mvar=bounds["q_max_mvar"].to_numpy(dtype=float),
    )


def load_snapshot_load_bounds_npz(path: str) -> pd.DataFrame:
    data = np.load(os.path.abspath(os.path.expanduser(str(path))), allow_pickle=False)
    return pd.DataFrame(
        {
            "load_key": data["load_key"].astype(str),
            "p_min_mw": data["p_min_mw"].astype(float),
            "p_max_mw": data["p_max_mw"].astype(float),
            "q_min_mvar": data["q_min_mvar"].astype(float),
            "q_max_mvar": data["q_max_mvar"].astype(float),
        }
    ).sort_values("load_key").reset_index(drop=True)


def get_or_build_snapshot_load_bounds(
    snapshot_root: str,
    *,
    cgmes_version: str = "2.4.15",
    ignore_errors: bool = True,
    cache_path: str = "",
    progress_every: int = 10,
) -> pd.DataFrame:
    cache_path = str(cache_path or "").strip()
    if cache_path:
        expanded = os.path.abspath(os.path.expanduser(cache_path))
        if os.path.exists(expanded):
            print(f"[INFO] Loading snapshot load bounds cache: {expanded}")
            return load_snapshot_load_bounds_npz(expanded)

    bounds = build_snapshot_load_bounds(
        snapshot_root,
        cgmes_version=cgmes_version,
        ignore_errors=ignore_errors,
        progress_every=progress_every,
    )

    if cache_path:
        expanded = os.path.abspath(os.path.expanduser(cache_path))
        save_snapshot_load_bounds_npz(bounds, expanded)
        print(f"[INFO] Wrote snapshot load bounds cache: {expanded}")

    return bounds


def prepare_lvn_base_net(
    base_cgmes_path: str,
    *,
    case_name: str = "LVN_snapshot_envelope",
    cgmes_version: str = "2.4.15",
    ignore_errors: bool = True,
    apply_model_a_cleanup: bool = True,
    base_sn_mva: Optional[float] = None,
):
    net, _loaded_name = _build_net_from_source(
        {
            "cgmes_files": str(base_cgmes_path),
            "case_name": str(case_name).strip() or "LVN_snapshot_envelope",
            "converter_kwargs": {
                "cgmes_version": str(cgmes_version).strip(),
                "ignore_errors": bool(ignore_errors),
            },
        },
        case_kwargs={},
    )

    if apply_model_a_cleanup:
        _apply_model_a_cgmes_cleanup(net)

    if base_sn_mva is not None and float(base_sn_mva) > 0:
        net.sn_mva = float(base_sn_mva)

    net.name = str(case_name).strip() or "LVN_snapshot_envelope"
    return net


def align_bounds_to_base_net(base_net, bounds: pd.DataFrame) -> Dict[str, np.ndarray]:
    if not hasattr(base_net, "load") or len(base_net.load) == 0:
        raise RuntimeError("Base LVN net has no loads.")

    name_col = "name" if "name" in base_net.load.columns else None
    cols = ["description", "p_mw", "q_mvar"]
    if name_col:
        cols.insert(0, name_col)
    df = base_net.load[cols].copy()
    df["load_key"] = df.apply(load_row_key, axis=1)

    dup = df["load_key"].duplicated(keep=False)
    if dup.any():
        dup_keys = sorted(df.loc[dup, "load_key"].unique().tolist())
        raise ValueError(f"Base LVN net has non-unique normalized load keys: {dup_keys[:10]}")

    bounds_indexed = bounds.set_index("load_key")
    aligned = bounds_indexed.reindex(df["load_key"])

    missing = aligned.index[aligned["p_min_mw"].isna()].tolist()
    if missing:
        raise KeyError(
            f"Missing snapshot bounds for {len(missing)} base loads. "
            f"Examples: {missing[:10]}"
        )

    return {
        "load_key": df["load_key"].to_numpy(dtype=str),
        "p_min_mw": aligned["p_min_mw"].to_numpy(dtype=float),
        "p_max_mw": aligned["p_max_mw"].to_numpy(dtype=float),
        "q_min_mvar": aligned["q_min_mvar"].to_numpy(dtype=float),
        "q_max_mvar": aligned["q_max_mvar"].to_numpy(dtype=float),
    }


def _sample_loads_from_bounds(
    rng,
    *,
    p_min_mw: np.ndarray,
    p_max_mw: np.ndarray,
    q_min_mvar: np.ndarray,
    q_max_mvar: np.ndarray,
    sample_mode: str,
) -> Dict[str, np.ndarray]:
    sample_mode = str(sample_mode).strip().lower()
    if sample_mode not in {"coupled", "independent"}:
        raise ValueError(f"Unsupported sample_mode: {sample_mode}")

    if sample_mode == "coupled":
        alpha = rng.uniform(0.0, 1.0, size=p_min_mw.shape[0])
        p_mw = p_min_mw + alpha * (p_max_mw - p_min_mw)
        q_mvar = q_min_mvar + alpha * (q_max_mvar - q_min_mvar)
    else:
        p_mw = rng.uniform(p_min_mw, p_max_mw)
        q_mvar = rng.uniform(q_min_mvar, q_max_mvar)

    return {
        "p_mw": p_mw.astype(float, copy=False),
        "q_mvar": q_mvar.astype(float, copy=False),
    }


def case_generation_lvn_snapshot_envelope(
    *,
    base_net,
    aligned_bounds: Dict[str, np.ndarray],
    seed: Optional[int] = None,
    rng=None,
    case_name: str = "LVN_snapshot_envelope",
    sample_mode: str = "coupled",
    ybus_mode: str = "ppcY",
    start_mode: str = "dc_compile",
    v2_load_scale_range=None,
    v2_jitter_load: float = 0.0,
    v2_jitter_load_q: float = 0.0,
    v2_jitter_gen: float = 0.0,
    v2_perturb_sgen: bool = False,
    v2_pv_q_from_vstart: bool = False,
):
    if rng is None:
        rng = np.random.default_rng(seed)

    sampled = _sample_loads_from_bounds(
        rng,
        p_min_mw=np.asarray(aligned_bounds["p_min_mw"], dtype=float),
        p_max_mw=np.asarray(aligned_bounds["p_max_mw"], dtype=float),
        q_min_mvar=np.asarray(aligned_bounds["q_min_mvar"], dtype=float),
        q_max_mvar=np.asarray(aligned_bounds["q_max_mvar"], dtype=float),
        sample_mode=sample_mode,
    )

    sampled_net = copy.deepcopy(base_net)
    sampled_net.name = str(case_name).strip() or "LVN_snapshot_envelope"
    sampled_net.load.loc[:, "p_mw"] = sampled["p_mw"]
    sampled_net.load.loc[:, "q_mvar"] = sampled["q_mvar"]

    def _sampled_net_factory(**_kwargs):
        return sampled_net

    _sampled_net_factory.__name__ = sampled_net.name

    sample_seed = None if seed is None else int(seed)

    return case_generation_pandapower(
        case_fn=_sampled_net_factory,
        case_kwargs={},
        ybus_mode=str(ybus_mode).strip(),
        cgmes_model_a_cleanup=False,
        seed=sample_seed,
        jitter_load=float(v2_jitter_load),
        jitter_load_q=float(v2_jitter_load_q),
        jitter_gen=float(v2_jitter_gen),
        pv_vset_range=None,
        rand_u_start=False,
        angle_jitter_deg=0.0,
        mag_jitter_pq=0.0,
        trafo_pfe_kw=None,
        trafo_i0_percent=None,
        force_branch_shunt_pu=None,
        start_mode=str(start_mode).strip(),
        load_scale_range=v2_load_scale_range,
        perturb_sgen=bool(v2_perturb_sgen),
        pv_q_from_vstart=bool(v2_pv_q_from_vstart),
    )


if __name__ == "__main__":
    print("=== LVN snapshot branch-row direct-SI reconstruction check ===")

    base_net = prepare_lvn_base_net(
        DEFAULT_LVN_CGMES,
        case_name="LVN_snapshot_envelope",
        cgmes_version="2.4.15",
        ignore_errors=True,
        apply_model_a_cleanup=True,
    )
    bounds = get_or_build_snapshot_load_bounds(
        DEFAULT_SNAPSHOT_ROOT,
        cgmes_version="2.4.15",
        ignore_errors=True,
        cache_path="",
        progress_every=10,
    )
    aligned_bounds = align_bounds_to_base_net(base_net, bounds)

    (
        gridtype,
        bus_typ,
        _s_multi,
        _u_start,
        Y_matrix,
        _is_connected,
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
        _Branch_hv_is_f,
        _Branch_n,
        _Y_Lines,
        _Y_C_Lines,
        _U_base,
        _S_base,
        vn_kv,
    ) = case_generation_lvn_snapshot_envelope(
        base_net=base_net,
        aligned_bounds=aligned_bounds,
        seed=0,
        case_name="LVN_snapshot_envelope",
        sample_mode="coupled",
        ybus_mode="ppcY",
        start_mode="dc_compile",
    )

    Vbase_bus = np.asarray(vn_kv, dtype=np.float64) * 1e3
    Y_rec = reconstruct_Y_pandapower_branchrows_direct_SI(
        len(bus_typ),
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
        Vbase_bus=Vbase_bus,
    )

    diff = np.abs(Y_rec - Y_matrix)
    n_on = int(np.sum(Branch_status))
    n_tr = int(np.sum(Is_trafo))
    print(
        f"{'LVN_snapshot':16s} | "
        f"gridtype={gridtype:30s} | "
        f"N={len(bus_typ):5d} nl={len(Branch_f_bus):6d} "
        f"on={n_on:6d} trafo_like={n_tr:6d} | "
        f"max|diff|={np.max(diff):.12g} "
        f"mean|diff|={np.mean(diff):.12g}"
    )
    print("=== done ===")
