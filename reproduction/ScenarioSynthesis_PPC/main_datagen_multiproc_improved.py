#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import io
import sys
import time
import math
import argparse
import traceback
import multiprocessing as mp
from typing import Any, Dict, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandapower.networks as pn

import case_generator_all_test_cases_pandapower_consider_ppc_branch_row as CGmod
from case_generator_all_test_cases_pandapower_consider_ppc_branch_row import (
    case_generation_pandapower,
    case_generation_pandapower_stamped,
)
from newton_raphson_improved import _compute_mismatch_inf, newtonrapson


# ============================================================
# Supported pandapower presets
# ============================================================

CASES = [
    "case4gs",
    "case5",
    "case6ww",
    "case9",
    "case14",
    "case24_ieee_rts",
    "case30",
    "case_ieee30",
    "case33bw",
    "case39",
    "case57",
    "case89pegase",
    "case118",
    "case145",
    "case_illinois200",
    "case300",
    "case1354pegase",
    "case1888rte",
    "case2848rte",
    "case2869pegase",
    "case3120sp",
    "case6470rte",
    "case6495rte",
    "case6515rte",
    "case9241pegase",
    "GBnetwork",
    "GBreducednetwork",
    "iceland",
]

FORCE_SHUNT_CASES = {
    "case4gs", "case5", "case6ww", "case9", "case30", "case33bw"
}

CASE_KWARGS = {
    # "case1888rte": {"ref_bus_idx": 1246},
    # "case2848rte": {"ref_bus_idx": 271},
    # "case6470rte": {"ref_bus_idx": 5988},
    # "case6495rte": {"ref_bus_idx": None},
    # "case6515rte": {"ref_bus_idx": 6171},
}


def _case_label_from_source(preset: str, cgmes_path: str, case_name: str) -> str:
    if preset:
        return str(preset).strip()

    if case_name:
        return str(case_name).strip()

    path = os.path.abspath(os.path.expanduser(str(cgmes_path).strip()))
    if not path:
        return "cgmes_case"

    if os.path.isdir(path):
        label = os.path.basename(path.rstrip(os.sep))
    else:
        label = os.path.splitext(os.path.basename(path))[0]

    return label or "cgmes_case"

SCENARIO_PRESETS = {
    "easy": dict(
        load_scale_range=(0.90, 1.10),   # ±10% global — proportionate to 2% per-bus noise
        scale_gen_with_load=True,
        jitter_load=0.02,
        jitter_load_q=0.03,
        jitter_gen=0.01,
        pv_vset_range=(0.998, 1.002),    # ±0.2%, very tight for easy scenario
        rand_u_start=False,
        angle_jitter_deg=0.5,
        mag_jitter_pq=0.002,
        line_outage_prob=0.0,
    ),
    # Phase-0 backbone: broad load/operating-point coverage, FIXED Y-bus.
    # B-level load knobs (best balance of coverage vs. NR convergence) with
    # topology/admittance OFF, so datasets stay share_grid-compatible and run
    # on the largest grids. Pair with --drop_nonconverged for clean labels.
    "backbone": dict(
        load_scale_range=(0.60, 1.40),   # ±40% — light → heavy → near-collapse
        scale_gen_with_load=True,
        jitter_load=0.10,                # σ_P
        jitter_load_q=0.15,              # σ_Q > σ_P (reactive less predictable)
        jitter_gen=0.05,                 # σ_G
        pv_vset_range=(0.95, 1.05),      # realistic generator setpoint band
        rand_u_start=True,
        angle_jitter_deg=5.0,
        mag_jitter_pq=0.02,
        line_outage_prob=0.0,            # Phase 0: NO topology change (Y-bus fixed)
    ),
    # ------------------------------------------------------------------
    # Held-out sets for testing whether a method depends on the artificially
    # low dimension of "backbone".
    #
    # "backbone" moves every load by one global factor, so the spatial shape of
    # the loading never changes.  Measured consequence on GBnetwork: 8 principal
    # directions capture 98.4% of the |V| variation over 12,339 scenarios and
    # 2224 buses.  Any method that exploits that low dimension -- manifold
    # projection above all -- is being scored on a corpus that guarantees it.
    # These presets keep the backbone load band and every other knob identical
    # and change only what is under test, so a model trained on backbone can be
    # scored on them directly.
    #
    #   ood_zone : shape diversity.  Independent lognormal multipliers on
    #              connected zones of the grid, mean-corrected so total load is
    #              unchanged -- a tilt, not a level change.
    #   ood_n1   : topology diversity.  Single-branch outages, so the Y-bus
    #              itself moves.  NOT share_grid-compatible; scoring it needs
    #              share_grid=False.
    # ------------------------------------------------------------------
    "ood_zone": dict(
        load_scale_range=(0.60, 1.40),
        scale_gen_with_load=True,
        jitter_load=0.10,
        jitter_load_q=0.15,
        jitter_gen=0.05,
        pv_vset_range=(0.95, 1.05),
        rand_u_start=True,
        angle_jitter_deg=5.0,
        mag_jitter_pq=0.02,
        line_outage_prob=0.0,
        load_zone_sigma=0.25,            # ~+-28% one-sigma tilt between zones
        # 14 to match the number of GB GSP groups, which is also the native
        # dimension of the measured tilt spectrum: at 8 the 13 GB eigenvalues
        # had to be interpolated down to 7, distorting the structure the
        # profile exists to reproduce.
        load_zone_count=14,
    ),
    "ood_n1": dict(
        load_scale_range=(0.60, 1.40),
        scale_gen_with_load=True,
        jitter_load=0.10,
        jitter_load_q=0.15,
        jitter_gen=0.05,
        pv_vset_range=(0.95, 1.05),
        rand_u_start=True,
        angle_jitter_deg=5.0,
        mag_jitter_pq=0.02,
        line_outage_prob=0.0,
        contingency_mode="ratio",        # P(N-0,N-1,N-2) = 0.55/0.27/0.18
        contingency_elements=("line",),
        contingency_k_cap=2,
    ),
    "no_change": dict(
        load_scale_range=None,
        scale_gen_with_load=True,
        jitter_load=0.0,
        jitter_load_q=0.0,
        jitter_gen=0.0,
        pv_vset_range=(1.0, 1.0),
        rand_u_start=False,
        angle_jitter_deg=0.0,
        mag_jitter_pq=0.0,
        line_outage_prob=0.0,
    ),
    "A": dict(
        load_scale_range=(0.70, 1.30),   # ±30% — typical day/night load swing
        scale_gen_with_load=True,
        jitter_load=0.05,
        jitter_load_q=0.08,
        jitter_gen=0.03,
        pv_vset_range=(0.97, 1.03),      # symmetric ±3% around 1.0
        rand_u_start=True,
        angle_jitter_deg=3.0,
        mag_jitter_pq=0.01,
        line_outage_prob=0.0,
    ),
    "B": dict(
        load_scale_range=(0.60, 1.40),   # ±40% — seasonal load variation
        scale_gen_with_load=True,
        jitter_load=0.10,
        jitter_load_q=0.15,
        jitter_gen=0.05,
        pv_vset_range=(0.95, 1.05),      # symmetric ±5% — practical operating range
        rand_u_start=True,
        angle_jitter_deg=5.0,
        mag_jitter_pq=0.02,
        line_outage_prob=0.01,
    ),
    "C": dict(
        load_scale_range=(0.50, 1.50),   # ±50% — stress testing, near-collapse coverage
        scale_gen_with_load=True,
        jitter_load=0.15,
        jitter_load_q=0.15,              # capped at 0.15 — 22% was unrealistically high
        jitter_gen=0.08,
        pv_vset_range=(0.93, 1.07),      # symmetric ±7% — pushed but still physical
        rand_u_start=True,
        angle_jitter_deg=7.0,
        mag_jitter_pq=0.03,
        line_outage_prob=0.02,
    ),
}


# ============================================================
# Helpers: .npy <-> bytes
# ============================================================

def ndarray_to_npy_bytes(x: Any) -> bytes:
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    buf = io.BytesIO()
    np.save(buf, x, allow_pickle=False)
    return buf.getvalue()


# ============================================================
# PU/SI conversion helpers for NR
# ============================================================

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


# ============================================================
# Parquet writer
# ============================================================

class ParquetAppendWriter:
    def __init__(
        self,
        path: str,
        compression: str = "zstd",
        overwrite: bool = True,
        save_y_matrix: bool = True,
        save_nr_trajectory: bool = False,
        row_group_size: int = 20,
    ):
        self.path = path
        self.save_y_matrix = bool(save_y_matrix)
        self.save_nr_trajectory = bool(save_nr_trajectory)
        # Row groups are the unit of random-access I/O. Left unset, pyarrow
        # writes one row group per flush, i.e. --save_steps rows, so a shuffled
        # loader reads thousands of rows to get one. Keep this small.
        self.row_group_size = int(row_group_size)

        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        if overwrite and os.path.exists(path):
            os.remove(path)

        fields = [
            # Stable attempt index.  Two corpora built with the same
            # corpus_seed can be joined on it, which is what allows a paired
            # comparison and a matched-intersection analysis.
            pa.field("scenario_id", pa.int64()),
            # The tilt as it was actually drawn, recorded at generation time.
            pa.field("tilt_clr_rms", pa.float64()),
            pa.field("zone_log_tilt", pa.binary()),
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
            # NR diagnostics — always present so the reader never has to handle missing columns
            pa.field("converged", pa.int8()),        # 1 = converged, 0 = did not converge
            pa.field("final_misinf", pa.float64()),  # inf-norm mismatch at last iteration (NaN if not diagnosed)
            pa.field("nr_iterations", pa.int32()),   # number of NR iterations executed
        ])

        if self.save_nr_trajectory:
            fields.extend([
                # Complex128 SI path [initial, NR step 1, ..., final].
                pa.field("u_nr_trajectory", pa.binary()),
                pa.field("nr_misinf_history", pa.binary()),
                pa.field("nr_step_history", pa.binary()),
                pa.field("nr_trajectory_length", pa.int32()),
                pa.field("nr_trajectory_converged", pa.int8()),
                pa.field("nr_trajectory_final_misinf", pa.float64()),
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
        for r in records:
            for k in cols.keys():
                cols[k].append(r[k])

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
        self._writer.write_table(table, row_group_size=self.row_group_size)

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None


# ============================================================
# Worker state
# ============================================================

_CFG: Dict[str, Any] = {}
_RNG = None
DBG = os.environ.get("GEN_DEBUG", "0") == "1"


def _init_worker(cfg: dict, seed_base: int):
    global _CFG, _RNG
    _CFG = cfg
    pid = os.getpid()
    ss = np.random.SeedSequence([seed_base & 0xFFFFFFFF, pid & 0xFFFFFFFF])
    _RNG = np.random.default_rng(ss)
    np.random.seed(int(ss.generate_state(1, dtype=np.uint32)[0]))


def _build_force_branch_shunt_from_cfg(preset: str, cfg: Dict[str, Any]) -> Optional[Dict[str, float]]:
    use_force_shunt_when_no_trafo = bool(cfg.get("use_force_shunt_when_no_trafo", False))
    g = float(cfg.get("force_branch_shunt_g_pu", 0.0))
    b = float(cfg.get("force_branch_shunt_b_pu", 0.0))
    g_asym = float(cfg.get("force_branch_shunt_g_asym_pu", 0.0))
    b_asym = float(cfg.get("force_branch_shunt_b_asym_pu", 0.0))

    anything_nonzero = any(abs(x) > 0 for x in [g, b, g_asym, b_asym])
    if anything_nonzero:
        return {"g": g, "b": b, "g_asym": g_asym, "b_asym": b_asym}

    if use_force_shunt_when_no_trafo and preset in FORCE_SHUNT_CASES:
        return {"g": 0.0, "b": 0.2, "g_asym": 0.0, "b_asym": 0.0}

    return None


def _generate_one_record_serialized(attempt_index=None,) -> Dict[str, Any]:
    global _CFG, _RNG

    K = int(_CFG["K"])
    preset = str(_CFG.get("preset", "")).strip()
    cgmes_path = str(_CFG.get("cgmes_path", "")).strip()
    case_name_override = str(_CFG.get("case_name", "")).strip()
    ybus_mode = str(_CFG["ybus_mode"]).strip()
    save_y_matrix = bool(_CFG["save_y_matrix"])
    pu_nr = bool(_CFG["pu_nr"])
    start_mode = str(_CFG["start_mode"]).strip()

    source_label = _case_label_from_source(preset, cgmes_path, case_name_override)

    if preset:
        case_fn = getattr(pn, preset, None)
        if case_fn is None:
            raise ValueError(f"Unknown pandapower case preset: {preset}")
        case_kwargs = CASE_KWARGS.get(preset, {}).copy()
    elif cgmes_path:
        case_fn = {
            "cgmes_files": cgmes_path,
            "case_name": source_label,
            "converter_kwargs": {
                "cgmes_version": str(_CFG["cgmes_version"]).strip(),
                "ignore_errors": bool(_CFG["cgmes_ignore_errors"]),
            },
        }
        case_kwargs = {}
    else:
        raise ValueError("Provide either --preset or --cgmes_path.")

    force_branch_shunt_pu = _build_force_branch_shunt_from_cfg(source_label, _CFG)
    # Derived from the attempt index, not from a time-seeded worker RNG, so two
    # corpora built with the same corpus_seed draw the SAME global scale, load
    # jitter, generator jitter and setpoint perturbation at the same attempt.
    # Only what differs by construction -- the zone latent transform -- differs.
    # Without this an IID-versus-correlated comparison is unpaired.
    if attempt_index is None:
        sample_seed = int(_RNG.integers(0, 2 ** 32 - 1, dtype=np.uint32))
    else:
        ss = np.random.SeedSequence(
            [int(_CFG.get("corpus_seed", 0)), int(attempt_index)])
        sample_seed = int(ss.generate_state(1, dtype=np.uint32)[0])

    gen_kwargs = dict(
        case_fn=case_fn,
        case_kwargs=case_kwargs,
        cgmes_model_a_cleanup=bool(_CFG.get("cgmes_model_a_cleanup", False)),
        base_sn_mva=_CFG.get("base_sn_mva", None),
        seed=sample_seed,
        jitter_load=float(_CFG["jitter_load"]),
        jitter_load_q=float(_CFG["jitter_load_q"]),
        jitter_gen=float(_CFG["jitter_gen"]),
        pv_vset_range=_CFG["pv_vset_range"],
        rand_u_start=bool(_CFG["rand_u_start"]),
        angle_jitter_deg=float(_CFG["angle_jitter_deg"]),
        mag_jitter_pq=float(_CFG["mag_jitter_pq"]),
        trafo_pfe_kw=_CFG["trafo_pfe_kw"],
        trafo_i0_percent=_CFG["trafo_i0_percent"],
        force_branch_shunt_pu=force_branch_shunt_pu,
        start_mode=start_mode,
        load_scale_range=_CFG["load_scale_range"],
        scale_gen_with_load=bool(_CFG["scale_gen_with_load"]),
        perturb_sgen=bool(_CFG.get("perturb_sgen", False)),
        pv_q_from_vstart=bool(_CFG.get("pv_q_from_vstart", False)),
        line_outage_prob=float(_CFG["line_outage_prob"]),
        # Held-out knobs.  .get with the generator's own defaults, so every
        # existing preset keeps producing byte-identical scenarios and only the
        # presets that set them are affected.
        load_zone_sigma=float(_CFG.get("load_zone_sigma", 0.0)),
        load_zone_count=int(_CFG.get("load_zone_count", 0)),
        load_zone_correlation=str(_CFG.get("load_zone_correlation", "none")),
        load_zone_structure_seed=int(_CFG.get("load_zone_structure_seed", 0)),
        zone_tilt_targets=str(_CFG.get("zone_tilt_targets", "load")),
        zone_tilt_normalization=str(
            _CFG.get("zone_tilt_normalization", "load_weighted")),
        zone_tilt_amplitude_mode=str(
            _CFG.get("zone_tilt_amplitude_mode", "sigma")),
        zone_tilt_orientation_seed=_CFG.get("zone_tilt_orientation_seed", None),
        contingency_mode=str(_CFG.get("contingency_mode", "none")),
        contingency_elements=tuple(_CFG.get("contingency_elements", ("line",))),
        contingency_k_cap=int(_CFG.get("contingency_k_cap", 2)),
        contingency_ratio_weights=tuple(
            _CFG.get("contingency_ratio_weights", (0.55, 0.27, 0.18))),
        return_pp_solution=(str(_CFG.get("label_solver", "custom_nr")) == "pandapower_nr"),
    )

    if ybus_mode.lower() == "stamped":
        out = case_generation_pandapower_stamped(**gen_kwargs)
    else:
        out = case_generation_pandapower(ybus_mode="ppcY", **gen_kwargs)

    base_out = out[:25]
    pp_label = out[25:] if len(out) > 25 else ()

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
    ) = base_out

    bus_number = int(len(bus_typ))
    branch_number = int(len(Branch_f_bus))

    ymat_bytes = ndarray_to_npy_bytes(np.asarray(Y_matrix, dtype=np.complex128).copy()) if save_y_matrix else None

    # NR diagnostic defaults (overwritten below if NR is actually run)
    nr_converged = 0
    nr_final_misinf = float("nan")
    nr_iterations = 0
    u_nr_trajectory_si = np.empty((0, bus_number), dtype=np.complex128)
    nr_misinf_history = np.empty((0,), dtype=np.float64)
    nr_step_history = np.empty((0,), dtype=np.float64)

    label_solver = str(_CFG.get("label_solver", "custom_nr"))

    if not is_connected:
        u_newton_si = np.zeros_like(u_start, dtype=np.complex128)
        S_newton_si = np.zeros_like(s_multi, dtype=np.complex128)
    elif label_solver == "pandapower_nr":
        if len(pp_label) != 3:
            raise RuntimeError("pandapower_nr requested, but case generator did not return pandapower labels")

        u_newton_si = np.asarray(pp_label[0], dtype=np.complex128)
        S_newton_si = np.asarray(pp_label[1], dtype=np.complex128)
        nr_converged = 1 if bool(pp_label[2]) else 0
        nr_iterations = 0

        try:
            bus_typ_arr = np.asarray(bus_typ, dtype=np.int64).copy()
            nr_final_misinf = _compute_mismatch_inf(
                bus_typ_arr,
                np.asarray(Y_matrix, dtype=np.complex128),
                u_newton_si,
                np.asarray(s_multi, dtype=np.complex128).real,
                np.asarray(s_multi, dtype=np.complex128).imag,
            )
        except Exception:
            nr_final_misinf = float("nan")
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
            return_trajectory=bool(_CFG.get("save_nr_trajectory", False)),
        )

        u_newton_raw, _I_unused, S_newton_raw, nr_diag = nr_out

        nr_converged    = 1 if bool(nr_diag.get("converged", False)) else 0
        nr_final_misinf = float(nr_diag["final_misinf"]) if nr_diag.get("final_misinf") is not None else float("nan")
        nr_iterations   = int(nr_diag.get("iterations", 0))

        u_nr_trajectory_raw = None
        if bool(_CFG.get("save_nr_trajectory", False)):
            u_nr_trajectory_raw = np.asarray(
                nr_diag.get("voltage_trajectory"), dtype=np.complex128
            )
            if u_nr_trajectory_raw.ndim != 2 or u_nr_trajectory_raw.shape[1] != bus_number:
                raise RuntimeError(
                    "NR trajectory must have shape [steps, buses], got "
                    f"{u_nr_trajectory_raw.shape} for {bus_number} buses"
                )
            nr_misinf_history = np.asarray(
                nr_diag.get("misinf_history", []), dtype=np.float64
            )
            nr_step_history = np.asarray(
                nr_diag.get("step_history", []), dtype=np.float64
            )

        if pu_nr:
            u_newton_arr = np.asarray(u_newton_raw)
            s_newton_arr = np.asarray(S_newton_raw)

            if u_newton_arr.size == 0:
                u_newton_si = np.asarray(u_newton_arr, dtype=np.complex128)
            else:
                u_newton_si = u_pu_to_si_per_bus(u_newton_arr, Vbase_bus)

            if u_nr_trajectory_raw is not None and u_nr_trajectory_raw.size:
                u_nr_trajectory_si = (
                    u_nr_trajectory_raw * Vbase_bus.reshape(1, -1)
                ).astype(np.complex128, copy=False)

            if s_newton_arr.size == 0:
                S_newton_si = np.asarray(s_newton_arr, dtype=np.complex128)
            else:
                S_newton_si = s_pu_to_si(s_newton_arr, float(S_base))
        else:
            u_newton_si = np.asarray(u_newton_raw, dtype=np.complex128)
            S_newton_si = np.asarray(S_newton_raw, dtype=np.complex128)
            if u_nr_trajectory_raw is not None:
                u_nr_trajectory_si = u_nr_trajectory_raw

    rec = {
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
        "converged":     nr_converged,
        "final_misinf":  nr_final_misinf,
        "nr_iterations": nr_iterations,
    }

    if save_y_matrix:
        rec["Y_matrix"] = ymat_bytes

    if bool(_CFG.get("save_nr_trajectory", False)):
        rec.update({
            "u_nr_trajectory": ndarray_to_npy_bytes(u_nr_trajectory_si),
            "nr_misinf_history": ndarray_to_npy_bytes(nr_misinf_history),
            "nr_step_history": ndarray_to_npy_bytes(nr_step_history),
            "nr_trajectory_length": int(u_nr_trajectory_si.shape[0]),
            "nr_trajectory_converged": int(nr_converged),
            "nr_trajectory_final_misinf": float(nr_final_misinf),
        })

    return rec


def _generate_batch(task) -> List[Dict[str, Any]]:
    start, n_rows = task if isinstance(task, tuple) else (None, task)
    drop_nc = bool(_CFG.get("drop_nonconverged", False))
    out: List[Dict[str, Any]] = []
    err_shown = 0

    for k in range(n_rows):
        ai = None if start is None else start + k
        try:
            rec = _generate_one_record_serialized(ai)
            if ai is not None:
                rec["scenario_id"] = int(ai)
            _t = getattr(CGmod, "_LAST_TILT", None) or {}
            _z = _t.get("zone_log_tilt")
            rec["tilt_clr_rms"] = float(_t.get("tilt_clr_rms", 0.0))
            rec["zone_log_tilt"] = (ndarray_to_npy_bytes(np.asarray(_z, dtype=np.float64))
                                    if _z is not None else None)
            if drop_nc and rec["converged"] == 0:
                continue
            out.append(rec)
        except Exception as e:
            if DBG and err_shown < 3:
                print("[WORKER ERROR]", repr(e))
                traceback.print_exc()
                err_shown += 1
            continue

    return out


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel dataset generation for pandapower and CGMES cases (branch-row direct-SI metadata)."
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--preset",
        type=str,
        choices=CASES,
        help="Pandapower test case name.",
    )
    source_group.add_argument(
        "--cgmes_path",
        type=str,
        default="",
        help="Path to a CGMES zip file or to a directory containing CGMES zip files.",
    )
    parser.add_argument(
        "--case_name",
        type=str,
        default="",
        help="Optional case label override, especially useful with --cgmes_path.",
    )
    parser.add_argument(
        "--cgmes_version",
        type=str,
        default="2.4.15",
        help="CGMES version passed to pandapower when --cgmes_path is used.",
    )
    cgmes_cleanup_group = parser.add_mutually_exclusive_group()
    cgmes_cleanup_group.add_argument(
        "--cgmes_model_a_cleanup",
        dest="cgmes_model_a_cleanup",
        action="store_true",
        help="Apply LVN-style CGMES cleanup before PPC compilation.",
    )
    cgmes_cleanup_group.add_argument(
        "--no_cgmes_model_a_cleanup",
        dest="cgmes_model_a_cleanup",
        action="store_false",
        help="Do not apply LVN-style CGMES cleanup (default).",
    )
    parser.set_defaults(cgmes_model_a_cleanup=False)
    parser.add_argument(
        "--base_sn_mva",
        type=float,
        default=None,
        help=(
            "Override net.sn_mva (per-unit base, MVA). Required for CGMES models "
            "whose native base leaves injections at thousands of per-unit; LVN heo1 "
            "ships with sn_mva=1 and must be run with --base_sn_mva 100."
        ),
    )
    cgmes_error_group = parser.add_mutually_exclusive_group()
    cgmes_error_group.add_argument(
        "--cgmes_ignore_errors",
        dest="cgmes_ignore_errors",
        action="store_true",
        help="Ignore CGMES converter errors when loading the source (default).",
    )
    cgmes_error_group.add_argument(
        "--no_cgmes_ignore_errors",
        dest="cgmes_ignore_errors",
        action="store_false",
        help="Do not ignore CGMES converter errors when loading the source.",
    )
    parser.set_defaults(cgmes_ignore_errors=True)
    parser.add_argument(
        "--ybus_mode",
        type=str,
        default="ppcY",
        choices=["ppcY", "stamped"],
        help="Use pandapower ppcY directly or rebuild Y_matrix by stamping PPC.",
    )

    parser.add_argument(
        "--jitter_load_q",
        type=float,
        default=None,
        help=(
            "Std-dev of the Gaussian multiplier applied independently to each load's q_mvar. "
            "If not set, falls back to the scenario preset value. "
            "Use this to override the preset's Q jitter without changing other knobs."
        ),
    )
    parser.add_argument(
        "--load_scale_lo",
        type=float,
        default=None,
        help="Lower bound of the uniform global load-scale factor (e.g. 0.7). "
             "Overrides the preset's load_scale_range lower bound.",
    )
    parser.add_argument(
        "--load_scale_hi",
        type=float,
        default=None,
        help="Upper bound of the uniform global load-scale factor (e.g. 1.3). "
             "Overrides the preset's load_scale_range upper bound.",
    )
    parser.add_argument(
        "--line_outage_prob",
        type=float,
        default=None,
        help="Per-line probability of N-1 outage. Overrides the preset value.",
    )
    parser.add_argument(
        "--drop_nonconverged",
        action="store_true",
        help="Skip (do not write) rows where NR did not converge. "
             "The total row count in the output file may be less than --runs.",
    )
    parser.add_argument("--K", type=int, default=40, help="Newton-Raphson max iterations")
    parser.add_argument(
        "--save_nr_trajectory",
        action="store_true",
        help=(
            "Save the custom Newton solver's initial voltage and every iterate "
            "as complex128 SI arrays. Requires --label_solver custom_nr."
        ),
    )
    parser.add_argument("--runs", type=int, default=10000, help="Total samples")
    parser.add_argument("--save_steps", type=int, default=2000, help="Rows per Parquet append")
    parser.add_argument("--rows_per_task", type=int, default=1000, help="Rows per worker batch")
    parser.add_argument("--workers", type=int, default=0, help="0 = all logical CPUs")
    parser.add_argument("--save_path", type=str, default="", help="Output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if exists")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--save_y_matrix", dest="save_y_matrix", action="store_true", help="Save Y_matrix (default)")
    group.add_argument("--no_save_y_matrix", dest="save_y_matrix", action="store_false", help="Do not save Y_matrix")
    parser.set_defaults(save_y_matrix=True)

    # Start-point jitter is OPT-IN.
    #
    # Independent per-bus angle noise corrupts branch angle differences, which
    # is what the NR basin actually depends on: +/-5 deg per bus injects branch
    # |dtheta| 2-4x larger than the physical DC solution and pushes the start
    # outside the basin. Measured effect on custom-NR convergence at +/-40%
    # load (case89pegase 3.3% jittered vs 100% clean; case_illinois200 0.0% vs
    # 100%). The scenario preset value is therefore NOT used unless the flag is
    # given explicitly, so a plain run always starts from the clean start point.
    jgroup = parser.add_mutually_exclusive_group()
    jgroup.add_argument(
        "--rand_u_start",
        dest="rand_u_start",
        action="store_true",
        help="Add jitter to the initial voltage (opt-in; overrides the preset).",
    )
    jgroup.add_argument(
        "--no_rand_u_start",
        dest="rand_u_start",
        action="store_false",
        help="Use the clean start point exactly as built by --start_mode (default).",
    )
    parser.set_defaults(rand_u_start=False)

    parser.add_argument(
        "--angle_jitter_deg",
        type=float,
        default=None,
        help="Override preset angle jitter [deg]. Only used with --rand_u_start.",
    )
    parser.add_argument(
        "--mag_jitter_pq",
        type=float,
        default=None,
        help="Override preset PQ magnitude jitter [fraction]. Only used with --rand_u_start.",
    )

    parser.add_argument(
        "--perturb_sgen",
        action="store_true",
        help="Apply the global load scale and the generator jitter to net.sgen "
             "as well. Off by default because the existing 31-file corpus was "
             "built without it. Matters most where net.gen is empty and all "
             "generation is static: on SimBench (0 gen, 121 sgen, 3,716 MW) "
             "leaving this off makes a +/-40%% load sweep move total injection "
             "by only 2.2%%.",
    )
    parser.add_argument(
        "--pv_q_from_vstart",
        action="store_true",
        help="Recompute the PV-bus reactive entry of S_start as "
             "Im[u_start * conj(Y u_start)]. Without it that entry is whatever "
             "the PPC compile back-computed at its own flat start voltage, "
             "which is not the u_start stored beside it, and the discrepancy "
             "varies per grid. Uses no solved quantity, so it is not leakage.",
    )

    parser.add_argument(
        "--contingency_ratio_weights",
        type=float,
        nargs=3,
        default=None,
        metavar=("P_N0", "P_N1", "P_N2"),
        help="Override the 'ratio' contingency mode's P(N-0, N-1, N-2). The "
             "ood_n1 preset mixes all three (0.55/0.27/0.18); pass '0 1 0' for "
             "a pure N-1 set or '0 0 1' for a pure N-2 set so the two outage "
             "orders can be attributed separately. Note the sampler falls back "
             "to N-0 when a draw would island the grid, so the realised mix "
             "must be measured, not assumed.",
    )

    parser.add_argument(
        "--zone_tilt_orientation_seed",
        type=int,
        default=None,
        help="Seeds only the covariance orientation and the assignment of "
             "marginal spreads to zones, leaving the partition on "
             "--load_zone_structure_seed. Our zones are not real GSP groups, so "
             "that assignment is arbitrary; varying this tests whether a result "
             "depends on one particular permutation.",
    )

    parser.add_argument(
        "--zone_tilt_amplitude_mode",
        type=str,
        default=None,
        choices=("sigma", "clr_rms"),
        help="What --load_zone_sigma means. 'sigma' (legacy) is the marginal "
             "latent sd, so the realised centred-log RMS depends on the "
             "covariance shape and an IID and a GB corpus at the same value "
             "are NOT the same strength. 'clr_rms' makes the value the target "
             "centred-log RMS itself, scaling the covariance by one global "
             "scalar, so two structures can be compared at equal amplitude.",
    )

    parser.add_argument(
        "--corpus_seed",
        type=int,
        default=0,
        help="Seeds the per-attempt scenario seeds. Two corpora sharing this "
             "value see identical global scale and jitter at each attempt "
             "index, which is what makes an IID-versus-correlated comparison "
             "paired rather than merely same-distribution.",
    )

    parser.add_argument(
        "--zone_tilt_normalization",
        type=str,
        default=None,
        choices=("load_weighted", "arithmetic"),
        help="How the zone multipliers are normalised. Default 'load_weighted' "
             "makes sum_z L_z m_z = sum_z L_z exactly, so the tilt changes the "
             "spatial distribution and not the load level. 'arithmetic' is the "
             "legacy mean-of-multipliers normalisation, which preserves total "
             "load only when all zones carry equal load.",
    )

    parser.add_argument(
        "--zone_tilt_targets",
        type=str,
        default=None,
        choices=("load", "load_and_gen"),
        help="What the spatial zone tilt multiplies. Default 'load': the tilt "
             "is calibrated on regional demand, which says nothing about the "
             "output of transmission-connected plant in that region. "
             "'load_and_gen' reproduces the earlier behaviour, in which the "
             "tilt also scaled generation and so partly cancelled itself.",
    )

    parser.add_argument(
        "--load_zone_correlation",
        type=str,
        default=None,
        choices=("none", "empirical", "empirical_gb", "empirical_isone",
                 "empirical_fr", "empirical_gb_diag",
                 "empirical_isone_diag", "empirical_fr_diag"),
        help="Zone-to-zone structure of the tilt. 'none' draws each zone "
             "independently. 'empirical' carries the correlation eigenvalue "
             "spectrum and per-zone spread inequality measured on EIA-930 "
             "real zonal demand. 'empirical' = 'empirical_gb', measured on "
             "Elexon GSP Group Take, the system this corpus is built on; "
             "'empirical_isone' and 'empirical_fr' are the other two measured "
             "systems, for sensitivity. Zone identities are not "
             "transferable, so the basis is random but held FIXED across the "
             "corpus, and the zone partition is fixed with it.",
    )

    parser.add_argument(
        "--load_zone_structure_seed",
        type=int,
        default=0,
        help="Seed for the fixed zone partition and correlation basis used by "
             "--load_zone_correlation empirical. Constant across scenarios by "
             "design; vary it to get a different but equally valid grid.",
    )

    parser.add_argument(
        "--load_zone_sigma",
        type=float,
        default=None,
        help="Override the preset's zone-tilt sigma. The zone multipliers are "
             "lognormal exp(N(0, sigma)) renormalised to mean 1, so this sets "
             "how strongly the spatial load pattern tilts without changing the "
             "total load. Used to sweep the tilt strength for OOD checks.",
    )

    parser.add_argument(
        "--scenario_level",
        type=str,
        default="no_change",
        choices=list(SCENARIO_PRESETS.keys()),
        help="Scenario preset for perturbations.",
    )

    parser.add_argument(
        "--use_force_shunt_when_no_trafo",
        action="store_true",
        help="For known no-trafo cases, apply default forced branch shunt (b=0.2 pu).",
    )
    parser.add_argument("--force_branch_shunt_g_pu", type=float, default=0.0)
    parser.add_argument("--force_branch_shunt_b_pu", type=float, default=0.0)
    parser.add_argument("--force_branch_shunt_g_asym_pu", type=float, default=0.0)
    parser.add_argument("--force_branch_shunt_b_asym_pu", type=float, default=0.0)

    parser.add_argument("--trafo_pfe_kw", type=float, default=None)
    parser.add_argument("--trafo_i0_percent", type=float, default=None)

    parser.add_argument("--pu_nr", action="store_true", help="Solve NR in per-unit, save results back in SI.")
    parser.add_argument(
        "--label_solver",
        type=str,
        default="custom_nr",
        choices=["custom_nr", "pandapower_nr"],
        help="Use the in-repo custom NR label solver or pandapower's converged NR solution as label.",
    )
    parser.add_argument("--diagnose_nr", action="store_true", help="Enable NR diagnostics.")
    parser.add_argument("--print_misinf", action="store_true", help="Print NR mismatch per iteration.")
    parser.add_argument("--near_misinf_tol", type=float, default=1e-3)
    parser.add_argument(
        "--start_mode",
        type=str,
        default="auto",
        choices=["auto", "manual_flat", "ppc_v0", "dc_compile"],
        help="Voltage start mode passed to case_generation_pandapower.",
    )

    parser.add_argument(
        "--convergence_mode",
        type=str,
        default="misinf",
        choices=["two_step", "misinf"],
        help="NR convergence criterion.",
    )
    parser.add_argument("--step_tol", type=float, default=5e-4)
    parser.add_argument("--mismatch_tol", type=float, default=1e-8)

    return parser.parse_args()


def resolve_start_jitter(args):
    """Effective (rand_u_start, angle_jitter_deg, mag_jitter_pq).

    rand_u_start is CLI-only and defaults to False, so the preset cannot switch
    jitter on implicitly. Magnitudes fall back to the preset when the flag is
    given without explicit overrides. Single source of truth for both the run
    config and the output filename.
    """
    on = bool(args.rand_u_start)
    if not on:
        return False, 0.0, 0.0

    cfg = SCENARIO_PRESETS[args.scenario_level]
    ang = args.angle_jitter_deg if args.angle_jitter_deg is not None else cfg["angle_jitter_deg"]
    mag = args.mag_jitter_pq if args.mag_jitter_pq is not None else cfg["mag_jitter_pq"]
    return True, float(ang), float(mag)


def build_output_filename(args) -> str:
    mode = str(args.ybus_mode).strip()
    nr_unit = "puNR" if args.pu_nr else "siNR"
    case_label = _case_label_from_source(
        preset=str(args.preset or "").strip(),
        cgmes_path=str(args.cgmes_path or "").strip(),
        case_name=str(args.case_name or "").strip(),
    )

    # Label solver and effective load-scale range must appear in the name:
    # runs that differ only in these two knobs otherwise collide on the same
    # path and silently overwrite each other under --overwrite.
    solver_tag = "ppNR" if str(args.label_solver) == "pandapower_nr" else "cNR"

    preset_range = SCENARIO_PRESETS[args.scenario_level]["load_scale_range"]
    if preset_range is None and args.load_scale_lo is None and args.load_scale_hi is None:
        load_tag = "lsNone"
    else:
        lo = args.load_scale_lo if args.load_scale_lo is not None else preset_range[0]
        hi = args.load_scale_hi if args.load_scale_hi is not None else preset_range[1]
        load_tag = f"ls{float(lo):.2f}-{float(hi):.2f}"

    # Start-point jitter also goes in the name: a jittered and a clean run are
    # different datasets and must not overwrite each other.
    jit_on, jit_ang, jit_mag = resolve_start_jitter(args)
    start_tag = f"u0jit{jit_ang:g}deg-{jit_mag:g}" if jit_on else "u0clean"

    # Both of these change the physics of the stored row, not just its
    # provenance, so they must appear in the name for the same reason the
    # solver and load-range fields do: a run with and a run without them are
    # different datasets and must not silently overwrite each other.
    extra = ""
    if args.perturb_sgen:
        extra += "_sgenpert"
    if args.pv_q_from_vstart:
        extra += "_pvqvstart"
    if args.save_nr_trajectory:
        extra += "_nrtraj"

    if args.load_zone_sigma is not None:
        extra += f"_zs{args.load_zone_sigma:g}"
    if args.load_zone_correlation not in (None, "none"):
        from perturbation_options import DEFAULT_TILT_PROFILE
        _prof = (DEFAULT_TILT_PROFILE if args.load_zone_correlation == "empirical"
                 else args.load_zone_correlation)
        extra += f"_zc{_prof}{args.load_zone_structure_seed:g}"
        # zone count changes the structure the profile is resampled onto, so it
        # belongs in the name: two corpora differing only in it are different
        # corpora, and were previously indistinguishable by filename.
        extra += f"_nz{int(SCENARIO_PRESETS[args.scenario_level].get('load_zone_count', 0))}"
        if SCENARIO_PRESETS[args.scenario_level].get("zone_tilt_targets",
                                                     "load") == "load_and_gen":
            extra += "_tgtlg"
        if SCENARIO_PRESETS[args.scenario_level].get(
                "zone_tilt_normalization", "load_weighted") == "arithmetic":
            extra += "_normarith"
        if SCENARIO_PRESETS[args.scenario_level].get(
                "zone_tilt_amplitude_mode", "sigma") == "clr_rms":
            extra += "_clr"
        _os = SCENARIO_PRESETS[args.scenario_level].get("zone_tilt_orientation_seed")
        if _os is not None:
            extra += f"_or{int(_os)}"
    if args.contingency_ratio_weights is not None:
        extra += "_cw" + "-".join(f"{x:g}" for x in args.contingency_ratio_weights)

    name = (
        f"{case_label}_{mode}_{args.scenario_level}_{args.start_mode}_"
        f"{solver_tag}_{load_tag}_{start_tag}{extra}_"
        f"{nr_unit}_{args.runs}_NR_branchrows_directSI.parquet"
    )
    return os.path.join(args.save_path, name)


def get_parquet_file_size(path: str) -> int:
    return os.path.getsize(path) if os.path.exists(path) else 0


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if args.save_nr_trajectory and args.label_solver != "custom_nr":
        raise ValueError(
            "--save_nr_trajectory currently requires --label_solver custom_nr. "
            "Use add_nr_trajectories_to_parquet.py to augment an existing "
            "pandapower-label parquet without changing its final labels."
        )

    if args.load_zone_sigma is not None:
        # Mutate the preset itself: several call sites re-read SCENARIO_PRESETS
        # independently, and they must all see the same tilt strength.
        SCENARIO_PRESETS[args.scenario_level]["load_zone_sigma"] = float(args.load_zone_sigma)
    if args.zone_tilt_orientation_seed is not None:
        SCENARIO_PRESETS[args.scenario_level]["zone_tilt_orientation_seed"] = \
            int(args.zone_tilt_orientation_seed)
    if args.zone_tilt_amplitude_mode is not None:
        SCENARIO_PRESETS[args.scenario_level]["zone_tilt_amplitude_mode"] = \
            str(args.zone_tilt_amplitude_mode)
    if args.zone_tilt_normalization is not None:
        SCENARIO_PRESETS[args.scenario_level]["zone_tilt_normalization"] = \
            str(args.zone_tilt_normalization)
    if args.zone_tilt_targets is not None:
        SCENARIO_PRESETS[args.scenario_level]["zone_tilt_targets"] = \
            str(args.zone_tilt_targets)
    if args.load_zone_correlation is not None:
        SCENARIO_PRESETS[args.scenario_level]["load_zone_correlation"] = \
            str(args.load_zone_correlation)
        SCENARIO_PRESETS[args.scenario_level]["load_zone_structure_seed"] = \
            int(args.load_zone_structure_seed)
    if args.contingency_ratio_weights is not None:
        SCENARIO_PRESETS[args.scenario_level]["contingency_ratio_weights"] = \
            tuple(float(x) for x in args.contingency_ratio_weights)
    scenario_cfg = SCENARIO_PRESETS[args.scenario_level]
    _jit_on, _jit_ang, _jit_mag = resolve_start_jitter(args)
    source_label = _case_label_from_source(
        preset=str(args.preset or "").strip(),
        cgmes_path=str(args.cgmes_path or "").strip(),
        case_name=str(args.case_name or "").strip(),
    )

    workers = args.workers if args.workers and args.workers > 0 else (os.cpu_count() or 1)
    runs = int(args.runs)
    rows_per_task = max(int(args.rows_per_task), 1)
    save_steps = max(int(args.save_steps), 1)
    save_y_matrix = bool(args.save_y_matrix)

    filename = build_output_filename(args)
    print(f"[INFO] Output file: {filename}")

    cfg = dict(
        preset=str(args.preset or "").strip(),
        cgmes_path=str(args.cgmes_path or "").strip(),
        case_name=str(args.case_name or "").strip(),
        cgmes_version=str(args.cgmes_version).strip(),
        cgmes_model_a_cleanup=bool(args.cgmes_model_a_cleanup),
        base_sn_mva=(float(args.base_sn_mva) if args.base_sn_mva is not None else None),
        cgmes_ignore_errors=bool(args.cgmes_ignore_errors),
        ybus_mode=str(args.ybus_mode).strip(),
        K=int(args.K),
        corpus_seed=int(args.corpus_seed),
        save_y_matrix=save_y_matrix,
        save_nr_trajectory=bool(args.save_nr_trajectory),

        jitter_load=float(scenario_cfg["jitter_load"]),
        jitter_load_q=float(
            args.jitter_load_q
            if args.jitter_load_q is not None
            else scenario_cfg["jitter_load_q"]
        ),
        jitter_gen=float(scenario_cfg["jitter_gen"]),
        # Global load scale: CLI overrides take priority over preset
        load_scale_range=(
            None
            if (scenario_cfg["load_scale_range"] is None
                and args.load_scale_lo is None
                and args.load_scale_hi is None)
            else (
                float(args.load_scale_lo if args.load_scale_lo is not None
                      else scenario_cfg["load_scale_range"][0]),
                float(args.load_scale_hi if args.load_scale_hi is not None
                      else scenario_cfg["load_scale_range"][1]),
            )
        ),
        scale_gen_with_load=bool(scenario_cfg["scale_gen_with_load"]),
        perturb_sgen=bool(args.perturb_sgen),
        pv_q_from_vstart=bool(args.pv_q_from_vstart),
        line_outage_prob=float(
            args.line_outage_prob
            if args.line_outage_prob is not None
            else scenario_cfg["line_outage_prob"]
        ),
        drop_nonconverged=bool(args.drop_nonconverged),
        pv_vset_range=scenario_cfg["pv_vset_range"],
        rand_u_start=_jit_on,
        angle_jitter_deg=_jit_ang,
        mag_jitter_pq=_jit_mag,

        trafo_pfe_kw=args.trafo_pfe_kw,
        trafo_i0_percent=args.trafo_i0_percent,

        use_force_shunt_when_no_trafo=bool(args.use_force_shunt_when_no_trafo),
        force_branch_shunt_g_pu=float(args.force_branch_shunt_g_pu),
        force_branch_shunt_b_pu=float(args.force_branch_shunt_b_pu),
        force_branch_shunt_g_asym_pu=float(args.force_branch_shunt_g_asym_pu),
        force_branch_shunt_b_asym_pu=float(args.force_branch_shunt_b_asym_pu),

        pu_nr=bool(args.pu_nr),
        label_solver=str(args.label_solver).strip(),
        diagnose_nr=bool(args.diagnose_nr),
        print_misinf=bool(args.print_misinf),
        near_misinf_tol=float(args.near_misinf_tol),
        start_mode=str(args.start_mode).strip(),

        convergence_mode=str(args.convergence_mode).strip(),
        step_tol=float(args.step_tol),
        mismatch_tol=float(args.mismatch_tol),

        # Scenario-preset knobs that the worker reads back out of _CFG.  cfg is
        # an explicit key list, so a preset entry missing here is silently lost
        # on the way to the pool -- which is how ood_zone and ood_n1 ran with
        # no tilt and no outage at all.
        load_zone_sigma=float(scenario_cfg.get("load_zone_sigma", 0.0)),
        load_zone_count=int(scenario_cfg.get("load_zone_count", 0)),
        load_zone_correlation=str(scenario_cfg.get("load_zone_correlation", "none")),
        load_zone_structure_seed=int(scenario_cfg.get("load_zone_structure_seed", 0)),
        zone_tilt_targets=str(scenario_cfg.get("zone_tilt_targets", "load")),
        zone_tilt_normalization=str(scenario_cfg.get(
            "zone_tilt_normalization", "load_weighted")),
        zone_tilt_amplitude_mode=str(scenario_cfg.get(
            "zone_tilt_amplitude_mode", "sigma")),
        zone_tilt_orientation_seed=scenario_cfg.get(
            "zone_tilt_orientation_seed", None),
        contingency_mode=str(scenario_cfg.get("contingency_mode", "none")),
        contingency_elements=tuple(scenario_cfg.get("contingency_elements", ("line",))),
        contingency_k_cap=int(scenario_cfg.get("contingency_k_cap", 2)),
        contingency_ratio_weights=tuple(
            scenario_cfg.get("contingency_ratio_weights", (0.55, 0.27, 0.18))),
        contingency_poisson_q=float(scenario_cfg.get("contingency_poisson_q", 5e-4)),
    )

    num_tasks = math.ceil(runs / rows_per_task)
    task_sizes = [rows_per_task] * (num_tasks - 1) + [runs - rows_per_task * (num_tasks - 1)]
    # (start_index, count): the start makes each attempt globally identifiable,
    # which is what pairs two corpora row for row.
    _starts = np.cumsum([0] + task_sizes[:-1]).tolist()
    task_sizes = list(zip(_starts, task_sizes))

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
        save_nr_trajectory=bool(args.save_nr_trajectory),
    )

    print("[INFO] Configuration")
    print(f"  source_label              = {source_label}")
    print(f"  preset                    = {args.preset}")
    print(f"  cgmes_path                = {args.cgmes_path}")
    print(f"  case_name                 = {args.case_name}")
    print(f"  cgmes_version             = {args.cgmes_version}")
    print(f"  cgmes_model_a_cleanup     = {args.cgmes_model_a_cleanup}")
    print(f"  base_sn_mva               = {args.base_sn_mva}")
    print(f"  cgmes_ignore_errors       = {args.cgmes_ignore_errors}")
    print(f"  ybus_mode                 = {args.ybus_mode}")
    print(f"  scenario_level            = {args.scenario_level}")
    print(f"  scenario_cfg              = {scenario_cfg}")
    print(f"  jitter_load_q (effective) = {cfg['jitter_load_q']}")
    print(f"  load_scale_range          = {cfg['load_scale_range']}")
    print(f"  perturb_sgen              = {cfg['perturb_sgen']}")
    print(f"  pv_q_from_vstart          = {cfg['pv_q_from_vstart']}")
    print(f"  rand_u_start (effective)  = {_jit_on}"
          f"{'' if _jit_on else '   [clean start: jitter is opt-in via --rand_u_start]'}")
    if _jit_on:
        print(f"  angle_jitter_deg          = {_jit_ang}")
        print(f"  mag_jitter_pq             = {_jit_mag}")
    print(f"  scale_gen_with_load       = {cfg['scale_gen_with_load']}")
    print(f"  line_outage_prob          = {cfg['line_outage_prob']}")
    print(f"  drop_nonconverged         = {cfg['drop_nonconverged']}")
    print(f"  K                         = {args.K}")
    print(f"  pu_nr                     = {args.pu_nr}")
    print(f"  label_solver              = {args.label_solver}")
    print(f"  start_mode                = {args.start_mode}")
    print(f"  use_force_shunt_when_no_trafo = {args.use_force_shunt_when_no_trafo}")
    print(f"  diagnose_nr               = {args.diagnose_nr}")
    print(f"  convergence_mode          = {args.convergence_mode}")
    print(f"  step_tol                  = {args.step_tol}")
    print(f"  mismatch_tol              = {args.mismatch_tol}")
    print(f"  workers                   = {workers}")
    print(f"  rows_per_task             = {rows_per_task}")
    print(f"  save_steps                = {save_steps}")
    print(f"  save_y_matrix             = {save_y_matrix}")
    print(f"  save_nr_trajectory        = {args.save_nr_trajectory}")

    if args.print_misinf and workers != 1:
        print("[WARN] print_misinf=True with workers>1 will produce interleaved logs.")

    buffer: List[Dict[str, Any]] = []

    try:
        with ctx.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(cfg, int(time.time())),
            maxtasksperchild=1000
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

    # The row-count field in the name is built from --runs, which counts
    # ATTEMPTS. With --drop_nonconverged the two differ, and under
    # oversampling they differ a lot: requesting 80,000 attempts to retain
    # 36,000 rows would otherwise leave a file called "_80000_" holding less
    # than half that. The merged array datasets already carry true counts
    # (the merger names them by actual rows), so without this the single-node
    # and array halves of the corpus mean different things by the same field.
    if total_written != args.runs:
        true_name = filename.replace(f"_{args.runs}_NR_", f"_{total_written}_NR_")
        if true_name != filename and not os.path.exists(true_name):
            os.replace(filename, true_name)
            print(f"[INFO] renamed to true row count: {args.runs} attempted -> "
                  f"{total_written} retained")
            filename = true_name

    elapsed = time.time() - start
    rate = total_written / max(elapsed, 1e-6)
    size_gb = get_parquet_file_size(filename) / 1e9
    print(f"[DONE] Wrote {total_written:,} rows in {elapsed:,.1f}s ({rate:,.0f} rows/s)")
    print(f"[INFO] File size: {size_gb:.3f} GB -> {filename}")


if __name__ == "__main__":
    main()
