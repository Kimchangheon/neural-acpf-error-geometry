#!/usr/bin/env python3
"""
Generate AC-OPF datasets from CGMES snapshot envelopes with pandapower's runopp.

This is the OPF counterpart of `main_datagen_lvn_snapshot_envelope.py`. The
scenario sampling, the compiled-PPC branch-row direct-SI representation, and the
parquet layout are identical, so the resulting file loads through the same
`ChanghunDataset` path. Only the supervision target changes: instead of solving
a power flow for a given dispatch with the custom Newton-Raphson, each scenario
is solved with `pandapower.runopp` and the OPF operating point is stored.

OPF definition ("curtailment OPF")
----------------------------------
The CGMES nets carry no cost data, no voltage band, and no thermal limits, so
the optimisation problem is constructed here:

  * bus voltage band          `--vmin` / `--vmax`            (default 0.9 / 1.1)
  * line/trafo thermal limit  `--max_loading_percent`        (default 100)
  * static generators         dispatchable in [0, P_snapshot], i.e. renewables
                              may be curtailed but not increased beyond the
                              power available in the snapshot
  * sgen reactive range       +/- max(`--sgen_q_frac` * P_avail, 1 MVAr)
  * synchronous generators    dispatchable over their existing P/Q limits
  * external grids            free to import and export (`--ext_grid_limit_mw`)
  * objective                 linear cost: renewables `--sgen_cost`, generators
                              `--gen_cost`, external grid `--ext_grid_cost`,
                              which prices curtailment against import

Holding the static generators at their snapshot values is infeasible on these
SimBench snapshots: the exported dispatch carries ~3.7 GW of generation against
~0.1 GW of load, exceeds the units' own `max_p_mw`, and cannot be pushed through
the network without violating thermal limits. Allowing curtailment is the
smallest change that yields a well-posed problem.

Stored supervision
------------------
`u_newton` holds the raw `runopp` voltage solution, so the target is exactly the
pandapower OPF answer. That solution satisfies the power-flow equations only to
the OPF solver's tolerance (order 1e-5 pu on a 100 MVA base, against 1e-13 pu
for the Newton targets of the power-flow datasets); the per-row residual is
stored in `opf_resid_dp_pu` / `opf_resid_dq_pu` so this noise floor is
measurable rather than implicit.

`S_start` holds the load injections only, with zero at every generator bus, so a
model consuming it is not handed the dispatch it is meant to predict. The full
OPF injection is stored separately in `S_newton`, and the decision space that
defines the problem (per-bus generation limits, available renewable power, cost
slope, voltage band, branch ratings) is stored in the `Gen_*` / `Bus_*` /
`Branch_rate_a` columns.
"""

from __future__ import annotations

import copy
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from case_generator_lvn_snapshot_envelope import (
    DEFAULT_LVN_CGMES,
    DEFAULT_SNAPSHOT_ROOT,
    _sample_loads_from_bounds,
    align_bounds_to_base_net,
    get_or_build_snapshot_load_bounds,
    prepare_lvn_base_net,
)
from case_generator_all_test_cases_pandapower_consider_ppc_branch_row import (
    case_generation_pandapower,
)
from build_snapshot_sgen_bounds import (
    align_sgen_bounds_to_net,
    load_snapshot_sgen_bounds_npz,
)


def ndarray_to_npy_bytes(x: Any) -> bytes:
    import io

    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(x), allow_pickle=False)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# OPF problem construction
# --------------------------------------------------------------------------- #

def apply_opf_setup(net, cfg: Dict[str, Any]):
    """Turn a sampled snapshot net into a well-posed curtailment OPF."""
    import pandapower as pp

    net.bus["min_vm_pu"] = float(cfg["vmin"])
    net.bus["max_vm_pu"] = float(cfg["vmax"])

    mlp = float(cfg["max_loading_percent"])
    if len(net.line):
        net.line["max_loading_percent"] = mlp
    if len(net.trafo):
        net.trafo["max_loading_percent"] = mlp
    if len(net.trafo3w):
        net.trafo3w["max_loading_percent"] = mlp

    eg_lim = float(cfg["ext_grid_limit_mw"])
    if len(net.ext_grid):
        net.ext_grid["min_p_mw"] = -eg_lim
        net.ext_grid["max_p_mw"] = eg_lim
        net.ext_grid["min_q_mvar"] = -eg_lim
        net.ext_grid["max_q_mvar"] = eg_lim

    # Renewables: dispatchable between zero and the power available in the
    # snapshot. p_avail is what the OPF may use; curtailment is the difference.
    p_avail = np.zeros(len(net.sgen), dtype=float)
    if len(net.sgen):
        p_avail = np.asarray(net.sgen["p_mw"], dtype=float).clip(min=0.0)
        net.sgen["controllable"] = True
        net.sgen["min_p_mw"] = 0.0
        net.sgen["max_p_mw"] = p_avail
        q_cap = np.maximum(float(cfg["sgen_q_frac"]) * p_avail, 1.0)
        net.sgen["max_q_mvar"] = q_cap
        net.sgen["min_q_mvar"] = -q_cap

    if len(net.gen):
        net.gen["controllable"] = True

    for idx in net.sgen.index:
        pp.create_poly_cost(net, idx, "sgen", cp1_eur_per_mw=float(cfg["sgen_cost"]))
    for idx in net.gen.index:
        pp.create_poly_cost(net, idx, "gen", cp1_eur_per_mw=float(cfg["gen_cost"]))
    for idx in net.ext_grid.index:
        pp.create_poly_cost(net, idx, "ext_grid", cp1_eur_per_mw=float(cfg["ext_grid_cost"]))

    return net, p_avail


def _bus_of(df) -> np.ndarray:
    return np.asarray(df["bus"], dtype=int)


def aggregate_to_ppc(values: np.ndarray, buses: np.ndarray, lookup: np.ndarray, n_ppc: int) -> np.ndarray:
    """Sum per-element quantities onto their compiled-PPC bus."""
    out = np.zeros(n_ppc, dtype=float)
    for value, bus in zip(values, buses):
        j = int(lookup[int(bus)])
        if 0 <= j < n_ppc:
            out[j] += float(value)
    return out


def opf_record_arrays(net_opf, net_struct, p_avail, lookup, n_ppc, vn_kv, cfg):
    """Map a solved OPF net onto compiled-PPC bus indices, in SI units."""
    Vbase = np.asarray(vn_kv, dtype=float) * 1e3

    vm = np.asarray(net_opf.res_bus["vm_pu"], dtype=float)
    va = np.deg2rad(np.asarray(net_opf.res_bus["va_degree"], dtype=float))
    u_opf = np.zeros(n_ppc, dtype=np.complex128)
    for pdbus in range(len(net_opf.bus)):
        j = int(lookup[pdbus])
        if 0 <= j < n_ppc and np.isfinite(vm[pdbus]) and np.isfinite(va[pdbus]):
            u_opf[j] = vm[pdbus] * np.exp(1j * va[pdbus]) * Vbase[j]

    S_load = np.zeros(n_ppc, dtype=np.complex128)
    S_gen = np.zeros(n_ppc, dtype=np.complex128)

    def accumulate(target, df, res, sign):
        if not len(df):
            return
        buses = _bus_of(df)
        p = np.asarray(res["p_mw"], dtype=float)
        q = np.asarray(res["q_mvar"], dtype=float)
        for k in range(len(df)):
            j = int(lookup[int(buses[k])])
            if 0 <= j < n_ppc:
                target[j] += sign * (p[k] + 1j * q[k]) * 1e6

    accumulate(S_load, net_opf.load, net_opf.res_load, -1.0)
    accumulate(S_gen, net_opf.sgen, net_opf.res_sgen, +1.0)
    accumulate(S_gen, net_opf.gen, net_opf.res_gen, +1.0)
    accumulate(S_gen, net_opf.ext_grid, net_opf.res_ext_grid, +1.0)

    # Decision space, aggregated per PPC bus, in SI.
    p_min = np.zeros(n_ppc); p_max = np.zeros(n_ppc)
    q_min = np.zeros(n_ppc); q_max = np.zeros(n_ppc)
    avail = np.zeros(n_ppc); cost = np.zeros(n_ppc)
    ctrl = np.zeros(n_ppc, dtype=np.int8)

    def add_limits(df, cost_slope, avail_vec=None):
        if not len(df):
            return
        buses = _bus_of(df)
        for col, target in (("min_p_mw", p_min), ("max_p_mw", p_max),
                            ("min_q_mvar", q_min), ("max_q_mvar", q_max)):
            if col in df.columns:
                vals = np.asarray(df[col], dtype=float)
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                target += aggregate_to_ppc(vals * 1e6, buses, lookup, n_ppc)
        for k in range(len(df)):
            j = int(lookup[int(buses[k])])
            if 0 <= j < n_ppc:
                ctrl[j] = 1
                cost[j] = float(cost_slope)
                if avail_vec is not None:
                    avail[j] += float(avail_vec[k]) * 1e6

    add_limits(net_opf.sgen, cfg["sgen_cost"], p_avail)
    add_limits(net_opf.gen, cfg["gen_cost"])
    add_limits(net_opf.ext_grid, cfg["ext_grid_cost"])

    return {
        "u_opf": u_opf,
        "S_load": S_load,
        "S_gen": S_gen,
        "gen_p_min": p_min, "gen_p_max": p_max,
        "gen_q_min": q_min, "gen_q_max": q_max,
        "gen_p_avail": avail, "gen_cost_c1": cost,
        "gen_controllable": ctrl,
    }


# --------------------------------------------------------------------------- #
# Parquet writer
# --------------------------------------------------------------------------- #

_BINARY_COLUMNS = [
    "bus_typ", "vn_kv", "Y_shunt_bus",
    "Branch_f_bus", "Branch_t_bus", "Branch_status", "Branch_tau", "Branch_shift_deg",
    "Branch_y_series_from", "Branch_y_series_to", "Branch_y_series_ft",
    "Branch_y_shunt_from", "Branch_y_shunt_to",
    "Is_trafo", "Branch_hv_is_f", "Branch_n", "Y_Lines", "Y_C_Lines",
    "u_start", "u_newton", "S_start", "S_newton",
    # OPF decision space and diagnostics
    "S_load", "S_gen_opf",
    "Gen_p_min", "Gen_p_max", "Gen_q_min", "Gen_q_max",
    "Gen_p_avail", "Gen_cost_c1", "Gen_controllable",
    "Bus_vmin", "Bus_vmax", "Branch_rate_a",
]

_SCALAR_COLUMNS = [
    ("bus_number", pa.int32()),
    ("branch_number", pa.int32()),
    ("gridtype", pa.string()),
    ("U_base", pa.float64()),
    ("S_base", pa.float64()),
    ("opf_cost", pa.float64()),
    ("opf_curtailed_mw", pa.float64()),
    ("opf_resid_dp_pu", pa.float64()),
    ("opf_resid_dq_pu", pa.float64()),
    ("opf_converged", pa.int32()),
]


class ParquetAppendWriter:
    def __init__(self, path: str, compression: str = "zstd", overwrite: bool = True,
                 save_y_matrix: bool = False, row_group_size: int = 20):
        self.path = path
        self.save_y_matrix = bool(save_y_matrix)
        # Row-group size drives random-access cost, and training shuffles. The
        # power-flow parquets use 20 rows per group; writing one group per flush
        # instead (500 rows) makes every shuffled batch decode ~500 rows to use
        # 4, which measured 3192 ms/batch against 176 ms/batch at 20 rows.
        self.row_group_size = max(1, int(row_group_size))
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        if overwrite and os.path.exists(path):
            os.remove(path)

        fields = [pa.field(name, dtype) for name, dtype in _SCALAR_COLUMNS]
        fields += [pa.field(name, pa.binary()) for name in _BINARY_COLUMNS]
        if self.save_y_matrix:
            fields.append(pa.field("Y_matrix", pa.binary()))

        self._schema = pa.schema(fields)
        self._writer = pq.ParquetWriter(where=path, schema=self._schema,
                                        compression=compression, use_dictionary=True)

    def write_records(self, records: List[Dict[str, Any]]):
        if not records:
            return
        cols = {name: [] for name in self._schema.names}
        for record in records:
            for key in cols:
                cols[key].append(record[key])
        arrays = []
        for name, field in zip(self._schema.names, self._schema):
            arrays.append(pa.array(cols[name], type=field.type))
        self._writer.write_table(pa.Table.from_arrays(arrays, schema=self._schema),
                                 row_group_size=self.row_group_size)

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #

_CFG: Dict[str, Any] = {}
_RNG = None
_BASE_NET = None
_ALIGNED_BOUNDS = None
_SGEN_BOUNDS = None


def _sample_sgen_availability(rng, p_min, p_max, sample_mode: str) -> np.ndarray:
    """Draw available renewable power from the snapshot envelope.

    Mirrors `_sample_loads_from_bounds`: "coupled" moves every unit together
    along one draw, which reflects the spatial correlation of wind over a single
    control area, while "independent" draws each unit separately.
    """
    lo = np.minimum(p_min, p_max)
    hi = np.maximum(p_min, p_max)
    if str(sample_mode).strip().lower() == "independent":
        t = rng.random(size=lo.shape)
    else:
        t = rng.random()
    return np.clip(lo + t * (hi - lo), 0.0, None)


def _init_worker(cfg: dict, seed_base: int):
    global _CFG, _RNG, _BASE_NET, _ALIGNED_BOUNDS, _SGEN_BOUNDS
    import warnings

    warnings.filterwarnings("ignore")
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
    _ALIGNED_BOUNDS = align_bounds_to_base_net(_BASE_NET, _CFG["load_bounds_df"])
    sgen_bounds_df = _CFG.get("sgen_bounds_df", None)
    if sgen_bounds_df is not None and len(_BASE_NET.sgen):
        _SGEN_BOUNDS = align_sgen_bounds_to_net(_BASE_NET, sgen_bounds_df)
    else:
        _SGEN_BOUNDS = None


def _generate_one_record_serialized() -> Optional[Dict[str, Any]]:
    global _CFG, _RNG, _BASE_NET, _ALIGNED_BOUNDS, _SGEN_BOUNDS
    import pandapower as pp

    cfg = _CFG
    sampled = _sample_loads_from_bounds(
        _RNG,
        p_min_mw=np.asarray(_ALIGNED_BOUNDS["p_min_mw"], dtype=float),
        p_max_mw=np.asarray(_ALIGNED_BOUNDS["p_max_mw"], dtype=float),
        q_min_mvar=np.asarray(_ALIGNED_BOUNDS["q_min_mvar"], dtype=float),
        q_max_mvar=np.asarray(_ALIGNED_BOUNDS["q_max_mvar"], dtype=float),
        sample_mode=str(cfg["sample_mode"]).strip(),
    )

    net = copy.deepcopy(_BASE_NET)
    net.name = str(cfg["case_name"]).strip()
    net.load.loc[:, "p_mw"] = sampled["p_mw"]
    net.load.loc[:, "q_mvar"] = sampled["q_mvar"]

    # Renewable availability is drawn from its own snapshot envelope. Holding it
    # at the base-net value collapses the OPF onto one corner of the feasible
    # set: the SimBench base export is a maximum-wind snapshot, so every
    # scenario would curtail ~97% of the same constant availability.
    if _SGEN_BOUNDS is not None:
        p_sampled = _sample_sgen_availability(
            _RNG,
            _SGEN_BOUNDS["p_min_mw"],
            _SGEN_BOUNDS["p_max_mw"],
            str(cfg.get("sgen_sample_mode", "coupled")),
        )
        net.sgen.loc[:, "p_mw"] = p_sampled

    # 1) solve the OPF on a copy carrying the synthesised problem data
    opf_net = copy.deepcopy(net)
    if bool(cfg["set_isolated_out_of_service"]):
        pp.set_isolated_areas_out_of_service(opf_net)
    opf_net, p_avail = apply_opf_setup(opf_net, cfg)
    try:
        pp.runopp(opf_net, init=str(cfg["opf_init"]).strip(), verbose=False)
    except Exception:
        return None
    if not bool(getattr(opf_net, "OPF_converged", True)):
        return None

    # 2) compile the PPC structure from the same sampled net
    def _factory(**_kwargs):
        return net

    _factory.__name__ = net.name
    out = case_generation_pandapower(
        case_fn=_factory, case_kwargs={}, ybus_mode=str(cfg["ybus_mode"]).strip(),
        cgmes_model_a_cleanup=False, seed=0, jitter_load=0.0, jitter_gen=0.0,
        pv_vset_range=None, rand_u_start=False, angle_jitter_deg=0.0, mag_jitter_pq=0.0,
        trafo_pfe_kw=None, trafo_i0_percent=None, force_branch_shunt_pu=None,
        start_mode=str(cfg["start_mode"]).strip(),
    )
    (gridtype, bus_typ, _s_multi, u_start, Y_matrix, is_connected,
     Branch_f_bus, Branch_t_bus, Branch_status, Branch_tau, Branch_shift_deg,
     Branch_y_series_from, Branch_y_series_to, Branch_y_series_ft,
     Branch_y_shunt_from, Branch_y_shunt_to, Y_shunt_bus,
     Is_trafo, Branch_hv_is_f, Branch_n, Y_Lines, Y_C_Lines,
     U_base, S_base, vn_kv) = out

    if not is_connected:
        return None

    n_ppc = int(len(bus_typ))
    lookup = np.asarray(net._pd2ppc_lookups["bus"])
    arrays = opf_record_arrays(opf_net, net, p_avail, lookup, n_ppc, vn_kv, cfg)

    # 3) power-balance residual of the stored OPF point, on the dataset's own Ybus
    u_opf = arrays["u_opf"]
    S_total = arrays["S_load"] + arrays["S_gen"]
    resid = S_total - u_opf * np.conj(np.asarray(Y_matrix) @ u_opf)
    dp_pu = float(np.abs(resid.real).max() / float(S_base))
    dq_pu = float(np.abs(resid.imag).max() / float(S_base))

    # 4) branch thermal ratings in SI (VA). pandapower's branch lookup gives the
    #    exact PPC branch slice per element table, so lines and transformers map
    #    row-for-row rather than being approximated per bus.
    rate_a = np.zeros(int(len(Branch_f_bus)), dtype=float)
    mlp = float(cfg["max_loading_percent"]) / 100.0
    branch_lookup = net._pd2ppc_lookups.get("branch", {}) or {}
    vb = np.asarray(vn_kv, dtype=float) * 1e3
    f_bus = np.asarray(Branch_f_bus, dtype=int)

    if "line" in branch_lookup and len(net.line) and "max_i_ka" in net.line.columns:
        start, end = (int(v) for v in branch_lookup["line"])
        max_i = np.nan_to_num(np.asarray(net.line["max_i_ka"], dtype=float), nan=0.0)
        n = min(end - start, len(max_i), len(rate_a) - start)
        for k in range(n):
            j = int(f_bus[start + k])
            if 0 <= j < n_ppc:
                rate_a[start + k] = np.sqrt(3.0) * vb[j] * max_i[k] * 1e3 * mlp

    for table in ("trafo", "trafo3w"):
        if table in branch_lookup and len(net[table]) and "sn_mva" in net[table].columns:
            start, end = (int(v) for v in branch_lookup[table])
            sn = np.nan_to_num(np.asarray(net[table]["sn_mva"], dtype=float), nan=0.0)
            n = min(end - start, len(sn), len(rate_a) - start)
            for k in range(n):
                rate_a[start + k] = sn[k] * 1e6 * mlp

    curtailed = float(np.sum(p_avail) - float(opf_net.res_sgen["p_mw"].sum())) if len(opf_net.sgen) else 0.0

    record = {
        "bus_number": n_ppc,
        "branch_number": int(len(Branch_f_bus)),
        "gridtype": gridtype,
        "U_base": float(U_base),
        "S_base": float(S_base),
        "opf_cost": float(getattr(opf_net, "res_cost", np.nan)),
        "opf_curtailed_mw": curtailed,
        "opf_resid_dp_pu": dp_pu,
        "opf_resid_dq_pu": dq_pu,
        "opf_converged": 1,
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
        "u_newton": ndarray_to_npy_bytes(np.asarray(u_opf, dtype=np.complex128)),
        "S_start": ndarray_to_npy_bytes(np.asarray(arrays["S_load"], dtype=np.complex128)),
        "S_newton": ndarray_to_npy_bytes(np.asarray(S_total, dtype=np.complex128)),
        "S_load": ndarray_to_npy_bytes(np.asarray(arrays["S_load"], dtype=np.complex128)),
        "S_gen_opf": ndarray_to_npy_bytes(np.asarray(arrays["S_gen"], dtype=np.complex128)),
        "Gen_p_min": ndarray_to_npy_bytes(np.asarray(arrays["gen_p_min"], dtype=np.float64)),
        "Gen_p_max": ndarray_to_npy_bytes(np.asarray(arrays["gen_p_max"], dtype=np.float64)),
        "Gen_q_min": ndarray_to_npy_bytes(np.asarray(arrays["gen_q_min"], dtype=np.float64)),
        "Gen_q_max": ndarray_to_npy_bytes(np.asarray(arrays["gen_q_max"], dtype=np.float64)),
        "Gen_p_avail": ndarray_to_npy_bytes(np.asarray(arrays["gen_p_avail"], dtype=np.float64)),
        "Gen_cost_c1": ndarray_to_npy_bytes(np.asarray(arrays["gen_cost_c1"], dtype=np.float64)),
        "Gen_controllable": ndarray_to_npy_bytes(np.asarray(arrays["gen_controllable"], dtype=np.int8)),
        "Bus_vmin": ndarray_to_npy_bytes(np.full(n_ppc, float(cfg["vmin"]), dtype=np.float64)),
        "Bus_vmax": ndarray_to_npy_bytes(np.full(n_ppc, float(cfg["vmax"]), dtype=np.float64)),
        "Branch_rate_a": ndarray_to_npy_bytes(np.asarray(rate_a, dtype=np.float64)),
    }
    if bool(cfg["save_y_matrix"]):
        record["Y_matrix"] = ndarray_to_npy_bytes(np.asarray(Y_matrix, dtype=np.complex128))
    return record


def _generate_batch(n_rows: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for _ in range(int(n_rows)):
        try:
            record = _generate_one_record_serialized()
        except Exception:
            if os.environ.get("GEN_DEBUG", "0") == "1":
                traceback.print_exc()
            record = None
        if record is not None:
            out.append(record)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description="Generate AC-OPF snapshot-envelope datasets with pandapower runopp.",
    )
    p.add_argument("--base_cgmes_path", type=str, default=DEFAULT_LVN_CGMES)
    p.add_argument("--snapshot_root", type=str, default=DEFAULT_SNAPSHOT_ROOT)
    p.add_argument("--case_name", type=str, default="LVN_snapshot_envelope_opf")
    p.add_argument("--cgmes_version", type=str, default="2.4.15")
    p.add_argument("--bounds_cache_path", type=str, default="")
    p.add_argument("--base_sn_mva", type=float, default=None)
    p.add_argument("--cgmes_ignore_errors", action="store_true")
    p.add_argument("--no_cgmes_model_a_cleanup", dest="cgmes_model_a_cleanup",
                   action="store_false")
    p.set_defaults(cgmes_model_a_cleanup=True)

    p.add_argument("--sample_mode", type=str, default="coupled")
    p.add_argument("--ybus_mode", type=str, default="ppcY")
    p.add_argument("--start_mode", type=str, default="manual_flat")
    p.add_argument("--sgen_bounds_path", type=str, default="",
                   help="npz from build_snapshot_sgen_bounds.py. Without it the "
                        "renewable availability stays fixed at the base snapshot, "
                        "which makes the OPF targets nearly constant.")
    p.add_argument("--sgen_sample_mode", type=str, default="coupled",
                   choices=("coupled", "independent"))

    # OPF problem definition
    p.add_argument("--vmin", type=float, default=0.9)
    p.add_argument("--vmax", type=float, default=1.1)
    p.add_argument("--max_loading_percent", type=float, default=100.0)
    p.add_argument("--sgen_q_frac", type=float, default=0.3)
    p.add_argument("--sgen_cost", type=float, default=0.0)
    p.add_argument("--gen_cost", type=float, default=30.0)
    p.add_argument("--ext_grid_cost", type=float, default=50.0)
    p.add_argument("--ext_grid_limit_mw", type=float, default=1e5)
    p.add_argument("--opf_init", type=str, default="flat")
    p.add_argument("--set_isolated_out_of_service", action="store_true")

    p.add_argument("--runs", type=int, default=10000)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--rows_per_task", type=int, default=20)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--save_path", type=str, default="")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save_y_matrix", action="store_true")
    p.add_argument("--row_group_size", type=int, default=20,
                   help="Parquet row-group size. Keep small (20, as in the "
                        "power-flow parquets): training shuffles, and large "
                        "groups make every batch decode a whole group.")
    p.add_argument("--seed_base", type=int, default=1234)
    return p.parse_args()


def build_output_filename(args) -> str:
    if args.save_path:
        return args.save_path
    return f"./out/{args.case_name}_{args.ybus_mode}_{args.start_mode}_{args.runs}_OPF_branchrows_directSI.parquet"


def main():
    import multiprocessing as mp

    args = parse_args()
    save_path = build_output_filename(args)
    print(f"[opf-datagen] case={args.case_name}")
    print(f"[opf-datagen] base_cgmes={args.base_cgmes_path}")
    print(f"[opf-datagen] output={save_path}")
    print(f"[opf-datagen] OPF: vm[{args.vmin},{args.vmax}] loading<={args.max_loading_percent}% "
          f"sgen_cost={args.sgen_cost} gen_cost={args.gen_cost} ext_grid_cost={args.ext_grid_cost}")

    bounds = get_or_build_snapshot_load_bounds(
        args.snapshot_root,
        cgmes_version=args.cgmes_version,
        ignore_errors=bool(args.cgmes_ignore_errors),
        cache_path=args.bounds_cache_path,
    )

    sgen_bounds = None
    if str(args.sgen_bounds_path).strip():
        sgen_bounds = load_snapshot_sgen_bounds_npz(args.sgen_bounds_path)
        print(f"[opf-datagen] sgen envelope: {len(sgen_bounds)} units, total available P "
              f"{sgen_bounds['p_min_mw'].sum():.1f}-{sgen_bounds['p_max_mw'].sum():.1f} MW")
    else:
        print("[opf-datagen] WARNING: no --sgen_bounds_path; renewable availability "
              "stays fixed at the base snapshot and OPF targets will barely vary.")

    cfg = {
        "sgen_bounds_df": sgen_bounds,
        "sgen_sample_mode": args.sgen_sample_mode,
        "base_cgmes_path": args.base_cgmes_path,
        "case_name": args.case_name,
        "cgmes_version": args.cgmes_version,
        "cgmes_ignore_errors": bool(args.cgmes_ignore_errors),
        "cgmes_model_a_cleanup": bool(args.cgmes_model_a_cleanup),
        "base_sn_mva": args.base_sn_mva,
        "load_bounds_df": bounds,
        "sample_mode": args.sample_mode,
        "ybus_mode": args.ybus_mode,
        "start_mode": args.start_mode,
        "vmin": args.vmin, "vmax": args.vmax,
        "max_loading_percent": args.max_loading_percent,
        "sgen_q_frac": args.sgen_q_frac,
        "sgen_cost": args.sgen_cost, "gen_cost": args.gen_cost,
        "ext_grid_cost": args.ext_grid_cost,
        "ext_grid_limit_mw": args.ext_grid_limit_mw,
        "opf_init": args.opf_init,
        "set_isolated_out_of_service": bool(args.set_isolated_out_of_service),
        "save_y_matrix": bool(args.save_y_matrix),
    }

    workers = int(args.workers) or max(1, (os.cpu_count() or 2) - 1)
    writer = ParquetAppendWriter(save_path, overwrite=bool(args.overwrite),
                                 save_y_matrix=bool(args.save_y_matrix),
                                 row_group_size=args.row_group_size)

    written = 0
    attempted = 0
    failed = 0
    t_start = time.time()
    pending: List[Dict[str, Any]] = []
    tasks = [args.rows_per_task] * (args.runs // args.rows_per_task)
    if args.runs % args.rows_per_task:
        tasks.append(args.runs % args.rows_per_task)

    try:
        if workers <= 1:
            _init_worker(cfg, args.seed_base)
            for n in tasks:
                batch = _generate_batch(n)
                attempted += n
                failed += n - len(batch)
                pending.extend(batch)
                if len(pending) >= args.save_steps:
                    writer.write_records(pending); written += len(pending); pending = []
                    print(f"[opf-datagen] wrote {written}/{args.runs} "
                          f"({failed} failed) {time.time()-t_start:.0f}s", flush=True)
        else:
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=workers, initializer=_init_worker,
                          initargs=(cfg, args.seed_base)) as pool:
                for batch in pool.imap_unordered(_generate_batch, tasks):
                    attempted += args.rows_per_task
                    failed += args.rows_per_task - len(batch)
                    pending.extend(batch)
                    if len(pending) >= args.save_steps:
                        writer.write_records(pending); written += len(pending); pending = []
                        print(f"[opf-datagen] wrote {written}/{args.runs} "
                              f"({failed} failed) {time.time()-t_start:.0f}s", flush=True)
        if pending:
            writer.write_records(pending); written += len(pending)
    finally:
        writer.close()

    elapsed = time.time() - t_start
    print(f"[opf-datagen] done: {written} rows written, {failed} scenarios failed, "
          f"{elapsed:.0f}s ({written/max(elapsed,1e-9):.2f} rows/s)")
    print(f"[opf-datagen] output: {save_path} "
          f"({os.path.getsize(save_path)/1e6:.1f} MB)" if os.path.exists(save_path) else "")


if __name__ == "__main__":
    main()
