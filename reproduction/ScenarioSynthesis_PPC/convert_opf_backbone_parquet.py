#!/usr/bin/env python3
"""
Adapt `*_opf_backbone_*_ACOPF_*` parquets to the training pipeline's schema.

Those files carry the same structural columns as the power-flow parquets but a
different supervision convention, and they omit the AC-OPF decision space:

    S_demand   -> S_start     (loads only: the model input)
    u_opf      -> u_newton    (target voltages)
    S_opf      -> S_newton    (full OPF injection)
    lam_p/lam_q, gen_p_opt/gen_q_opt, opf_cost, opf_converged  (kept as-is)

The decision space (per-bus generation limits, cost slope, controllable mask,
voltage band, branch ratings) is not in the file, but these datasets are built
on standard pandapower cases on a fixed topology, so it can be recovered once
from the case itself and broadcast to every row.

Angle reference
---------------
`pandapower`'s case118 places its slack bus at +30 deg, so every angle in that
dataset is offset and no bus sits near zero; case14, case300 and the SimBench
OPF set all use 0 deg. That datum is a gauge choice -- S = V conj(Y V) is exactly
invariant under a global phase rotation -- so it carries no physics, but it
penalises any model that cannot observe the bus angle state. LUMINA's bus schema
has no such input (7 columns: base_kv, vmin, vmax, one-hot type), so on case118
it cannot know the datum and pays ~30 deg before solving anything, while GridFM
(which ingests V_start) and GridSFM (which forms a DC prior from the slack) can.
Pass `--normalize_angle_reference` to rotate the slack to 0 deg and compare
models on physics rather than on an arbitrary reference.

The row groups are also re-laid-out. The source files use 1900-2000 rows per
group; training shuffles, so a batch of 4 then decodes a whole group. Measured
on this pipeline that costs about 18x (3191 ms/batch against 176 ms/batch at 20
rows per group).

    python convert_opf_backbone_parquet.py --src IN.parquet --dst OUT.parquet \
        --case case14 --vmin 0.9 --vmax 1.1
"""

from __future__ import annotations

import argparse
import io
import os
import time
from typing import Dict, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


RENAME = {"S_demand": "S_start", "u_opf": "u_newton", "S_opf": "S_newton"}
DERIVED = ["Gen_p_min", "Gen_p_max", "Gen_q_min", "Gen_q_max", "Gen_p_avail",
           "Gen_cost_c1", "Gen_controllable", "Gen_box_injection",
           "Bus_vmin", "Bus_vmax", "Branch_rate_a"]


def npy_bytes(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(a), allow_pickle=False)
    return buf.getvalue()


def load_case(name: str):
    import pandapower.networks as pn

    fn = {"case14": pn.case14, "case118": pn.case118, "case300": pn.case300,
          "case30": pn.case30, "case57": pn.case57}.get(name)
    if fn is None:
        raise SystemExit(f"unknown case '{name}'")
    return fn()


def decision_space_from_case(case_name: str, n_ppc: int, n_branch: int,
                             vmin: float, vmax: float,
                             max_loading: float = 100.0) -> Dict[str, np.ndarray]:
    """Per-PPC-bus generation limits / cost, and per-branch ratings."""
    import pandapower as pp

    net = load_case(case_name)
    pp.runpp(net, init="flat", max_iteration=1, tolerance_mva=1e9,
             calculate_voltage_angles=True, enforce_q_lims=False)
    ppc = net._ppc["internal"]
    lookup = np.asarray(net._pd2ppc_lookups["bus"])
    N = int(ppc["bus"].shape[0])
    if N != n_ppc:
        raise SystemExit(f"case {case_name} compiles to {N} PPC buses, parquet has {n_ppc}")

    base_mva = float(net.sn_mva)
    p_min = np.zeros(N); p_max = np.zeros(N)
    q_min = np.zeros(N); q_max = np.zeros(N)
    cost = np.zeros(N); ctrl = np.zeros(N, dtype=np.int8)

    def add(df, cost_lookup):
        if df is None or not len(df):
            return
        for k in df.index:
            j = int(lookup[int(df.at[k, "bus"])])
            if not (0 <= j < N):
                continue
            ctrl[j] = 1
            for col, tgt in (("min_p_mw", p_min), ("max_p_mw", p_max),
                             ("min_q_mvar", q_min), ("max_q_mvar", q_max)):
                if col in df.columns and np.isfinite(df.at[k, col]):
                    tgt[j] += float(df.at[k, col]) * 1e6
            c = cost_lookup.get(k, 0.0)
            if c:
                cost[j] = c

    def poly_costs(et):
        out = {}
        if not hasattr(net, "poly_cost") or not len(net.poly_cost):
            return out
        sel = net.poly_cost[net.poly_cost.et == et]
        for _, r in sel.iterrows():
            out[int(r.element)] = float(r.get("cp1_eur_per_mw", 0.0))
        return out

    add(net.gen, poly_costs("gen"))
    add(net.sgen, poly_costs("sgen"))
    add(net.ext_grid, poly_costs("ext_grid"))

    # Branch ratings, mapped through pandapower's own branch lookup.
    rate = np.zeros(n_branch)
    bl = net._pd2ppc_lookups.get("branch", {}) or {}
    vb = np.asarray(ppc["bus"][:, 9], dtype=float) * 1e3  # BASE_KV
    fbus = np.asarray(ppc["branch"][:, 0].real, dtype=int)
    if "line" in bl and len(net.line) and "max_i_ka" in net.line.columns:
        s, e = (int(v) for v in bl["line"])
        mi = np.nan_to_num(np.asarray(net.line["max_i_ka"], dtype=float), nan=0.0)
        for k in range(min(e - s, len(mi), n_branch - s)):
            j = int(fbus[s + k])
            if 0 <= j < N:
                rate[s + k] = np.sqrt(3.0) * vb[j] * mi[k] * 1e3 * max_loading / 100.0
    for table in ("trafo", "trafo3w"):
        if table in bl and len(net[table]) and "sn_mva" in net[table].columns:
            s, e = (int(v) for v in bl[table])
            sn = np.nan_to_num(np.asarray(net[table]["sn_mva"], dtype=float), nan=0.0)
            for k in range(min(e - s, len(sn), n_branch - s)):
                rate[s + k] = sn[k] * 1e6 * max_loading / 100.0

    return {
        "Gen_p_min": p_min, "Gen_p_max": p_max,
        "Gen_q_min": q_min, "Gen_q_max": q_max,
        "Gen_p_avail": p_max.copy(),
        "Gen_cost_c1": cost,
        "Gen_controllable": ctrl,
        "Bus_vmin": np.full(N, float(vmin)),
        "Bus_vmax": np.full(N, float(vmax)),
        "Branch_rate_a": rate,
        "_base_mva": np.array([base_mva]),
    }


def empirical_envelope(src: pq.ParquetFile, n_ppc: int, s_base: float,
                       n_sample: int = 2000, margin: float = 0.10,
                       tol_pu: float = 1e-3,
                       box: str = "injection") -> Dict[str, np.ndarray]:
    """Per-bus decision envelope measured from the dataset itself.

    The stock pandapower limits do not match the OPF that produced these files:
    the stored solution violates them. Rather than impose limits the ground
    truth breaks, take the observed range per bus and widen it. This makes the
    reference solution feasible by construction, at the cost of the box being
    the observed range rather than the true one.

    Which quantity is bounded matters. `S_demand` in these files is not the load
    but the *pre-OPF injection*: at case118's pure-generator buses it equals the
    base generator setpoint exactly (bus 9: 450.00 MW). So `S_opf - S_demand` is
    a redispatch delta, not a generation, and its range is dominated by the
    slack absorbing the system-wide correction -- on case118 that gives a box
    midpoint of -28 pu, i.e. -2800 MW, which drives GridSFM's DC angle prior to
    a degenerate state and kills its gradients.

    The load cannot be separated out (case118's stored base generation does not
    match stock pandapower, so it cannot be subtracted), but the post-OPF
    injection is directly observable and is a physically meaningful bounded
    quantity: free within a range at controllable buses, fixed by load
    elsewhere. box="injection" bounds it; box="delta" keeps the old behaviour.
    """
    if box not in ("injection", "delta"):
        raise ValueError(f"box must be injection|delta, got {box!r}")
    n_groups = src.metadata.num_row_groups
    step = max(1, n_groups // 12)
    p_lo = np.full(n_ppc, np.inf); p_hi = np.full(n_ppc, -np.inf)
    q_lo = np.full(n_ppc, np.inf); q_hi = np.full(n_ppc, -np.inf)
    # Controllability is a separate question from what the box bounds: a bus is
    # controllable iff the OPF actually moves its injection. Testing |S_opf|
    # instead would flag every load bus, since a load bus has a large injection
    # that the OPF never changes -- and that would turn its power-balance
    # equality into a free variable.
    d_hi = np.zeros(n_ppc)
    seen = 0
    for g in range(0, n_groups, step):
        d = src.read_row_group(g).to_pydict()
        n = len(d["bus_number"])
        stride = max(1, n // max(1, n_sample // 12))
        for i in range(0, n, stride):
            sd = np.load(io.BytesIO(d["S_demand"][i]), allow_pickle=False)
            so = np.load(io.BytesIO(d["S_opf"][i]), allow_pickle=False)
            delta = (so - sd) / s_base
            gen = (so / s_base) if box == "injection" else delta
            p_lo = np.minimum(p_lo, gen.real); p_hi = np.maximum(p_hi, gen.real)
            q_lo = np.minimum(q_lo, gen.imag); q_hi = np.maximum(q_hi, gen.imag)
            d_hi = np.maximum(d_hi, np.maximum(np.abs(delta.real), np.abs(delta.imag)))
            seen += 1
    span_p = np.maximum(p_hi - p_lo, 1e-9)
    span_q = np.maximum(q_hi - q_lo, 1e-9)
    ctrl = d_hi > tol_pu
    p_min = np.where(ctrl, p_lo - margin * span_p, 0.0) * s_base
    p_max = np.where(ctrl, p_hi + margin * span_p, 0.0) * s_base
    q_min = np.where(ctrl, q_lo - margin * span_q, 0.0) * s_base
    q_max = np.where(ctrl, q_hi + margin * span_q, 0.0) * s_base
    print(f"[convert] empirical envelope on {box} from {seen} scenarios: "
          f"{int(ctrl.sum())} dispatching buses (tol {tol_pu:g} pu, margin {margin:.0%}); "
          f"P box [{p_min.min()/s_base:.3f}, {p_max.max()/s_base:.3f}] pu, "
          f"midpoint range [{(0.5*(p_min+p_max)/s_base).min():.3f}, "
          f"{(0.5*(p_min+p_max)/s_base).max():.3f}] pu")
    return {"p_min": p_min, "p_max": p_max, "q_min": q_min, "q_max": q_max,
            "ctrl": ctrl.astype(np.int8)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--vmin", type=float, default=0.9)
    p.add_argument("--vmax", type=float, default=1.1)
    p.add_argument("--max_loading_percent", type=float, default=100.0)
    p.add_argument("--row_group_size", type=int, default=20)
    p.add_argument("--load_sign", choices=("demand_positive", "injection"),
                   default="injection",
                   help="injection: S_demand is already a signed injection (verified for the "
                        "opf_backbone files: at PQ buses S_demand == S_opf exactly).")
    p.add_argument("--limits", choices=("empirical", "case"), default="empirical",
                   help="empirical: envelope measured from the data, so the reference "
                        "solution is feasible. case: stock pandapower limits, which the "
                        "stored solutions violate.")
    p.add_argument("--limit_margin", type=float, default=0.10)
    p.add_argument("--gen_box", choices=("injection", "delta"), default="injection",
                   help="Quantity the generation box bounds. injection: the post-OPF "
                        "injection S_opf, which is observable and physically bounded. "
                        "delta: the redispatch S_opf - S_demand, which on case118 is "
                        "dominated by the slack correction and is not a generation level.")
    p.add_argument("--normalize_angle_reference", action="store_true",
                   help="Rotate every voltage so the slack bus sits at 0 deg. The angle "
                        "datum is a gauge choice: S = V conj(Y V) is exactly invariant "
                        "under a global phase rotation, so this changes no physics. "
                        "pandapower's case118 puts its slack at +30 deg, which penalises "
                        "any model that cannot observe the bus angle state (LUMINA's bus "
                        "schema has no such input) for a purely arbitrary reference.")
    return p.parse_args()


def main():
    args = parse_args()
    src = pq.ParquetFile(args.src)
    n_rows = src.metadata.num_rows
    names = list(src.schema.names)
    print(f"[convert] {os.path.basename(args.src)}: {n_rows} rows, "
          f"{src.metadata.num_row_groups} groups ({n_rows//src.metadata.num_row_groups}/group)")

    head = src.read_row_group(0).to_pydict()
    n_ppc = int(head["bus_number"][0])
    n_branch = int(head["branch_number"][0])
    space = decision_space_from_case(args.case, n_ppc, n_branch,
                                     args.vmin, args.vmax, args.max_loading_percent)
    if args.limits == "empirical":
        env = empirical_envelope(src, n_ppc, float(head["S_base"][0]),
                                 margin=args.limit_margin, box=args.gen_box)
        space["Gen_p_min"] = env["p_min"]; space["Gen_p_max"] = env["p_max"]
        space["Gen_q_min"] = env["q_min"]; space["Gen_q_max"] = env["q_max"]
        space["Gen_p_avail"] = env["p_max"].copy()
        space["Gen_controllable"] = env["ctrl"]
        space["Gen_cost_c1"] = np.where(env["ctrl"] > 0, space["Gen_cost_c1"], 0.0)
    space["Gen_box_injection"] = np.full(
        n_ppc, 1 if (args.limits == "empirical" and args.gen_box == "injection") else 0,
        dtype=np.int8)
    print(f"[convert] case {args.case}: {n_ppc} PPC buses, {n_branch} branches, "
          f"{int(space['Gen_controllable'].sum())} controllable buses, "
          f"cost slopes set on {int((space['Gen_cost_c1'] != 0).sum())} buses")

    # Sanity: the compiled case must match the parquet's own grid description.
    vn_parquet = np.load(io.BytesIO(head["vn_kv"][0]), allow_pickle=False)
    import pandapower as pp
    net = load_case(args.case)
    pp.runpp(net, init="flat", max_iteration=1, tolerance_mva=1e9,
             calculate_voltage_angles=True, enforce_q_lims=False)
    vn_case = np.asarray(net._ppc["internal"]["bus"][:, 9], dtype=float)
    agree = np.allclose(np.sort(vn_parquet), np.sort(vn_case), rtol=1e-6, atol=1e-6)
    print(f"[convert] vn_kv agrees between parquet and compiled case: {agree}")
    if not agree:
        raise SystemExit("PPC bus ordering mismatch; refusing to attach a decision space")

    out_fields = []
    for n in names:
        f = src.schema_arrow.field(n)
        out_fields.append(pa.field(RENAME.get(n, n), f.type))
    for n in DERIVED:
        out_fields.append(pa.field(n, pa.binary()))
    schema = pa.schema(out_fields)

    derived_cells = {n: npy_bytes(space[n].astype(
        np.int8 if n in ("Gen_controllable", "Gen_box_injection") else np.float64))
        for n in DERIVED}

    writer = pq.ParquetWriter(args.dst, schema, compression="zstd", use_dictionary=True)
    t0 = time.time()
    written = 0
    try:
        for g in range(src.metadata.num_row_groups):
            tbl = src.read_row_group(g)
            d = tbl.to_pydict()
            n = len(d["bus_number"])
            cols = {}
            rot = None
            if args.normalize_angle_reference:
                bt0 = np.load(io.BytesIO(d["bus_typ"][0]), allow_pickle=False)
                ref = np.where(np.asarray(bt0) == 1)[0]
                rot = []
                for i in range(n):
                    uo = np.load(io.BytesIO(d["u_opf"][i]), allow_pickle=False)
                    a = np.angle(uo[int(ref[0])]) if len(ref) else 0.0
                    rot.append(np.exp(-1j * a))
            for name in names:
                target = RENAME.get(name, name)
                if rot is not None and name in ("u_opf", "u_start"):
                    vals = []
                    for i in range(n):
                        a = np.load(io.BytesIO(d[name][i]), allow_pickle=False)
                        vals.append(npy_bytes(a * rot[i]))
                    cols[target] = vals
                elif name == "S_demand" and args.load_sign == "demand_positive":
                    vals = []
                    for i in range(n):
                        a = np.load(io.BytesIO(d[name][i]), allow_pickle=False)
                        vals.append(npy_bytes(-a))
                    cols[target] = vals
                else:
                    cols[target] = d[name]
            for name in DERIVED:
                cols[name] = [derived_cells[name]] * n
            arrays = [pa.array(cols[f.name], type=f.type) for f in schema]
            writer.write_table(pa.Table.from_arrays(arrays, schema=schema),
                               row_group_size=args.row_group_size)
            written += n
            if (g + 1) % 5 == 0 or g + 1 == src.metadata.num_row_groups:
                print(f"[convert] {written}/{n_rows} rows ({time.time()-t0:.0f}s)", flush=True)
    finally:
        writer.close()

    chk = pq.ParquetFile(args.dst)
    print(f"[convert] wrote {args.dst}: rows={chk.metadata.num_rows} "
          f"row_groups={chk.metadata.num_row_groups} "
          f"({chk.metadata.num_rows//chk.metadata.num_row_groups}/group) "
          f"cols={chk.metadata.num_columns} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
