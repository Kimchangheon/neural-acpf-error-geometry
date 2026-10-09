"""Run GridSFM predict() on a .pyg.json sample and emit a viewer-compatible
*_gridsfm_results.json that mirrors the schema of *_dc_results.json /
*_ac_results.json from power_grid_data_release.

Usage (from GridSFM-Staging/model/):
    .venv/bin/python examples/predict_to_viewer.py \
        --sample samples/msr_texas.pyg.json \
        --out    ../../power_grid_data_release/16h/texas_gridsfm_results.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from gridsfm import load_model, predict
from gridsfm.schema import GEN_CP0_IDX, GEN_CP1_IDX, GEN_CP2_IDX

ROOT = Path(__file__).parent.parent


def build_viewer_json(pred: dict, sample_obj: dict, t_solve: float) -> dict:
    md = sample_obj["metadata"]
    bus_id_map = md["bus_id_map"]                # row -> PM bus id
    gen_id_map = md["gen_id_map"]                # row -> PM gen id
    ac_ids     = md.get("ac_line_branch_ids", [])
    tr_ids     = md.get("transformer_branch_ids", [])

    gen_x = torch.tensor(sample_obj["grid"]["nodes"]["generator"], dtype=torch.float32)
    c2 = gen_x[:, GEN_CP2_IDX]
    c1 = gen_x[:, GEN_CP1_IDX]
    c0 = gen_x[:, GEN_CP0_IDX]
    Pg = pred["Pg"]
    pg_costs = (c2 * Pg ** 2 + c1 * Pg + c0).tolist()
    objective = float(sum(pg_costs))

    # bus solution
    bus_sol = {}
    V     = pred["V"].tolist()
    theta = pred["theta"].tolist()
    for row, pm_id in enumerate(bus_id_map):
        bus_sol[str(pm_id)] = {"vm": V[row], "va": theta[row]}

    # gen solution
    gen_sol = {}
    Qg = pred["Qg"].tolist()
    Pg_l = Pg.tolist()
    for row, pm_id in enumerate(gen_id_map):
        gen_sol[str(pm_id)] = {
            "pg": Pg_l[row],
            "qg": Qg[row],
            "pg_cost": pg_costs[row],
        }

    # branch solution: predict() concatenates flow tensors in the order given by
    # pred['flow_edge_types'] / pred['flow_edge_counts']. Build the matching ID
    # list from metadata so a future change in concat order is caught loudly
    # rather than silently mis-mapping per-branch flows.
    n_ac = len(ac_ids)
    n_tr = len(tr_ids)
    branch_sol = {}
    Pij = pred["Pij"].tolist(); Qij = pred["Qij"].tolist()
    Pji = pred["Pji"].tolist(); Qji = pred["Qji"].tolist()
    counts = list(pred["flow_edge_counts"])
    types  = list(pred["flow_edge_types"])
    ID_BY_TYPE = {"ac_line": list(ac_ids), "transformer": list(tr_ids)}
    EXPECTED   = {"ac_line": n_ac, "transformer": n_tr}
    if len(types) != len(counts):
        raise ValueError(f"flow_edge_types={types} length differs from flow_edge_counts={counts}")
    all_ids = []
    for t, c in zip(types, counts):
        if t not in ID_BY_TYPE:
            raise ValueError(f"Unknown flow_edge_type {t!r}; sample metadata has ac_line+transformer only")
        if c != EXPECTED[t]:
            raise ValueError(f"flow_edge_counts[{t}]={c} does not match metadata ({EXPECTED[t]})")
        all_ids.extend(ID_BY_TYPE[t])
    if len(all_ids) != len(Pij):
        raise ValueError(f"Flow count mismatch: {len(all_ids)} branch ids vs {len(Pij)} predicted flows")
    for i, pm_id in enumerate(all_ids):
        branch_sol[str(pm_id)] = {
            "pf": Pij[i], "qf": Qij[i],
            "pt": Pji[i], "qt": Qji[i],
        }

    # totals (in MW; predict outputs are per-unit, so multiply by baseMVA=100 by convention)
    baseMVA = 100.0
    total_gen_mw  = float(Pg.sum().item()) * baseMVA
    total_load_mw = sum(row[0] for row in sample_obj["grid"]["nodes"]["load"]) * baseMVA

    return {
        "formulation": "gridsfm",
        "relaxation_name": "GridSFM-Open v1.1",
        "relaxation_label": "ML",
        "relaxation_level": -1,
        "termination_status": "ML_PREDICTION",
        "objective": objective,
        "solve_time": t_solve,
        "n_buses":    len(bus_id_map),
        "n_gens":     len(gen_id_map),
        "n_branches": n_ac + n_tr,
        "n_loads":    len(sample_obj["grid"]["nodes"]["load"]),
        "n_shunts":   len(sample_obj["grid"]["nodes"].get("shunt", [])),
        "n_decommitted": 0,
        "total_gen_mw":  total_gen_mw,
        "total_load_mw": total_load_mw,
        "feas_score":  pred["feas"],
        "solution": {
            "baseMVA": baseMVA,
            "per_unit": True,
            "multinetwork": False,
            "multiinfrastructure": False,
            "bus":    bus_sol,
            "gen":    gen_sol,
            "branch": branch_sol,
            "dcline": {},
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True, type=Path)
    ap.add_argument("--out",    required=True, type=Path)
    ap.add_argument("--ckpt",   default=str(ROOT / "checkpoints" / "gridsfm_open_v1.1.pt"))
    ap.add_argument("--gpu",    type=int, default=-1)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device} ...")
    model = load_model(args.ckpt, device=device)

    print(f"Loading sample {args.sample}")
    with open(args.sample) as f:
        sample_obj = json.load(f)

    print("Running predict() ...")
    t0 = time.perf_counter()
    pred = predict(model, str(args.sample))
    t_solve = time.perf_counter() - t0
    print(f"  done in {t_solve:.2f}s  feas={pred['feas']:.3f}")

    out = build_viewer_json(pred, sample_obj, t_solve)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"Wrote {args.out}  "
          f"(buses={out['n_buses']} gens={out['n_gens']} branches={out['n_branches']} "
          f"objective={out['objective']:.2f})")


if __name__ == "__main__":
    main()
