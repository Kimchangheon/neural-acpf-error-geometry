#!/usr/bin/env python3
"""Validation-selected scalar shrinkage baseline for calibrated PIGNN states.

Stage ``validation`` never loads a test batch.  It selects lambda from a
predeclared grid by validation Mean PB, subject to not worsening validation
|V| RMSE relative to calibrated prediction.  Stage ``test`` accepts only the
already-recorded lambda and scores the frozen intervention.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from collate_blockdiag_optimized_complex_columns import collate_blockdiag
from controlled_error_geometry import calibrated_forward, make_model, pb_per_scenario, split_dataset
from diagnose_residual_distributions import angle_diff, fit_per_bus_offset, manifold_basis
from pf_known_mask import apply_known_v

LAMBDAS = tuple(i / 8 for i in range(9))


def restore_known(v, theta, batch, nbus):
    packed = torch.stack((v, theta), dim=-1)
    held = apply_known_v(packed, batch["bus_type"][0].reshape(-1)[:nbus],
                         batch["V_start"][0].reshape(-1, 2))
    return held[..., 0], held[..., 1]


def shrink(v, theta, vbar, tbar, lam):
    vp = vbar + lam * (v - vbar)
    dt = torch.atan2(torch.sin(theta - tbar), torch.cos(theta - tbar))
    tp = tbar + lam * dt
    return vp, torch.atan2(torch.sin(tp), torch.cos(tp))


def new_record():
    return {"pb": [], "max_pb": 0., "mag_sse": 0., "ang_sse": 0., "n": 0,
            "sum_p": None, "sum_r": None, "sum_p2": None, "sum_r2": None,
            "sum_pr": None, "n_scen": 0}


def update(r, v, theta, ref, batch, device):
    per, mx = pb_per_scenario(v, theta, batch, device, return_global_max=True)
    r["pb"].append(per); r["max_pb"] = max(r["max_pb"], mx)
    r["mag_sse"] += float((v - ref[..., 0]).square().sum())
    r["ang_sse"] += float(angle_diff(theta, ref[..., 1]).square().sum())
    r["n"] += int(v.numel()); nbus, ns = v.shape[1], v.shape[0]
    if r["sum_p"] is None:
        z = lambda: torch.zeros(nbus, dtype=torch.float64, device=device)
        r.update(sum_p=z(), sum_r=z(), sum_p2=z(), sum_r2=z(), sum_pr=z())
    r["sum_p"] += v.sum(0); r["sum_r"] += ref[..., 0].sum(0)
    r["sum_p2"] += v.square().sum(0); r["sum_r2"] += ref[..., 0].square().sum(0)
    r["sum_pr"] += (v * ref[..., 0]).sum(0); r["n_scen"] += ns


def finalize(r):
    pb = torch.cat([torch.from_numpy(x) for x in r["pb"]]).numpy(); ns = r["n_scen"]
    var_p = (r["sum_p2"] - r["sum_p"].square() / ns).sum().item()
    var_r = (r["sum_r2"] - r["sum_r"].square() / ns).sum().item()
    cov = (r["sum_pr"] - r["sum_p"] * r["sum_r"] / ns).sum().item()
    return {"vmag_rmse": math.sqrt(r["mag_sse"] / r["n"]),
            "angle_rmse_deg": math.degrees(math.sqrt(r["ang_sse"] / r["n"])),
            "within_r2": cov * cov / (var_p * var_r) if var_p > 0 and var_r > 0 else 0.,
            "within_slope": cov / var_r if var_r > 0 else float("nan"),
            "mean_pb": float(pb.mean()), "max_pb": r["max_pb"]}


def score_subset(forward, offset, subset, batch_size, device, vbar, tbar, lambdas):
    records = {"raw": new_record(), "C": new_record(),
               **{f"shrink_{lam:g}": new_record() for lam in lambdas}}
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate_blockdiag)
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            raw_packed = forward(batch).detach().to(torch.float64)
            cal_packed = calibrated_forward(lambda _unused: raw_packed, offset, batch)
            sizes = batch["sizes"].numpy().astype(int)
            if len(set(sizes.tolist())) != 1:
                raise ValueError("Scalar shrinkage baseline requires a common topology.")
            nbus, ns = int(sizes[0]), len(sizes)
            ref = batch["V_newton"].to(device=device, dtype=torch.float64)[0].reshape(ns, nbus, 2)
            raw = raw_packed[0].reshape(ns, nbus, 2); cal = cal_packed[0].reshape(ns, nbus, 2)
            # The held C state is the lambda=1 reference and enforces the same
            # PF specified variables as the shrinkage candidates.
            raw_v, raw_t = restore_known(raw[..., 0], raw[..., 1], batch, nbus)
            cal_v, cal_t = restore_known(cal[..., 0], cal[..., 1], batch, nbus)
            update(records["raw"], raw_v, raw_t, ref, batch, device)
            update(records["C"], cal_v, cal_t, ref, batch, device)
            for lam in lambdas:
                v, t = shrink(cal[..., 0], cal[..., 1], vbar, tbar, lam)
                v, t = restore_known(v, t, batch, nbus)
                update(records[f"shrink_{lam:g}"], v, t, ref, batch, device)
            if bi and bi % 25 == 0:
                print(f"[shrinkage] scored {bi}/{len(loader)} batches", flush=True)
    return {key: finalize(value) for key, value in records.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("validation", "test"))
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--parquet", required=True)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--lambda-fixed", type=float)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.stage == "test" and args.lambda_fixed is None:
        raise ValueError("test stage requires --lambda-fixed selected from validation")
    if args.lambda_fixed is not None and args.lambda_fixed not in LAMBDAS:
        raise ValueError(f"lambda-fixed must be in {LAMBDAS}")
    train, valid, test = split_dataset(args.parquet)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, forward = make_model("g3", args.checkpoint, args.parquet, args.batch, device)
    offset = fit_per_bus_offset(model, forward, train, args.batch, device)
    _uv, _ut, vbar, tbar, _ev, _et = manifold_basis(train, args.batch, k=1)
    vbar, tbar = vbar.to(device), tbar.to(device)
    if args.stage == "validation":
        metrics = score_subset(forward, offset, valid, args.batch, device, vbar, tbar, LAMBDAS)
        c_rmse = metrics["C"]["vmag_rmse"]
        admissible = [lam for lam in LAMBDAS if metrics[f"shrink_{lam:g}"]["vmag_rmse"] <= c_rmse]
        if not admissible:
            raise RuntimeError("No admissible lambda; test stage must not be launched")
        selected = min(admissible, key=lambda lam: metrics[f"shrink_{lam:g}"]["mean_pb"])
        payload = {"protocol": {"stage": "validation-only", "split": "random_split seed=42",
                                  "calibration": "train-only", "lambdas": list(LAMBDAS),
                                  "selection": "minimize validation Mean PB subject to |V| RMSE <= calibrated |V| RMSE",
                                  "known_variables": "restore PV/slack magnitude and slack angle from V_start"},
                   "n_validation_scenarios": len(valid), "metrics": metrics,
                   "calibrated_vmag_rmse": c_rmse, "admissible_lambdas": admissible,
                   "selected_lambda": selected}
    else:
        metrics = score_subset(forward, offset, test, args.batch, device, vbar, tbar,
                               (args.lambda_fixed,))
        payload = {"protocol": {"stage": "test-only after frozen validation selection",
                                  "split": "random_split seed=42", "calibration": "train-only",
                                  "lambda_fixed": args.lambda_fixed,
                                  "known_variables": "restore PV/slack magnitude and slack angle from V_start"},
                   "n_test_scenarios": len(test), "metrics": metrics}
    Path(args.out).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
