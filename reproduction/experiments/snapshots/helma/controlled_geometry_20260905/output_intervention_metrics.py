#!/usr/bin/env python3
"""One-forward-pass test table for Raw, C, P_k, and CSP_k output states.

The rank is supplied by the caller only after validation selection.  Calibration
and both magnitude/angle bases use training data only.  PB is recomputed from
complex128 voltage states and Max PB is the maximum individual bus residual.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from collate_blockdiag_optimized_complex_columns import collate_blockdiag
from controlled_error_geometry import (
    calibrated_forward, make_model, pb_per_scenario, split_dataset,
)
from diagnose_residual_distributions import angle_diff, fit_per_bus_offset, manifold_basis
from manifold_projection import project_state


def train_per_bus_mean(train, batch_size, device):
    """Training-only per-bus mean state with circular phase averaging."""
    loader = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate_blockdiag)
    v_sum = theta_sin_sum = theta_cos_sum = None
    count = 0
    for batch in loader:
        sizes = batch["sizes"].numpy().astype(int)
        if len(set(sizes.tolist())) != 1:
            raise ValueError("The per-bus constant baseline requires a common topology.")
        nbus, ns = int(sizes[0]), len(sizes)
        ref = batch["V_newton"].to(device=device, dtype=torch.float64)[0].reshape(ns, nbus, 2)
        if v_sum is None:
            v_sum = torch.zeros(nbus, dtype=torch.float64, device=device)
            theta_sin_sum = torch.zeros_like(v_sum)
            theta_cos_sum = torch.zeros_like(v_sum)
        v_sum += ref[..., 0].sum(0)
        theta_sin_sum += ref[..., 1].sin().sum(0)
        theta_cos_sum += ref[..., 1].cos().sum(0)
        count += ns
    return torch.stack((v_sum / count, torch.atan2(theta_sin_sum, theta_cos_sum)), dim=-1)


def evaluate(kind, checkpoint, train, test, parquet, batch_size, rank, device):
    if kind == "constant":
        mean_state = train_per_bus_mean(train, batch_size, device)

        def forward(batch):
            ns = len(batch["sizes"])
            return mean_state.repeat(ns, 1).unsqueeze(0)

        model = None
    else:
        model, forward = make_model(kind, checkpoint, parquet, batch_size, device)
    offset = fit_per_bus_offset(model, forward, train, batch_size, device)
    Uv, Ut, vbar, tbar, _ev, _et = manifold_basis(train, batch_size, k=rank)
    basis = (Uv[:, :rank].to(device=device, dtype=torch.float64),
             Ut[:, :rank].to(device=device, dtype=torch.float64), vbar, tbar)
    loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate_blockdiag)
    rec = {name: {"pb": [], "max_pb": 0.0, "mag_sse": 0.0, "ang_sse": 0.0,
                  "n": 0} for name in ("raw", "C", f"P{rank}", f"CSP{rank}")}
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            raw = forward(batch).detach().to(torch.float64)
            cal = calibrated_forward(forward=lambda _unused: raw, offset=offset, batch=batch)
            sizes = batch["sizes"].numpy().astype(int); nbus = int(sizes[0]); ns = len(sizes)
            ref = batch["V_newton"].to(device=device, dtype=torch.float64)[0].reshape(ns, nbus, 2)
            raw = raw[0].reshape(ns, nbus, 2)
            cal = cal[0].reshape(ns, nbus, 2)
            pv_raw, pt_raw = project_state(raw[..., 0], raw[..., 1], basis)
            pv_cal, pt_cal = project_state(cal[..., 0], cal[..., 1], basis)
            states = {
                "raw": (raw[..., 0], raw[..., 1]),
                "C": (cal[..., 0], cal[..., 1]),
                f"P{rank}": (pv_raw, pt_raw),
                f"CSP{rank}": (pv_cal, pt_cal),
            }
            for name, (v, theta) in states.items():
                per_scenario, max_pb = pb_per_scenario(v, theta, batch, device, return_global_max=True)
                r = rec[name]
                r["pb"].append(per_scenario); r["max_pb"] = max(r["max_pb"], max_pb)
                r["mag_sse"] += float((v - ref[..., 0]).square().sum())
                r["ang_sse"] += float(angle_diff(theta, ref[..., 1]).square().sum())
                r["n"] += int(v.numel())
                # Per-bus-centred regression sufficient statistics.  They
                # reproduce the within-bus slope a_w and Pearson R_w^2
                # without retaining all test predictions in host memory.
                if "sum_p" not in r:
                    r.update(sum_p=torch.zeros(nbus, dtype=torch.float64, device=device),
                             sum_r=torch.zeros(nbus, dtype=torch.float64, device=device),
                             sum_p2=torch.zeros(nbus, dtype=torch.float64, device=device),
                             sum_r2=torch.zeros(nbus, dtype=torch.float64, device=device),
                             sum_pr=torch.zeros(nbus, dtype=torch.float64, device=device),
                             n_scen=0)
                r["sum_p"] += v.sum(0); r["sum_r"] += ref[..., 0].sum(0)
                r["sum_p2"] += v.square().sum(0); r["sum_r2"] += ref[..., 0].square().sum(0)
                r["sum_pr"] += (v * ref[..., 0]).sum(0); r["n_scen"] += ns
            if bi and bi % 25 == 0:
                print(f"[{kind}] evaluated {bi}/{len(loader)} batches", flush=True)
    results = {}
    for name, r in rec.items():
        pb = torch.cat([torch.from_numpy(x) for x in r["pb"]]).numpy()
        ns = r["n_scen"]
        var_p = (r["sum_p2"] - r["sum_p"].square() / ns).sum().item()
        var_r = (r["sum_r2"] - r["sum_r"].square() / ns).sum().item()
        cov = (r["sum_pr"] - r["sum_p"] * r["sum_r"] / ns).sum().item()
        aw = cov / var_r if var_r > 0 else float("nan")
        rw2 = cov * cov / (var_p * var_r) if var_p > 0 and var_r > 0 else 0.0
        results[name] = {
            "vmag_rmse": math.sqrt(r["mag_sse"] / r["n"]),
            "angle_rmse_deg": math.degrees(math.sqrt(r["ang_sse"] / r["n"])),
            "mean_pb": float(pb.mean()),
            "max_pb": r["max_pb"],
            "within_r2": rw2,
            "within_slope": aw,
        }
    return {"model": kind, "rank": rank, "n_scenarios": len(test),
            "calibration_offset_rms_vmag": float(offset[0].square().mean().sqrt()),
            "metrics": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("constant", "g3", "pignn_base", "gridsfm", "graphkit", "lumina"), required=True)
    ap.add_argument("--checkpoint", default=""); ap.add_argument("--parquet", required=True)
    ap.add_argument("--rank", type=int, required=True); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    train, _valid, test = split_dataset(args.parquet)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {"protocol": {"split": "random_split seed=42",
                        "rank": args.rank, "calibration": "train-only per-bus magnitude and circular-angle offsets",
                        "pb": "complex128 GENCO-style per-bus norm, all bus-scenario pairs",
                        "max_pb": "maximum individual bus PB over the test split",
                        "split_seed": 42},
           "result": evaluate(args.model, args.checkpoint, train, test, args.parquet,
                               args.batch, args.rank, device)}
    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
