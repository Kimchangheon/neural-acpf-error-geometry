#!/usr/bin/env python3
"""Seed-42 GBnetwork off-subspace error and conditional-association audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader

from collate_blockdiag_optimized_complex_columns import collate_blockdiag
from controlled_error_geometry import calibrated_forward, make_model, pb_per_scenario, split_dataset
from diagnose_residual_distributions import angle_diff, fit_per_bus_offset, manifold_basis


def partial_spearman(x, y, z):
    """Spearman partial correlation via OLS residuals of ranked x/y on ranked z."""
    rx, ry, rz = (rankdata(np.asarray(a), method="average") for a in (x, y, z))
    design = np.column_stack((np.ones_like(rz), rz))
    ex = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    ey = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    den = np.linalg.norm(ex) * np.linalg.norm(ey)
    return float(ex @ ey / den) if den else float("nan")


def bootstrap_ci(x, y, z, draws, seed):
    rng = np.random.default_rng(seed)
    n = len(x)
    values = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        ix = rng.integers(0, n, n)
        values[i] = partial_spearman(x[ix], y[ix], z[ix])
    return [float(v) for v in np.quantile(values[np.isfinite(values)], [.025, .975])]


def describe(x):
    q = np.quantile(x, [.25, .5, .75])
    return {"q25": float(q[0]), "median": float(q[1]), "q75": float(q[2]),
            "mean": float(np.mean(x)), "min": float(np.min(x)), "max": float(np.max(x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--kind", choices=("g3", "gridsfm", "graphkit", "lumina"), required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--lumina-args", default="")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260914)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()
    if args.rank != 16:
        raise ValueError("The manuscript protocol is frozen at k=16.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, _validation, test = split_dataset(args.parquet)
    Uv, Ut, _vbar, _tbar, ev_v, ev_t = manifold_basis(train, args.batch, k=args.rank)
    Uv = Uv.to(device=device, dtype=torch.float64)
    Ut = Ut.to(device=device, dtype=torch.float64)
    model, forward = make_model(args.kind, args.checkpoint, args.parquet, args.batch,
                                device, lumina_args=args.lumina_args)
    offset = fit_per_bus_offset(model, forward, train, args.batch, device)

    arrays = {k: [] for k in ("qperp_v", "qperp_theta", "norm_v", "norm_theta", "mean_pb")}
    zero_v = zero_t = 0
    loader = DataLoader(test, batch_size=args.batch, shuffle=False, num_workers=0,
                        collate_fn=collate_blockdiag)
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            sizes = batch["sizes"].numpy().astype(int)
            if len(set(map(int, sizes))) != 1:
                raise RuntimeError("Expected one fixed GBnetwork topology per batch")
            ns, nb = len(sizes), int(sizes[0])
            ref = batch["V_newton"].to(device=device, dtype=torch.float64)[0].reshape(ns, nb, 2)
            cal = calibrated_forward(forward, offset, batch)[0].reshape(ns, nb, 2)
            ev = cal[..., 0] - ref[..., 0]
            et = angle_diff(cal[..., 1], ref[..., 1])
            off_v = ev - (ev @ Uv) @ Uv.T
            off_t = et - (et @ Ut) @ Ut.T
            den_v = ev.square().sum(1)
            den_t = et.square().sum(1)
            zero_v += int((den_v == 0).sum())
            zero_t += int((den_t == 0).sum())
            qv = torch.where(den_v > 0, off_v.square().sum(1) / den_v, torch.nan)
            qt = torch.where(den_t > 0, off_t.square().sum(1) / den_t, torch.nan)
            arrays["qperp_v"].append(qv.cpu().numpy())
            arrays["qperp_theta"].append(qt.cpu().numpy())
            arrays["norm_v"].append(torch.sqrt(den_v).cpu().numpy())
            arrays["norm_theta"].append(torch.sqrt(den_t).cpu().numpy())
            arrays["mean_pb"].append(pb_per_scenario(cal[..., 0], cal[..., 1], batch, device))
            if bi and bi % 100 == 0:
                print(f"[{args.name}] {bi}/{len(loader)} test batches", flush=True)

    data = {k: np.concatenate(v) for k, v in arrays.items()}
    valid_v = np.isfinite(data["qperp_v"]) & np.isfinite(data["mean_pb"]) & np.isfinite(data["norm_v"])
    valid_t = np.isfinite(data["qperp_theta"]) & np.isfinite(data["mean_pb"]) & np.isfinite(data["norm_theta"])
    assoc = {}
    for block, valid in (("magnitude", valid_v), ("angle", valid_t)):
        q = data["qperp_v" if block == "magnitude" else "qperp_theta"][valid]
        norm = data["norm_v" if block == "magnitude" else "norm_theta"][valid]
        pb = data["mean_pb"][valid]
        rho = partial_spearman(q, pb, norm)
        assoc[block] = {
            "partial_spearman_qperp_mean_pb_given_error_l2": rho,
            "paired_scenario_bootstrap_95_ci": bootstrap_ci(q, pb, norm, args.bootstrap,
                                                              args.bootstrap_seed + (block == "angle")),
            "n": int(len(q)),
        }
    result = {
        "protocol": {
            "model": args.name, "kind": args.kind, "checkpoint": args.checkpoint,
            "split": "controlled_error_geometry.split_dataset; seed=42 fixed memberships",
            "calibration": "fit_per_bus_offset on training predictions only; circular angle offset",
            "basis": "manifold_basis on training NR labels only; separate magnitude/circular-angle bases",
            "rank": args.rank, "actual_rank_magnitude": int(Uv.shape[1]),
            "actual_rank_angle": int(Ut.shape[1]),
            "pb": "pb_per_scenario; final calibrated complex128 voltage; GENCO structural-zero convention",
            "partial_spearman": "Pearson correlation of rank residuals after separately regressing ranked q_perp and ranked Mean PB on intercept + ranked scenario error L2",
            "bootstrap": {"draws": args.bootstrap, "seed": args.bootstrap_seed,
                           "unit": "held-out test scenario", "percentile_ci": [.025, .975]},
            "test_scenarios": int(len(data["mean_pb"])), "zero_error_denominator_magnitude": zero_v,
            "zero_error_denominator_angle": zero_t,
        },
        "training_explained_variance_at_16": {"magnitude": float(ev_v), "angle": float(ev_t)},
        "per_scenario_off_subspace_error_fraction": {
            "magnitude": describe(data["qperp_v"][valid_v]),
            "angle": describe(data["qperp_theta"][valid_t]),
        },
        "conditional_association": assoc,
        "audit": {
            "ratio_of_summed_energy_magnitude": float(np.nansum(data["qperp_v"] * data["norm_v"] ** 2) / np.sum(data["norm_v"] ** 2)),
            "ratio_of_summed_energy_angle": float(np.nansum(data["qperp_theta"] * data["norm_theta"] ** 2) / np.sum(data["norm_theta"] ** 2)),
            "calibrated_mean_pb": float(np.mean(data["mean_pb"])),
        },
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(result, indent=2, allow_nan=False))
    np.savez_compressed(args.out_npz, **data)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
