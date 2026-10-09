#!/usr/bin/env python3
"""Frozen GBnetwork PIGNN CSP controls: random, graph spectral, and smoothing.

No test labels enter calibration, centring, SVD, topology basis construction,
or the filter.  The script is intentionally a post-training evaluator: one
PIGNN forward pass is made per batch, then all output-space interventions are
scored against the same final complex128 AC residual implementation.
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
from manifold_projection import project_state
from pf_known_mask import apply_known_v


def _topology_operators(train, nbus, device):
    """Return topology-only low-frequency basis and one-hop low-pass operator.

    A uses symmetrized off-diagonal admittance magnitudes.  L=I-D^-1/2 A
    D^-1/2 is the normalized weighted graph Laplacian.  No labels or test
    quantities are used here.
    """
    batch = next(iter(DataLoader(train, batch_size=1, shuffle=False,
                                 num_workers=0, collate_fn=collate_blockdiag)))
    y = batch["Ybus"]
    if y.is_sparse:
        y = y.coalesce().to_dense()
    y = y[:nbus, :nbus].to(dtype=torch.complex128, device=device)
    a = y.abs().to(torch.float64)
    a.fill_diagonal_(0)
    a = .5 * (a + a.T)
    d = a.sum(1)
    dinv = torch.where(d > 0, d.rsqrt(), torch.zeros_like(d))
    s = dinv[:, None] * a * dinv[None, :]
    lap = torch.eye(nbus, dtype=torch.float64, device=device) - s
    eigvals, eigvecs = torch.linalg.eigh(lap)
    return eigvecs, .5 * (torch.eye(nbus, dtype=torch.float64, device=device) + s), eigvals


def _random_basis(nbus, rank, seed, device):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(nbus, rank, generator=gen, dtype=torch.float64)
    return torch.linalg.qr(x, mode="reduced").Q.to(device)


def _restore_known(v, theta, batch, nbus):
    packed = torch.stack((v, theta), dim=-1)
    bt = batch["bus_type"][0].reshape(-1)[:nbus]
    # Keep scenario-specific setpoints if present.
    start = batch["V_start"][0].reshape(-1, 2)
    held = apply_known_v(packed, bt, start)
    return held[..., 0], held[..., 1]


def _lowpass(v, theta, vbar, tbar, h):
    dv = v - vbar
    dt = torch.atan2(torch.sin(theta - tbar), torch.cos(theta - tbar))
    vp = vbar + dv @ h.T
    tp = tbar + dt @ h.T
    return vp, torch.atan2(torch.sin(tp), torch.cos(tp))


def _new_record():
    return {"pb": [], "max_pb": 0.0, "mag_sse": 0.0, "ang_sse": 0.0,
            "n": 0, "sum_p": None, "sum_r": None, "sum_p2": None,
            "sum_r2": None, "sum_pr": None, "n_scen": 0}


def _update(record, v, theta, ref, batch, device):
    per_scen, max_pb = pb_per_scenario(v, theta, batch, device, return_global_max=True)
    record["pb"].append(per_scen); record["max_pb"] = max(record["max_pb"], max_pb)
    record["mag_sse"] += float((v - ref[..., 0]).square().sum())
    record["ang_sse"] += float(angle_diff(theta, ref[..., 1]).square().sum())
    record["n"] += int(v.numel())
    nbus, ns = v.shape[1], v.shape[0]
    if record["sum_p"] is None:
        z = lambda: torch.zeros(nbus, dtype=torch.float64, device=device)
        record.update(sum_p=z(), sum_r=z(), sum_p2=z(), sum_r2=z(), sum_pr=z())
    record["sum_p"] += v.sum(0); record["sum_r"] += ref[..., 0].sum(0)
    record["sum_p2"] += v.square().sum(0); record["sum_r2"] += ref[..., 0].square().sum(0)
    record["sum_pr"] += (v * ref[..., 0]).sum(0); record["n_scen"] += ns


def _finalize(record):
    pb = torch.cat([torch.from_numpy(x) for x in record["pb"]]).numpy()
    ns = record["n_scen"]
    var_p = (record["sum_p2"] - record["sum_p"].square() / ns).sum().item()
    var_r = (record["sum_r2"] - record["sum_r"].square() / ns).sum().item()
    cov = (record["sum_pr"] - record["sum_p"] * record["sum_r"] / ns).sum().item()
    return {
        "vmag_rmse": math.sqrt(record["mag_sse"] / record["n"]),
        "angle_rmse_deg": math.degrees(math.sqrt(record["ang_sse"] / record["n"])),
        "within_r2": cov * cov / (var_p * var_r) if var_p > 0 and var_r > 0 else 0.0,
        "within_slope": cov / var_r if var_r > 0 else float("nan"),
        "mean_pb": float(pb.mean()), "max_pb": record["max_pb"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--parquet", required=True)
    ap.add_argument("--rank", type=int, default=16); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--random-draws", type=int, default=8); ap.add_argument("--random-seed", type=int, default=20260909)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.rank <= 0 or args.random_draws <= 0:
        raise ValueError("rank and random-draws must be positive")
    train, _valid, test = split_dataset(args.parquet)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, forward = make_model("g3", args.checkpoint, args.parquet, args.batch, device)
    offset = fit_per_bus_offset(model, forward, train, args.batch, device)
    Uv, Ut, vbar, tbar, ev, et = manifold_basis(train, args.batch, k=args.rank)
    nbus = int(vbar.numel()); rank = min(args.rank, nbus)
    solution_basis = (Uv[:, :rank].to(device=device, dtype=torch.float64),
                      Ut[:, :rank].to(device=device, dtype=torch.float64), vbar, tbar)
    lap_evec, lowpass_h, lap_eval = _topology_operators(train, nbus, device)
    lap_basis = (lap_evec[:, :rank], lap_evec[:, :rank], vbar.to(device), tbar.to(device))
    random_bases = {}
    for j in range(args.random_draws):
        # Independent magnitude and angle bases avoid sharing an accidental draw.
        random_bases[f"random_{j:02d}"] = (
            _random_basis(nbus, rank, args.random_seed + 2 * j, device),
            _random_basis(nbus, rank, args.random_seed + 2 * j + 1, device),
            vbar.to(device), tbar.to(device))
    methods = {"solution_svd": solution_basis, "laplacian_lowfreq": lap_basis,
               **random_bases, "one_hop_lowpass": None}
    rec = {method: {state: _new_record() for state in ("P16", "CSP16")}
           for method in methods}
    # Raw/C are common reference states and make the output self-contained.
    common = {state: _new_record() for state in ("raw", "C")}
    loader = DataLoader(test, batch_size=args.batch, shuffle=False, num_workers=0,
                        collate_fn=collate_blockdiag)
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            raw = forward(batch).detach().to(torch.float64)
            cal = calibrated_forward(lambda _unused: raw, offset, batch)
            sizes = batch["sizes"].numpy().astype(int)
            if len(set(sizes.tolist())) != 1 or int(sizes[0]) != nbus:
                raise ValueError("Expected one fixed GBnetwork topology per batch")
            ns = len(sizes)
            ref = batch["V_newton"].to(device=device, dtype=torch.float64)[0].reshape(ns, nbus, 2)
            raw, cal = raw[0].reshape(ns, nbus, 2), cal[0].reshape(ns, nbus, 2)
            _update(common["raw"], raw[..., 0], raw[..., 1], ref, batch, device)
            _update(common["C"], cal[..., 0], cal[..., 1], ref, batch, device)
            for method, basis in methods.items():
                if method == "one_hop_lowpass":
                    pv_raw, pt_raw = _lowpass(raw[..., 0], raw[..., 1], vbar.to(device), tbar.to(device), lowpass_h)
                    pv_cal, pt_cal = _lowpass(cal[..., 0], cal[..., 1], vbar.to(device), tbar.to(device), lowpass_h)
                else:
                    pv_raw, pt_raw = project_state(raw[..., 0], raw[..., 1], basis)
                    pv_cal, pt_cal = project_state(cal[..., 0], cal[..., 1], basis)
                pv_raw, pt_raw = _restore_known(pv_raw, pt_raw, batch, nbus)
                pv_cal, pt_cal = _restore_known(pv_cal, pt_cal, batch, nbus)
                _update(rec[method]["P16"], pv_raw, pt_raw, ref, batch, device)
                _update(rec[method]["CSP16"], pv_cal, pt_cal, ref, batch, device)
            if bi and bi % 25 == 0:
                print(f"[controls] scored {bi}/{len(loader)} test batches", flush=True)
    payload = {
        "protocol": {
            "model": "PIGNN-GC/G3 seed-42 40-epoch checkpoint", "split": "random_split seed=42",
            "rank": rank, "calibration": "train-only per-bus magnitude/circular-angle offset",
            "known_variables": "PV/slack magnitude and slack angle restored from scenario V_start after each intervention",
            "solution_basis": "train-only NR magnitude and circular-angle SVD",
            "random_basis": f"{args.random_draws} independent deterministic rank-{rank} orthonormal draws; seed={args.random_seed}",
            "laplacian_basis": "lowest normalized weighted-Laplacian modes; A_ij=0.5*(|Y_ij|+|Y_ji|), diagonal zero",
            "lowpass": "one hop H=0.5*(I + D^-1/2 A D^-1/2), applied to state deviations about train mean",
            "pb": "final complex128 GENCO structural-zero PB, all bus-scenario pairs",
        },
        "basis_diagnostics": {"solution_explained_variance_vmag": ev,
                              "solution_explained_variance_angle": et,
                              "laplacian_smallest_eigenvalues": lap_eval[:rank].detach().cpu().tolist()},
        "n_test_scenarios": len(test), "common": {k: _finalize(v) for k, v in common.items()},
        "methods": {method: {state: _finalize(r) for state, r in states.items()}
                    for method, states in rec.items()},
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
