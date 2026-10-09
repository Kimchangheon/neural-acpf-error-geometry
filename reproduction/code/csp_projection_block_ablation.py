#!/usr/bin/env python3
"""GBnetwork CSP magnitude/angle-block ablation.

This is deliberately a post-hoc scorer: each checkpoint is forwarded once per
split, calibration and both SVD bases are fitted on training data only, and
the four output blocks are scored against the identical frozen test split.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from collate_blockdiag_optimized_complex_columns import collate_blockdiag, ybus_matvec
from controlled_error_geometry import calibrated_forward, make_model, split_dataset
from diagnose_residual_distributions import angle_diff, fit_per_bus_offset, manifold_basis
from manifold_projection import project_state
from pf_known_mask import apply_known_v


def restore_known(v, theta, batch, nbus):
    packed = torch.stack((v, theta), dim=-1)
    bt = batch["bus_type"][0].reshape(-1)[:nbus]
    vs = batch["V_start"][0].reshape(-1, 2)
    held = apply_known_v(packed, bt, vs)
    return held[..., 0], held[..., 1]


def pb_with_pq_components(v, theta, batch, device):
    """Current complex128 GENCO-style PB plus separately auditable |dP|/|dQ|."""
    sizes = batch["sizes"].numpy().astype(int)
    y = batch["Ybus"].to(device=device, dtype=torch.complex128)
    s = batch["S_start"].to(device=device, dtype=torch.complex128)[0]
    bt = batch["bus_type"][0].to(device)
    vc = v.reshape(-1).to(device) * torch.exp(1j * theta.reshape(-1).to(device))
    sc = vc * ybus_matvec(y, vc).conj()
    dp, dq = s.real - sc.real, s.imag - sc.imag
    p_mask, q_mask = bt != 1, (bt != 1) & (bt != 2)
    pb = torch.sqrt((dp * p_mask) ** 2 + (dq * q_mask) ** 2)
    nbus = int(sizes[0])
    return (pb.reshape(len(sizes), nbus).mean(1).detach().cpu(),
            float(pb.max().detach().cpu()),
            float((dp.abs() * p_mask).sum().detach().cpu()),
            float((dq.abs() * q_mask).sum().detach().cpu()),
            int(p_mask.sum().item()), int(q_mask.sum().item()))


def fresh_record():
    return dict(pb=[], max_pb=0.0, mag_sse=0., ang_sse=0., n=0,
                abs_dp_sum=0., abs_dq_sum=0., p_count=0, q_count=0,
                sum_p=None, sum_r=None, sum_p2=None, sum_r2=None, sum_pr=None,
                n_scen=0, restoration_abs_delta_v=0., restoration_abs_delta_theta=0.)


def score(kind, checkpoint, parquet, rank, batch_size, device, restore_setpoints=False, lumina_args=""):
    train, _valid, test = split_dataset(parquet)
    model, forward = make_model(kind, checkpoint, parquet, batch_size, device, lumina_args=lumina_args)
    offset = fit_per_bus_offset(model, forward, train, batch_size, device)
    Uv, Ut, vbar, tbar, ev, et = manifold_basis(train, batch_size, k=rank)
    basis = tuple(x.to(device=device, dtype=torch.float64) for x in (Uv[:, :rank], Ut[:, :rank], vbar, tbar))
    loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate_blockdiag)
    names = ("C", "V_only", "theta_only", "CSP16")
    rec = {n: fresh_record() for n in names}
    with torch.no_grad():
        for bi, data in enumerate(loader):
            raw = forward(data).detach().to(torch.float64)
            cal = calibrated_forward(lambda _unused: raw, offset, data)[0]
            sizes = data["sizes"].numpy().astype(int); ns, nbus = len(sizes), int(sizes[0])
            cal = cal.reshape(ns, nbus, 2)
            ref = data["V_newton"].to(device=device, dtype=torch.float64)[0].reshape(ns, nbus, 2)
            vp, tp = project_state(cal[..., 0], cal[..., 1], basis)
            before = {
                "C": (cal[..., 0], cal[..., 1]),
                "V_only": (vp, cal[..., 1]),
                "theta_only": (cal[..., 0], tp),
                "CSP16": (vp, tp),
            }
            for name, (v0, t0) in before.items():
                if restore_setpoints:
                    v, theta = restore_known(v0, t0, data, nbus)
                else:
                    v, theta = v0, t0
                r = rec[name]
                r["restoration_abs_delta_v"] += float((v - v0).abs().sum())
                r["restoration_abs_delta_theta"] += float(angle_diff(theta, t0).abs().sum())
                ps, mx, ap, aq, np_, nq_ = pb_with_pq_components(v, theta, data, device)
                r["pb"].append(ps); r["max_pb"] = max(r["max_pb"], mx)
                r["mag_sse"] += float((v - ref[..., 0]).square().sum())
                r["ang_sse"] += float(angle_diff(theta, ref[..., 1]).square().sum())
                r["n"] += int(v.numel()); r["abs_dp_sum"] += ap; r["abs_dq_sum"] += aq
                r["p_count"] += np_ * ns; r["q_count"] += nq_ * ns
                if r["sum_p"] is None:
                    r["sum_p"] = torch.zeros(nbus, dtype=torch.float64, device=device)
                    r["sum_r"] = torch.zeros_like(r["sum_p"]); r["sum_p2"] = torch.zeros_like(r["sum_p"])
                    r["sum_r2"] = torch.zeros_like(r["sum_p"]); r["sum_pr"] = torch.zeros_like(r["sum_p"])
                r["sum_p"] += v.sum(0); r["sum_r"] += ref[..., 0].sum(0)
                r["sum_p2"] += v.square().sum(0); r["sum_r2"] += ref[..., 0].square().sum(0)
                r["sum_pr"] += (v * ref[..., 0]).sum(0); r["n_scen"] += ns
            if bi and bi % 25 == 0:
                print(f"[{kind}] {bi}/{len(loader)} test batches", flush=True)
    result = {}
    for name, r in rec.items():
        ns = r["n_scen"]
        vp = (r["sum_p2"] - r["sum_p"].square() / ns).sum().item()
        vr = (r["sum_r2"] - r["sum_r"].square() / ns).sum().item()
        cov = (r["sum_pr"] - r["sum_p"] * r["sum_r"] / ns).sum().item()
        result[name] = {
            "vmag_rmse": math.sqrt(r["mag_sse"] / r["n"]),
            "angle_rmse_deg": math.degrees(math.sqrt(r["ang_sse"] / r["n"])),
            "within_r2": cov * cov / (vp * vr) if vp > 0 and vr > 0 else 0.,
            "within_slope": cov / vr if vr > 0 else float("nan"),
            "mean_pb": float(torch.cat(r["pb"]).mean()), "max_pb": r["max_pb"],
            "mean_abs_delta_p": r["abs_dp_sum"] / r["p_count"],
            "mean_abs_delta_q": r["abs_dq_sum"] / r["q_count"],
            "restoration_abs_delta_v": r["restoration_abs_delta_v"],
            "restoration_abs_delta_theta": r["restoration_abs_delta_theta"],
        }
    return {"model": kind, "checkpoint": checkpoint, "rank": rank, "n_test": len(test),
            "basis_explained_variance": {"magnitude": ev, "angle": et},
            "calibration_offset_rms_vmag": float(offset[0].square().mean().sqrt()),
            "conditions": result}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("g3", "gridsfm", "graphkit", "lumina"), required=True)
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lumina-args", default="")
    ap.add_argument("--restore-known", action="store_true",
                    help="Restore PV/slack magnitude and slack angle from V_start after each output block.")
    a = ap.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {"protocol": {"split": "seed-42 current-paper split", "calibration": "train-only per-bus offsets",
                        "projection": "separate train-only magnitude/angle SVD; k=16",
                        "post_projection": ("restore PV/slack magnitude and slack angle from V_start"
                                            if a.restore_known else "none"),
                        "pb": "complex128 GENCO structural-zero PB"},
           "result": score(a.model, a.checkpoint, a.parquet, a.rank, a.batch, device,
                           restore_setpoints=a.restore_known, lumina_args=a.lumina_args)}
    Path(a.out).write_text(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
