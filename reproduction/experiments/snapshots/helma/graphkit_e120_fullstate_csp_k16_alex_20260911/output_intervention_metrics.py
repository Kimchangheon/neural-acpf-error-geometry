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
from manifold_projection import fit_basis, project_state
from pf_known_mask import apply_known_v


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


def _unknown_only_basis(train, batch_size, rank):
    """Fit separate train-only SVD bases on AC-PF unknown coordinates.

    AC power flow specifies magnitude at PV/slack buses and angle at the
    slack.  This helper deliberately excludes those prescribed coordinates
    from both fitting and projection; it does *not* overwrite their values.
    That distinction lets the evaluator separately test projection scope
    (``unknown_only``) and post-projection setpoint restoration
    (``restore_known``).
    """
    loader = DataLoader(train, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_blockdiag)
    vm, th = [], []
    known = None
    for batch in loader:
        sizes = batch["sizes"].numpy().astype(int)
        if len(set(sizes.tolist())) != 1:
            raise ValueError("Unknown-coordinate CSP requires a common topology.")
        nbus = int(sizes[0])
        if known is None:
            bt = batch["bus_type"][0].reshape(-1)[:nbus].to(torch.long)
            unknown_mag = (bt != 1) & (bt != 2)
            unknown_ang = bt != 1
            known = (unknown_mag, unknown_ang)
        ref = batch["V_newton"][0].to(torch.float64).reshape(len(sizes), nbus, 2)
        vm.append(ref[..., 0][:, known[0]])
        th.append(ref[..., 1][:, known[1]])
    vm, th = torch.cat(vm, 0), torch.cat(th, 0)
    basis, ev, et = fit_basis(vm, th, rank)
    print(f"[unknown-only basis] {vm.shape[0]} train scenarios; "
          f"|V| coordinates={vm.shape[1]}, theta coordinates={th.shape[1]}; "
          f"k={basis[0].shape[1]} captures {100 * ev:.3f}%/{100 * et:.3f}%", flush=True)
    return basis, known


def _project_unknown_only(v, theta, basis, unknown_masks):
    """Project only PQ magnitude and non-slack angle entries."""
    Uv, Ut, vbar, tbar = (x.to(device=v.device, dtype=v.dtype) for x in basis)
    um, ua = (x.to(device=v.device) for x in unknown_masks)
    vp, tp = v.clone(), theta.clone()
    dv = v[..., um] - vbar
    dt = torch.atan2(torch.sin(theta[..., ua] - tbar),
                     torch.cos(theta[..., ua] - tbar))
    vp[..., um] = vbar + (dv @ Uv) @ Uv.T
    tp_unknown = tbar + (dt @ Ut) @ Ut.T
    tp[..., ua] = torch.atan2(torch.sin(tp_unknown), torch.cos(tp_unknown))
    return vp, tp


def _restore_known_setpoints(v, theta, batch, nbus):
    """Restore |V| at PV/slack and angle at slack from the input setpoint."""
    packed = torch.stack((v, theta), dim=-1)
    bt = batch["bus_type"][0].reshape(-1)[:nbus]
    # Preserve scenario-specific setpoints if the dataset varies them.
    vs = batch["V_start"][0].reshape(-1, 2)
    held = apply_known_v(packed, bt, vs)
    return held[..., 0], held[..., 1]


def evaluate(kind, checkpoint, train, test, parquet, batch_size, rank, device,
             projection_variants=("full",)):
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
    variants = tuple(projection_variants)
    invalid = set(variants) - {"full", "restore_known", "unknown_only"}
    if invalid:
        raise ValueError(f"Unknown projection variants: {sorted(invalid)}")
    if "unknown_only" in variants:
        unknown_basis, unknown_masks = _unknown_only_basis(train, batch_size, rank)
    else:
        unknown_basis = unknown_masks = None
    loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate_blockdiag)
    rec = {
        variant: {name: {"pb": [], "max_pb": 0.0, "mag_sse": 0.0,
                         "ang_sse": 0.0, "n": 0}
                  for name in ("raw", "C", f"P{rank}", f"CSP{rank}")}
        for variant in variants
    }
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            raw = forward(batch).detach().to(torch.float64)
            cal = calibrated_forward(forward=lambda _unused: raw, offset=offset, batch=batch)
            sizes = batch["sizes"].numpy().astype(int); nbus = int(sizes[0]); ns = len(sizes)
            ref = batch["V_newton"].to(device=device, dtype=torch.float64)[0].reshape(ns, nbus, 2)
            raw = raw[0].reshape(ns, nbus, 2)
            cal = cal[0].reshape(ns, nbus, 2)
            for variant in variants:
                if variant == "unknown_only":
                    pv_raw, pt_raw = _project_unknown_only(raw[..., 0], raw[..., 1],
                                                            unknown_basis, unknown_masks)
                    pv_cal, pt_cal = _project_unknown_only(cal[..., 0], cal[..., 1],
                                                            unknown_basis, unknown_masks)
                else:
                    pv_raw, pt_raw = project_state(raw[..., 0], raw[..., 1], basis)
                    pv_cal, pt_cal = project_state(cal[..., 0], cal[..., 1], basis)
                if variant == "restore_known":
                    pv_raw, pt_raw = _restore_known_setpoints(pv_raw, pt_raw, batch, nbus)
                    pv_cal, pt_cal = _restore_known_setpoints(pv_cal, pt_cal, batch, nbus)
                states = {
                    "raw": (raw[..., 0], raw[..., 1]),
                    "C": (cal[..., 0], cal[..., 1]),
                    f"P{rank}": (pv_raw, pt_raw),
                    f"CSP{rank}": (pv_cal, pt_cal),
                }
                for name, (v, theta) in states.items():
                    per_scenario, max_pb = pb_per_scenario(v, theta, batch, device, return_global_max=True)
                    r = rec[variant][name]
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
    for variant, records in rec.items():
        metrics = {}
        for name, r in records.items():
            pb = torch.cat([torch.from_numpy(x) for x in r["pb"]]).numpy()
            ns = r["n_scen"]
            var_p = (r["sum_p2"] - r["sum_p"].square() / ns).sum().item()
            var_r = (r["sum_r2"] - r["sum_r"].square() / ns).sum().item()
            cov = (r["sum_pr"] - r["sum_p"] * r["sum_r"] / ns).sum().item()
            aw = cov / var_r if var_r > 0 else float("nan")
            rw2 = cov * cov / (var_p * var_r) if var_p > 0 and var_r > 0 else 0.0
            metrics[name] = {
                "vmag_rmse": math.sqrt(r["mag_sse"] / r["n"]),
                "angle_rmse_deg": math.degrees(math.sqrt(r["ang_sse"] / r["n"])),
                "mean_pb": float(pb.mean()),
                "max_pb": r["max_pb"],
                "within_r2": rw2,
                "within_slope": aw,
            }
        results[variant] = metrics
    return {"model": kind, "rank": rank, "projection_variants": list(variants),
            "n_scenarios": len(test),
            "calibration_offset_rms_vmag": float(offset[0].square().mean().sqrt()),
            "metrics": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("constant", "g3", "pignn_base", "gridsfm", "graphkit", "lumina"), required=True)
    ap.add_argument("--checkpoint", default=""); ap.add_argument("--parquet", required=True)
    ap.add_argument("--rank", type=int, required=True); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--projection-variants", nargs="+", default=["full"],
                    choices=("full", "restore_known", "unknown_only"),
                    help="One or more projection policies evaluated from one model inference pass each.")
    args = ap.parse_args()
    train, _valid, test = split_dataset(args.parquet)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {"protocol": {"split": "random_split seed=42",
                        "rank": args.rank, "calibration": "train-only per-bus magnitude and circular-angle offsets",
                        "pb": "complex128 GENCO-style per-bus norm, all bus-scenario pairs",
                        "max_pb": "maximum individual bus PB over the test split",
                        "split_seed": 42},
           "result": evaluate(args.model, args.checkpoint, train, test, args.parquet,
                              args.batch, args.rank, device,
                              projection_variants=args.projection_variants)}
    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
