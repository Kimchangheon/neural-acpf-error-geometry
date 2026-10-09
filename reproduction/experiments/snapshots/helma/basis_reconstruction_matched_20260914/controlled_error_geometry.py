#!/usr/bin/env python3
"""Controlled, fixed-norm error-direction diagnostic for GBnetwork PF surrogates.

This is intentionally a diagnostic, not a deployable correction: it constructs
counterfactual magnitudes around the held-out Newton solution while retaining
each model's train-calibrated angle prediction.  All calibration, SVD fitting,
and rank selection use training/validation data only; test is read only after a
rank has been supplied with ``--rank``.
"""
from __future__ import annotations

import argparse, json, math, shlex
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from Dataset_optimized_complex_columns import ChanghunDataset
from collate_blockdiag_optimized_complex_columns import collate_blockdiag, ybus_matvec
from diagnose_residual_distributions import (
    angle_diff, fit_per_bus_offset, forward_gridsfm, forward_pignn,
    make_graphkit_or_lumina, make_gridsfm_model, manifold_basis, pignn_model,
)
from manifold_projection import project_state

RANKS = (4, 8, 16, 32, 64)
ALPHAS = (0.0, .25, .5, .75, 1.0)


def split_dataset(path):
    ds = ChanghunDataset(path, per_unit=True, target_S_base=1e8,
                         share_grid=True, share_ybus=True,
                         lazy_row_groups=True, row_group_cache_size=4,
                         complex_dtype="complex128")
    n = len(ds); nt = int(.3333 * n); nv = int(.3333 * n)
    subsets = random_split(ds, [nt, nv, n - nt - nv],
                           generator=torch.Generator().manual_seed(42))
    # ``random_split`` fixes membership but leaves each Subset's indices in a
    # random order.  For a lazy row-group parquet this causes needless random
    # I/O while changing neither split membership nor any metric.  Retain the
    # exact seed-42 sets and make only their evaluation order monotone.
    for subset in subsets:
        subset.indices.sort()
    return tuple(subsets)


def make_model(
    kind, checkpoint, parquet, batch, device, lumina_args="",
    pignn_batched_armijo=False,
    pignn_multi_rhs_armijo=False,
    pignn_global_context_packed_fastpath=False,
):
    if kind in ("graphkit", "lumina"):
        model, ext_forward, ext_to_dev = make_graphkit_or_lumina(
            kind, checkpoint, device, parquet, batch,
            lumina_args=shlex.split(lumina_args))
        def forward(b):
            bb = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in b.items()} if ext_to_dev else b
            return ext_forward(bb)
    elif kind in ("gridsfm", "gridsfm_mask"):
        model = make_gridsfm_model(checkpoint, device)
        forward = lambda b: forward_gridsfm(model, b, device,
                                             mask_known_v=(kind == "gridsfm_mask"))
    elif kind == "pignn_base":
        # The case1354 v2fix 40-epoch baseline predates the G3 global-context
        # sidecar.  It must be reconstructed without the G3 attention module.
        # This ppNR-v2 checkpoint predates residual-feature conditioning.  Its
        # input contract is [v, theta, dP, dQ, learned d-dimensional state]
        # (8 channels for d=4), not the later signed-log 9-channel contract.
        model = pignn_model(
            "none", "scalar_tanh", residual_feature_norm="none",
            armijo_batched_candidates=pignn_batched_armijo,
            armijo_multi_rhs=pignn_multi_rhs_armijo,
            global_context_packed_fastpath=pignn_global_context_packed_fastpath,
        ).to(device)
        state = torch.load(checkpoint, map_location=device)
        r = model.load_state_dict(state, strict=False)
        if r.missing_keys or r.unexpected_keys:
            raise RuntimeError(f"PIGNN baseline checkpoint mismatch: {r}")
        forward = lambda b: forward_pignn(model, b, device)
    else:
        model = pignn_model(
            "attn_post", "scalar_tanh",
            armijo_batched_candidates=pignn_batched_armijo,
            armijo_multi_rhs=pignn_multi_rhs_armijo,
            global_context_packed_fastpath=pignn_global_context_packed_fastpath,
        ).to(device)
        state = torch.load(checkpoint, map_location=device)
        r = model.load_state_dict(state, strict=False)
        if r.missing_keys or r.unexpected_keys:
            raise RuntimeError(f"PIGNN checkpoint mismatch: {r}")
        forward = lambda b: forward_pignn(model, b, device)
    model.eval()
    return model, forward


def calibrated_forward(forward, offset, batch):
    pred = forward(batch).detach().to(torch.float64)
    sizes = batch["sizes"].numpy().astype(int); nrep = len(sizes)
    dvm, dth = offset
    out = pred.clone()
    out[0, :, 0] -= dvm.repeat(nrep)
    out[0, :, 1] = torch.atan2(torch.sin(out[0, :, 1] - dth.repeat(nrep)),
                                torch.cos(out[0, :, 1] - dth.repeat(nrep)))
    return out


def pb_per_scenario(vmag, theta, batch, device, return_global_max=False):
    """Exact complex128 GENCO-style Mean PB, one mean per scenario.

    ``return_global_max`` additionally returns the maximum individual bus PB,
    not the maximum of scenario means.  This is the tail quantity requested
    for the calibrated-versus-CSP summary table.
    """
    sizes = batch["sizes"].numpy().astype(int)
    y = batch["Ybus"].to(device=device, dtype=torch.complex128)
    s = batch["S_start"].to(device=device, dtype=torch.complex128)[0]
    bt = batch["bus_type"][0].to(device)
    vc = vmag.reshape(-1).to(device) * torch.exp(1j * theta.reshape(-1).to(device))
    sc = vc * ybus_matvec(y, vc).conj()
    dp = s.real - sc.real; dq = s.imag - sc.imag
    p_mask = bt != 1; q_mask = (bt != 1) & (bt != 2)
    pb = torch.sqrt((dp * p_mask) ** 2 + (dq * q_mask) ** 2)
    nbus = int(sizes[0])
    per_scenario = pb.reshape(len(sizes), nbus).mean(1).detach().cpu().numpy()
    if return_global_max:
        return per_scenario, float(pb.max().detach().cpu())
    return per_scenario


def counterfactual(vref, vcal, theta, U, alpha, mode):
    """Return fixed-norm counterfactual magnitude and count zero denominators."""
    e = vcal - vref
    ep = (e @ U) @ U.T
    eo = e - ep
    direction = ep + alpha * eo if mode == "off" else alpha * ep + eo
    numer = torch.linalg.vector_norm(e, dim=1)
    denom = torch.linalg.vector_norm(direction, dim=1)
    bad = (denom == 0) & (numer != 0)
    if bool(bad.any()):
        raise RuntimeError(f"{mode}, alpha={alpha}: nonzero error with zero direction denominator")
    # Exact-zero original errors have a defined zero counterfactual; no epsilon is added.
    scale = torch.where(denom > 0, numer / denom, torch.zeros_like(denom))
    out = vref + direction * scale[:, None]
    err = torch.linalg.vector_norm(out - vref, dim=1) - numer
    return out, float(err.abs().max()), int((denom == 0).sum())


def bootstrap_delta(a, b, seed=42, draws=2000):
    """Paired bootstrap CI for mean(a-b), with a and b indexed by scenario."""
    rng = np.random.default_rng(seed); n = len(a); d = np.asarray(a) - np.asarray(b)
    means = np.empty(draws)
    for i in range(draws): means[i] = d[rng.integers(0, n, n)].mean()
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def run_model(kind, checkpoint, train, evaluation, parquet, batch, device, ranks, mode, max_batches=0, lumina_args=""):
    model, forward = make_model(kind, checkpoint, parquet, batch, device, lumina_args=lumina_args)
    offset = fit_per_bus_offset(model, forward, train, batch, device)
    Uall, Utall, vbar, tbar, _ev, _et = manifold_basis(train, batch, k=max(ranks))
    loader = DataLoader(evaluation, batch_size=batch, shuffle=False, num_workers=0,
                        collate_fn=collate_blockdiag)
    out = {"model": kind, "n_scenarios": 0, "rank": {}}
    # Saved separately as compressed NPZ rather than embedded in the JSON;
    # the latter remains a compact, human-auditable summary.
    scenario_arrays = {}
    # Allocate all rank-wise accumulators before the evaluation loop. The
    # calibrated model output below is obtained exactly once per batch.
    rec = {}
    for k in ranks:
        rec[k] = {"csp_pb": [], "csp_sse": 0.0, "csp_n": 0,
                  "values": {d: {str(a): [] for a in ALPHAS} for d in ("off", "in")},
                  "max_norm_error": 0.0, "zero_denoms": 0}
    cal_pb, cal_sse, cal_n, cal_max_pb = [], 0.0, 0, 0.0
    with torch.no_grad():
        for bi, b in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            pred = calibrated_forward(forward, offset, b)  # one neural forward pass
            sizes = b["sizes"].numpy().astype(int); nbus = int(sizes[0])
            ref = b["V_newton"].to(device=device, dtype=torch.float64)[0].reshape(len(sizes), nbus, 2)
            cal = pred[0].reshape(len(sizes), nbus, 2)
            cal_batch_pb, cal_batch_max = pb_per_scenario(
                cal[..., 0], cal[..., 1], b, device, return_global_max=True)
            cal_pb.append(cal_batch_pb); cal_max_pb = max(cal_max_pb, cal_batch_max)
            cal_sse += float((cal[..., 0] - ref[..., 0]).square().sum()); cal_n += cal[..., 0].numel()
            for k in ranks:
                r = rec[k]; U = Uall[:, :k].to(device=device, dtype=torch.float64)
                pv, pt = project_state(cal[..., 0], cal[..., 1], (U, Utall[:, :k], vbar, tbar))
                csp_batch_pb, csp_batch_max = pb_per_scenario(
                    pv, pt, b, device, return_global_max=True)
                r["csp_pb"].append(csp_batch_pb)
                r["csp_max_pb"] = max(r.get("csp_max_pb", 0.0), csp_batch_max)
                r["csp_sse"] += float((pv - ref[..., 0]).square().sum()); r["csp_n"] += pv.numel()
                if mode == "test":
                    for direction in ("off", "in"):
                        for a in ALPHAS:
                            vm, norm_err, zeros = counterfactual(ref[..., 0], cal[..., 0], cal[..., 1], U, a, direction)
                            r["values"][direction][str(a)].append(pb_per_scenario(vm, cal[..., 1], b, device))
                            r["max_norm_error"] = max(r["max_norm_error"], norm_err)
                            r["zero_denoms"] += zeros
            if bi and bi % 25 == 0: print(f"[{kind}] evaluated batch {bi}/{len(loader)}", flush=True)
    cal = np.concatenate(cal_pb)
    out["n_scenarios"] = int(cal.size)
    out["calibrated"] = {"mean_pb": float(cal.mean()), "max_pb": cal_max_pb,
                         "vmag_rmse": float(math.sqrt(cal_sse / cal_n))}
    for k, r in rec.items():
        cpb = np.concatenate(r["csp_pb"])
        entry = {"csp": {"mean_pb": float(cpb.mean()), "max_pb": r.get("csp_max_pb", 0.0),
                           "relative_pb_to_calibrated": float(cpb.mean() / cal.mean()),
                           "vmag_rmse": float(math.sqrt(r["csp_sse"] / r["csp_n"]))}}
        if mode == "test":
            packed = {d: {a: np.concatenate(v) for a, v in by_a.items()} for d, by_a in r["values"].items()}
            def summarize(series, base):
                delta = series - base
                return {"mean_pb": float(series.mean()), "relative_pb": float(series.mean() / base.mean()),
                        "delta_pb": float(delta.mean()),
                        "fraction_scenarios_pb_decreases": float(np.mean(delta < 0)),
                        "paired_bootstrap_ci_delta_pb": bootstrap_delta(series, base)}
            entry.update({d: {a: summarize(x, packed[d]["1.0"]) for a, x in by_a.items()} for d, by_a in packed.items()})
            entry["max_abs_fixed_norm_error"] = r["max_norm_error"]
            entry["zero_denominator_scenarios"] = r["zero_denoms"]
            entry["max_abs_alpha1_pb_reproduction_error"] = float(max(
                np.max(np.abs(packed[d]["1.0"] - cal)) for d in ("off", "in")))
            for direction, by_alpha in packed.items():
                for alpha, values in by_alpha.items():
                    scenario_arrays[f"rank{k}_{direction}_alpha{alpha}"] = values
            scenario_arrays[f"rank{k}_calibrated"] = cal
        out["rank"][str(k)] = entry
    out["calibration_offset_rms_vmag"] = float(offset[0].square().mean().sqrt())
    return out, scenario_arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--phase", choices=("validation", "test", "summary"), required=True)
    ap.add_argument("--ranks", nargs="+", type=int, default=list(RANKS))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--models", nargs="+", choices=("g3", "gridsfm", "graphkit", "lumina"),
                    default=("g3", "gridsfm", "graphkit", "lumina"))
    ap.add_argument("--max-batches", type=int, default=0,
                    help="smoke-test cap only; never use capped output as an experiment result")
    ap.add_argument("--pignn", required=True); ap.add_argument("--gridsfm", required=True)
    ap.add_argument("--graphkit", required=True); ap.add_argument("--lumina", required=True)
    ap.add_argument("--lumina-args", default="")
    args = ap.parse_args()
    train, valid, test = split_dataset(args.parquet)
    evaluation = valid if args.phase == "validation" else test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {"g3": args.pignn, "gridsfm": args.gridsfm,
              "graphkit": args.graphkit, "lumina": args.lumina}
    payload = {"protocol": {"split": "random_split seed=42, 1/3 train/valid/test",
                            "phase": args.phase, "ranks": args.ranks,
                            "alphas": list(ALPHAS), "pb": "GENCO-style complex128, all bus-scenario pairs"},
               "models": {}}
    for kind, ckpt in models.items():
        if kind not in args.models:
            continue
        print(f"[begin] {kind}", flush=True)
        summary, scenario_arrays = run_model(kind, ckpt, train, evaluation, args.parquet,
                                             args.batch, device, args.ranks, args.phase,
                                             args.max_batches, args.lumina_args if kind == "lumina" else "")
        payload["models"][kind] = summary
        if args.phase == "test":
            np.savez_compressed(Path(args.out).with_name(f"{kind}_test_per_scenario.npz"),
                                **scenario_arrays)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"[done] {kind}", flush=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
