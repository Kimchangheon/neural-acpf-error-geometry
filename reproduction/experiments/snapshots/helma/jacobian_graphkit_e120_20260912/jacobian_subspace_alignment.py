#!/usr/bin/env python3
"""Direct Jacobian alignment test for the train-solution voltage subspace.

The reduced state is [theta at non-slack buses, |V| at PQ buses] and the
reduced residual is [Delta P at non-slack buses, Delta Q at PQ buses].  At NR
solutions, analytic complex128 JVPs compare matched components Pq and (I-P)q.
Because their dimensions differ substantially, unit-normalized component gains
are the primary statistic; the unnormalized quantities requested in the review
are retained as an audit statistic.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from collate_blockdiag_optimized_complex_columns import collate_blockdiag
from controlled_error_geometry import split_dataset
from diagnose_residual_distributions import manifold_basis


def _orthonormal_restricted_basis(u, mask, rank, device):
    """Orthonormalize the train-SVD span after restricting to PF unknowns."""
    restricted = u[mask, :rank].to(device=device, dtype=torch.float64)
    q, r = torch.linalg.qr(restricted, mode="reduced")
    diag = torch.diagonal(r).abs()
    tol = torch.finfo(r.dtype).eps * max(restricted.shape) * diag.max()
    keep = int((diag > tol).sum().item())
    if keep == 0:
        raise RuntimeError("Training subspace has zero rank after PF-variable restriction")
    return q[:, :keep], keep


def _ybus_mv(y, x):
    """Apply one N x N Ybus to rows of x in complex128."""
    if y.is_sparse:
        return torch.sparse.mm(y.coalesce(), x.T).T
    return (y @ x.T).T


def reduced_jvp(y, vmag, theta, directions, angle_unknown, mag_unknown):
    """Analytic J_r d for r=S_set-V*conj(YV); no finite differencing."""
    m, _ = directions.shape
    nt = int(angle_unknown.sum().item())
    dtheta = torch.zeros(m, vmag.numel(), dtype=torch.float64, device=vmag.device)
    dvm = torch.zeros_like(dtheta)
    dtheta[:, angle_unknown] = directions[:, :nt]
    dvm[:, mag_unknown] = directions[:, nt:]
    ejt = torch.exp(1j * theta)
    vc = vmag * ejt
    dvc = ejt[None, :] * (dvm + 1j * vmag[None, :] * dtheta)
    ibus = _ybus_mv(y, vc[None, :])[0]
    ydv = _ybus_mv(y, dvc)
    dscalc = dvc * ibus.conj()[None, :] + vc[None, :] * ydv.conj()
    dr = -dscalc
    return torch.cat((dr.real[:, angle_unknown], dr.imag[:, mag_unknown]), dim=1)


def summarize(values):
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)),
            "median": float(np.median(x)), "p05": float(np.quantile(x, .05)),
            "p25": float(np.quantile(x, .25)), "p75": float(np.quantile(x, .75)),
            "p95": float(np.quantile(x, .95)), "min": float(x.min()),
            "max": float(x.max()), "n": int(x.size)}


def paired_bootstrap(log_ratio, seed, draws=10000):
    """Scenario-clustered bootstrap CI of median log10(perp/parallel gain)."""
    x = np.asarray(log_ratio, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n_scen = x.shape[0]
    med = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        med[i] = np.median(x[rng.integers(0, n_scen, n_scen)].reshape(-1))
    return [float(np.quantile(med, .025)), float(np.quantile(med, .975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True); ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--basis-batch", type=int, default=16)
    ap.add_argument("--scenarios", type=int, default=128)
    ap.add_argument("--directions", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if min(args.rank, args.scenarios, args.directions) <= 0:
        raise ValueError("rank, scenarios, and directions must be positive")
    train, _valid, test = split_dataset(args.parquet)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    uv, ut, _vbar, _tbar, ev, et = manifold_basis(train, args.basis_batch, k=args.rank)

    # Fixed held-out sample chosen before any Jacobian outcome is computed.
    gen = torch.Generator().manual_seed(args.seed)
    local_test_indices = torch.randperm(len(test), generator=gen)[:args.scenarios].tolist()
    selected = Subset(test, local_test_indices)
    loader = DataLoader(selected, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_blockdiag)
    records = {key: [] for key in ("gain_parallel", "gain_perp",
                                    "raw_jp", "raw_jperp",
                                    "norm_parallel", "norm_perp", "log10_gain_ratio")}
    per_scenario_log_ratio = []
    ranks = None; n_state = n_residual = None
    direction_gen = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    original_indices = []
    with torch.no_grad():
        for si, batch in enumerate(loader):
            nbus = int(batch["sizes"][0])
            bt = batch["bus_type"][0].reshape(-1)[:nbus].to(torch.long)
            angle_unknown_cpu = bt != 1
            mag_unknown_cpu = (bt != 1) & (bt != 2)
            if ranks is None:
                qtheta, rt = _orthonormal_restricted_basis(ut, angle_unknown_cpu,
                                                            args.rank, device)
                qmag, rv = _orthonormal_restricted_basis(uv, mag_unknown_cpu,
                                                          args.rank, device)
                ranks = {"angle": rt, "magnitude": rv, "combined": rt + rv}
                n_state = int(angle_unknown_cpu.sum() + mag_unknown_cpu.sum())
                n_residual = n_state
            angle_unknown = angle_unknown_cpu.to(device); mag_unknown = mag_unknown_cpu.to(device)
            nt = int(angle_unknown.sum().item())
            q = torch.randn(args.directions, n_state, generator=direction_gen,
                            dtype=torch.float64).to(device)
            q /= q.norm(dim=1, keepdim=True)
            qth, qvm = q[:, :nt], q[:, nt:]
            pth = (qth @ qtheta) @ qtheta.T
            pvm = (qvm @ qmag) @ qmag.T
            parallel = torch.cat((pth, pvm), dim=1)
            perp = q - parallel
            np_norm = parallel.norm(dim=1); no_norm = perp.norm(dim=1)
            eps = torch.finfo(torch.float64).eps
            if bool((np_norm <= eps).any() or (no_norm <= eps).any()):
                raise RuntimeError("Zero projected-direction denominator")
            parallel_unit = parallel / np_norm[:, None]
            perp_unit = perp / no_norm[:, None]
            ref = batch["V_newton"][0, :nbus].to(device=device, dtype=torch.float64)
            # batch_size=1, so Ybus already has exactly this grid's N x N
            # shape.  Avoid sparse slicing, which PyTorch does not implement.
            y = batch["Ybus"].to(device=device, dtype=torch.complex128)
            jp = reduced_jvp(y, ref[:, 0], ref[:, 1], parallel,
                             angle_unknown, mag_unknown).norm(dim=1)
            jo = reduced_jvp(y, ref[:, 0], ref[:, 1], perp,
                             angle_unknown, mag_unknown).norm(dim=1)
            gp = reduced_jvp(y, ref[:, 0], ref[:, 1], parallel_unit,
                             angle_unknown, mag_unknown).norm(dim=1)
            go = reduced_jvp(y, ref[:, 0], ref[:, 1], perp_unit,
                             angle_unknown, mag_unknown).norm(dim=1)
            log_ratio = torch.log10(go / gp)
            for key, val in (("gain_parallel", gp), ("gain_perp", go),
                             ("raw_jp", jp), ("raw_jperp", jo),
                             ("norm_parallel", np_norm), ("norm_perp", no_norm),
                             ("log10_gain_ratio", log_ratio)):
                records[key].extend(val.cpu().tolist())
            per_scenario_log_ratio.append(log_ratio.cpu().numpy())
            original_indices.append(int(test.indices[local_test_indices[si]]))
            if (si + 1) % 16 == 0:
                print(f"[jacobian] {si + 1}/{len(loader)} scenarios", flush=True)
    per_scenario_log_ratio = np.stack(per_scenario_log_ratio)
    ratio = np.asarray(records["gain_perp"]) / np.asarray(records["gain_parallel"])
    payload = {
        "protocol": {"split": "random_split seed=42", "basis": "train-only solution SVD",
                     "rank_requested": args.rank,
                     "reduced_state": "[theta(non-slack), |V|(PQ)]",
                     "reduced_residual": "[DeltaP(non-slack), DeltaQ(PQ)]",
                     "jacobian": "analytic complex128 JVP at held-out NR solution",
                     "heldout_sampling": f"{args.scenarios} test scenarios, deterministic seed {args.seed}",
                     "directions": f"{args.directions} matched isotropic Gaussian unit vectors per scenario",
                     "primary_comparison": "||J Pq/||Pq||||2 versus ||J(I-P)q/||(I-P)q||||2",
                     "bootstrap": "scenario-clustered paired bootstrap, 10000 draws"},
        "dimensions": {"buses": nbus, "reduced_state": n_state,
                       "reduced_residual": n_residual, "subspace_ranks": ranks},
        "basis_explained_variance": {"magnitude": ev, "angle": et},
        "selected_test_indices": original_indices,
        "summary": {key: summarize(val) for key, val in records.items()},
        "comparison": {"median_perp_over_parallel_gain": float(np.median(ratio)),
                       "mean_perp_over_parallel_gain": float(np.mean(ratio)),
                       "fraction_perp_gain_larger": float(np.mean(ratio > 1)),
                       "median_log10_ratio_scenario_bootstrap_ci95": paired_bootstrap(
                           per_scenario_log_ratio, args.seed + 2)},
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
