#!/usr/bin/env python3
"""Cache GBnetwork Raw/CSP16 neural warm starts for CPU NR benchmarking.

The file deliberately separates GPU inference from CPU solver timing.  It
performs each model/split forward pass once, fits calibration and bases from the
training split only, and stores the exact test starts passed to NR.  Prescribed
PF coordinates are restored after both Raw and CSP outputs: |V| at PV/slack
and angle at slack use the scenario's V_start setpoints.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from collate_blockdiag_optimized_complex_columns import collate_blockdiag
from controlled_error_geometry import make_model, split_dataset
from diagnose_residual_distributions import fit_per_bus_offset, manifold_basis
from manifold_projection import project_state


def restore_known(state: torch.Tensor, vstart: torch.Tensor, bt: torch.Tensor) -> torch.Tensor:
    """Restore prescribed PF coordinates without changing unknown coordinates."""
    out = state.clone()
    known_vm = (bt == 1) | (bt == 2)
    slack = bt == 1
    out[:, known_vm, 0] = vstart[:, known_vm, 0]
    out[:, slack, 1] = vstart[:, slack, 1]
    return out


def calibrated(pred: torch.Tensor, dvm: torch.Tensor, dth: torch.Tensor) -> torch.Tensor:
    out = pred.to(torch.float64).clone()
    out[..., 0] -= dvm
    out[..., 1] = torch.atan2(torch.sin(out[..., 1] - dth),
                              torch.cos(out[..., 1] - dth))
    return out


def save_once(path: Path, arr: np.ndarray):
    if path.exists():
        previous = np.load(path, mmap_mode="r")
        if previous.shape != arr.shape or previous.dtype != arr.dtype:
            raise RuntimeError(f"existing cache incompatible: {path}")
        if not np.array_equal(previous, arr):
            raise RuntimeError(f"existing cache differs: {path}")
        return
    np.save(path, arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=("g3", "gridsfm", "gridsfm_mask", "graphkit", "lumina"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--fit-batch", type=int, default=16)
    ap.add_argument("--lumina-args", default="")
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("warm-start cache must be built on CUDA")
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    train, _valid, test = split_dataset(args.parquet)
    device = torch.device("cuda")
    # `make_model` in the current evaluator accepts the LUMINA argument string
    # positionally; keep this compatible with both older and current overlays.
    model, forward = make_model(args.model, args.checkpoint, args.parquet,
                                args.batch, device, args.lumina_args)
    offsets = fit_per_bus_offset(model, forward, train, args.fit_batch, device)
    uv, ut, vbar, tbar, _ev, _et = manifold_basis(train, args.fit_batch, k=args.rank)
    uv = uv[:, :args.rank].to(device=device, dtype=torch.float64)
    ut = ut[:, :args.rank].to(device=device, dtype=torch.float64)
    vbar = vbar.to(device=device, dtype=torch.float64)
    tbar = tbar.to(device=device, dtype=torch.float64)
    dvm, dth = (x.to(device=device, dtype=torch.float64) for x in offsets)

    n, nbus = len(test), None
    raw_mm = csp_mm = sbus_mm = vstart_mm = None
    static_y = static_bt = None
    loader = DataLoader(test, batch_size=args.batch, shuffle=False, num_workers=0,
                        collate_fn=collate_blockdiag)
    t0 = time.perf_counter(); pos = 0
    restore_changes = {"raw_max_abs": 0.0, "csp_max_abs": 0.0}
    with torch.inference_mode():
        for batch in loader:
            sizes = batch["sizes"].numpy().astype(int)
            if not np.all(sizes == sizes[0]):
                raise RuntimeError("GBnetwork cache expects a fixed topology")
            b, nbus = len(sizes), int(sizes[0])
            pred = forward(batch).detach().to(torch.float64).reshape(b, nbus, 2)
            vstart = batch["V_start"].to(device=device, dtype=torch.float64).reshape(b, nbus, 2)
            # Collation stores fixed grid vectors as a packed [1, B*N] block
            # for this dataset.  Extract exactly one scenario's bus-type map;
            # indexing `[0]` alone would retain all B packed scenarios.
            bt = batch["bus_type"].reshape(-1)[:nbus].to(device=device)
            raw = restore_known(pred, vstart, bt)
            cal = calibrated(pred, dvm, dth)
            vm, th = project_state(cal[..., 0], cal[..., 1], (uv, ut, vbar, tbar))
            csp = restore_known(torch.stack((vm, th), dim=-1), vstart, bt)
            restore_changes["raw_max_abs"] = max(restore_changes["raw_max_abs"],
                                                   float((raw - pred).abs().max().cpu()))
            restore_changes["csp_max_abs"] = max(restore_changes["csp_max_abs"],
                                                   float((csp - torch.stack((vm, th), dim=-1)).abs().max().cpu()))
            if raw_mm is None:
                raw_mm = np.lib.format.open_memmap(outdir / "raw_restored.npy", mode="w+",
                                                   dtype=np.float64, shape=(n, nbus, 2))
                csp_mm = np.lib.format.open_memmap(outdir / "csp16_restored.npy", mode="w+",
                                                   dtype=np.float64, shape=(n, nbus, 2))
                sbus_mm = np.lib.format.open_memmap(outdir / "sbus.npy", mode="w+",
                                                    dtype=np.complex128, shape=(n, nbus))
                vstart_mm = np.lib.format.open_memmap(outdir / "vstart.npy", mode="w+",
                                                      dtype=np.float64, shape=(n, nbus, 2))
                static_bt = bt.cpu().numpy().astype(np.int64)
                yblock = batch["Ybus"].coalesce()
                yi, yj = yblock.indices()
                take = (yi < nbus) & (yj < nbus)
                static_y = torch.sparse_coo_tensor(
                    torch.stack((yi[take], yj[take])), yblock.values()[take],
                    (nbus, nbus), dtype=torch.complex128).to_dense().numpy()
            raw_mm[pos:pos+b] = raw.cpu().numpy()
            csp_mm[pos:pos+b] = csp.cpu().numpy()
            sbus_mm[pos:pos+b] = batch["S_start"].reshape(b, nbus).numpy().astype(np.complex128)
            vstart_mm[pos:pos+b] = vstart.cpu().numpy()
            pos += b
    if pos != n: raise RuntimeError(f"cache count mismatch: {pos} != {n}")
    raw_mm.flush(); csp_mm.flush(); sbus_mm.flush(); vstart_mm.flush()
    save_once(outdir / "bus_type.npy", static_bt)
    save_once(outdir / "ybus.npy", static_y)
    meta = {
        "model": args.model, "checkpoint": args.checkpoint, "rank": args.rank,
        "test_scenarios": n, "n_bus": nbus, "batch": args.batch,
        "split": "random_split seed=42; original membership, sorted evaluation order",
        "raw": "neural output, then prescribed PV/slack |V| and slack angle restored",
        "csp16": "train-only calibration + train-only magnitude/angle rank-16 bases, then prescribed coordinates restored",
        "restore_changes_max_abs": restore_changes,
        "cache_wall_seconds": time.perf_counter() - t0,
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
