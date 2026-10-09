#!/usr/bin/env python3
"""Numerical equivalence check for the opt-in packed Armijo fast path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from collate_blockdiag_optimized_complex_columns import collate_blockdiag
from controlled_error_geometry import make_model, split_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--parquet", required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--full-test", action="store_true",
                   help="Compare every held-out scenario rather than one batch.")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    device = torch.device("cuda")
    _, _, test = split_dataset(args.parquet)
    batches = DataLoader(
        test, batch_size=args.batch, shuffle=False, num_workers=0,
        collate_fn=collate_blockdiag,
    )
    serial, forward_serial = make_model(
        "g3", args.checkpoint, args.parquet, args.batch, device,
        pignn_batched_armijo=False,
    )
    fast, forward_fast = make_model(
        "g3", args.checkpoint, args.parquet, args.batch, device,
        pignn_batched_armijo=True,
    )
    n_scenarios = 0
    sum_dv = sum_dt = 0.0
    max_dv = max_dt = 0.0
    for batch_index, batch in enumerate(batches):
        if batch_index > 0 and not args.full_test:
            break
        with torch.inference_mode():
            y_serial = forward_serial(batch).detach()
            torch.cuda.synchronize()
            y_fast = forward_fast(batch).detach()
            torch.cuda.synchronize()
        dv = (y_serial[..., 0] - y_fast[..., 0]).abs()
        dtheta = torch.atan2(
            torch.sin(y_serial[..., 1] - y_fast[..., 1]),
            torch.cos(y_serial[..., 1] - y_fast[..., 1]),
        ).abs()
        n_scenarios += int(len(batch["sizes"]))
        sum_dv += float(dv.sum().cpu())
        sum_dt += float(dtheta.sum().cpu())
        max_dv = max(max_dv, float(dv.max().cpu()))
        max_dt = max(max_dt, float(dtheta.max().cpu()))
        del y_serial, y_fast, dv, dtheta
    result = {
        "checkpoint": args.checkpoint,
        "compared_scenarios": n_scenarios,
        "full_test": bool(args.full_test),
        "max_abs_voltage_difference": max_dv,
        "max_abs_angle_difference_rad": max_dt,
        "mean_abs_voltage_difference": sum_dv / (n_scenarios * 2224),
        "mean_abs_angle_difference_rad": sum_dt / (n_scenarios * 2224),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
