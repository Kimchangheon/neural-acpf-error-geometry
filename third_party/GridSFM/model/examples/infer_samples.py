"""Batched inference + per-case metrics on shipped samples (diagnostic).

Run from the `model/` directory:
    python examples/infer_samples.py             # CPU, default ckpt
    python examples/infer_samples.py --gpu 0     # GPU 0
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as stats
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import unbatch

from gridsfm import batch_data_list, load_model, load_pyg_json, prepare_for_inference
from gridsfm.schema import GEN_CP0_IDX, GEN_CP1_IDX, GEN_CP2_IDX

ROOT = Path(__file__).parent.parent


_SOLVED_STATUSES = {
    "LOCALLY_SOLVED", "OPTIMAL",
     "ALMOST_OPTIMAL",
}


def load_ground_truth(sample_path: Path) -> Dict[str, Any]:
    with open(sample_path) as f:
        obj = json.load(f)
    sol = obj["solution"]["nodes"]
    md = obj["metadata"]

    bus_y = torch.tensor(sol["bus"],       dtype=torch.float32)
    gen_y = torch.tensor(sol["generator"], dtype=torch.float32)
    gen_x = torch.tensor(obj["grid"]["nodes"]["generator"], dtype=torch.float32)

    c2 = gen_x[:, GEN_CP2_IDX]
    c1 = gen_x[:, GEN_CP1_IDX]
    c0 = gen_x[:, GEN_CP0_IDX]
    Pg_gt = gen_y[:, 0]
    cost_gt = float((c2 * Pg_gt ** 2 + c1 * Pg_gt + c0).sum().item())

    ts = md.get("termination_status", "")
    feasible = ts.upper() in _SOLVED_STATUSES if ts else bool(md.get("feasible", False))

    return {
        "theta_gt":  bus_y[:, 0], "V_gt":  bus_y[:, 1],
        "Pg_gt":     Pg_gt,       "Qg_gt": gen_y[:, 1],
        "c2": c2, "c1": c1, "c0": c0,
        "cost_gt":   cost_gt,
        "feasible":  feasible,
    }


def split_per_case(out: HeteroData, n_graphs: int) -> List[Dict[str, Any]]:
    bus_pred  = out["bus"].pred
    gen_pred  = out["generator"].pred
    bus_batch = out["bus"].batch
    gen_batch = out["generator"].batch
    feas      = torch.sigmoid(out.feas_logit)
    theta_split = unbatch(bus_pred[:, 0], bus_batch)
    V_split     = unbatch(bus_pred[:, 1], bus_batch)
    Pg_split    = unbatch(gen_pred[:, 0], gen_batch)
    Qg_split    = unbatch(gen_pred[:, 1], gen_batch)
    return [
        {
            "theta": theta_split[g].cpu(),
            "V":     V_split[g].cpu(),
            "Pg":    Pg_split[g].cpu(),
            "Qg":    Qg_split[g].cpu(),
            "feas":  float(feas[g].cpu()),
        }
        for g in range(n_graphs)
    ]


def per_case_metrics(pred: Dict[str, Any], gt: Dict[str, Any]) -> Dict[str, Any]:
    V_mae     = (pred["V"]     - gt["V_gt"]   ).abs().mean().item()
    theta_mae = (pred["theta"] - gt["theta_gt"]).abs().mean().item()
    Pg_mae    = (pred["Pg"]    - gt["Pg_gt"]  ).abs().mean().item()
    Qg_mae    = (pred["Qg"]    - gt["Qg_gt"]  ).abs().mean().item()
    cost_mape = None
    if gt["feasible"] and abs(gt["cost_gt"]) > 1.0:
        cost_pred = float(
            (gt["c2"] * pred["Pg"] ** 2 + gt["c1"] * pred["Pg"] + gt["c0"]).sum().item()
        )
        cost_mape = abs(cost_pred - gt["cost_gt"]) / abs(gt["cost_gt"]) * 100
    return {
        "V_mae": V_mae, "theta_mae": theta_mae,
        "Pg_mae": Pg_mae, "Qg_mae": Qg_mae,
        "cost_mape": cost_mape,
        "feas_pred":   pred["feas"],
        "feas_label":  int(gt["feasible"]),
        "feas_correct": int((pred["feas"] >= 0.5) == gt["feasible"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_path", nargs="?",
                    default=str(ROOT / "checkpoints" / "gridsfm_open_v1.1.pt"))
    ap.add_argument("--gpu", type=int, default=-1, help="GPU index (-1 = CPU).")
    args = ap.parse_args()

    samples_dir = ROOT / "samples"
    samples = sorted(samples_dir.glob("*.pyg.json"))
    if not samples:
        sys.exit(f"No samples in {samples_dir}")

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        sys.exit(
            f"Checkpoint not found at {ckpt_path}. Download with:\n"
            f"  hf download microsoft/GridSFM_Open gridsfm_open_v1.1.pt "
            f"--local-dir {ROOT / 'checkpoints'}"
        )

    if args.gpu >= 0 and not torch.cuda.is_available():
        print(f"Requested --gpu {args.gpu} but CUDA is not available; falling back to CPU.")
    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    print(f"Loading {ckpt_path}  (device={device})")
    model = load_model(str(ckpt_path), device=device)

    print(f"Preparing {len(samples)} samples (cycle basis + PE features)...")
    t0 = time.perf_counter()
    prepared = [prepare_for_inference(load_pyg_json(s)) for s in samples]
    gts = [load_ground_truth(s) for s in samples]
    t_prep = time.perf_counter() - t0

    print(f"Batched forward on {len(samples)} mixed-topology graphs...")
    t0 = time.perf_counter()
    batch = batch_data_list(prepared).to(device)
    with torch.no_grad():
        out = model(batch)
    t_fwd = time.perf_counter() - t0

    rows = split_per_case(out, n_graphs=len(samples))
    metrics = [per_case_metrics(p, g) for p, g in zip(rows, gts)]

    print(f"\n{'case':30s}  {'V_mae':>7s}  {'th_mae':>7s}  {'Pg_mae':>7s}  "
          f"{'Qg_mae':>7s}  {'cost%':>6s}  {'feas_pred':>9s}  {'feas_label':>10s}")
    print("-" * 105)
    for s, m in zip(samples, metrics):
        case = s.stem.replace(".pyg", "")
        cost_str = f"{m['cost_mape']:5.2f}%" if m["cost_mape"] is not None else "  N/A"
        print(f"{case:30s}  {m['V_mae']:7.4f}  {m['theta_mae']:7.4f}  "
              f"{m['Pg_mae']:7.4f}  {m['Qg_mae']:7.4f}  {cost_str:>6s}  "
              f"{m['feas_pred']:9.3f}  {m['feas_label']:>10d}")
    print("-" * 105)

    n = len(metrics)
    feas_n = sum(m["feas_label"] for m in metrics)
    cost_vals = [m["cost_mape"] for m in metrics if m["cost_mape"] is not None]
    n_correct = sum(m["feas_correct"] for m in metrics)

    print(f"\nAGGREGATE")
    print(f"  Samples: {n} total | {feas_n} feasible | {n - feas_n} infeasible")
    if cost_vals:
        print(f"  Cost MAPE (n={len(cost_vals)}): mean={stats.mean(cost_vals):.2f}%  "
              f"median={stats.median(cost_vals):.2f}%  max={max(cost_vals):.2f}%")
    print(f"  V_MAE     mean = {stats.mean(m['V_mae']     for m in metrics):.4f}")
    th_mean_rad = stats.mean(m['theta_mae'] for m in metrics)
    print(f"  theta_MAE mean = {th_mean_rad:.4f} rad ({th_mean_rad * 180.0 / math.pi:.4f} deg)")
    print(f"  Pg_MAE    mean = {stats.mean(m['Pg_mae']    for m in metrics):.4f} pu")
    print(f"  Qg_MAE    mean = {stats.mean(m['Qg_mae']    for m in metrics):.4f} pu")
    print(f"  Feas accuracy: {n_correct}/{n} = {n_correct / n * 100:.1f}%")
    print(f"  Time: prep={t_prep:.1f}s  forward={t_fwd:.1f}s ({device})")


if __name__ == "__main__":
    main()
