"""Predict on OPFData scenarios via torch_geometric.datasets.OPFDataset.

Usage:
    python examples/opfdata.py --case pglib_opf_case500_goc
    python examples/opfdata.py --case pglib_opf_case2000_goc --batch-size 128 --gpu 3

Iterates the entire `--split` in chunks of `--batch-size` and reports
aggregate per-bus / per-generator MAE against ground truth.

NOTE: GridSFM-Open is trained on grids with >=500 buses. Smaller cases
(case14_ieee, case30_ieee, case57_ieee, case118_ieee) are out of
distribution; results are not meaningful.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch_geometric.datasets import OPFDataset
from torch_geometric.utils import unbatch

from gridsfm import batch_data_list, load_model, prepare_for_inference
from gridsfm.schema import GEN_CP0_IDX, GEN_CP1_IDX, GEN_CP2_IDX

ROOT = Path(__file__).parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="pglib_opf_case500_goc")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--root", default=str(Path.home() / ".cache" / "opfdata"))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on scenarios processed (default: all in split)")
    ap.add_argument("--num-groups", type=int, default=1,
                    help="OPFDataset num_groups (1..20); cache dir is processed_<N>")
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "gridsfm_open_v1.1.pt"))
    ap.add_argument("--gpu", type=int, default=-1, help="GPU index (-1 = CPU)")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        sys.exit(
            f"Checkpoint not found at {ckpt}. Download with:\n"
            f"  hf download microsoft/GridSFM_Open gridsfm_open_v1.1.pt "
            f"--local-dir {ROOT / 'checkpoints'}"
        )

    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    print(f"loading OPFDataset(case={args.case}, split={args.split}, "
          f"root={args.root}, num_groups={args.num_groups})  device={device}")
    ds = OPFDataset(root=args.root, case_name=args.case, split=args.split,
                    num_groups=args.num_groups)
    N = len(ds) if args.limit is None else min(args.limit, len(ds))
    n_chunks = math.ceil(N / args.batch_size)
    print(f"  {len(ds)} scenarios in split; running {N} in {n_chunks} chunks "
          f"of {args.batch_size}")

    model = load_model(str(ckpt), device=device)

    V_se = th_se = Pg_se = Qg_se = 0.0
    n_bus = n_gen = 0
    feas_sum = 0.0
    cost_mape_list = []
    t0 = time.perf_counter()
    for chunk_start in range(0, N, args.batch_size):
        chunk_end = min(chunk_start + args.batch_size, N)
        prepared = [prepare_for_inference(ds[i]) for i in range(chunk_start, chunk_end)]
        batch = batch_data_list(prepared).to(device)
        with torch.no_grad():
            out = model(batch)
        bus_pred = out['bus'].pred.cpu()
        gen_pred = out['generator'].pred.cpu()
        gen_batch = out['generator'].batch.cpu()
        gen_x = batch['generator'].x.cpu()
        V_gt   = batch['bus'].y[:, 1].cpu()
        th_gt  = batch['bus'].y[:, 0].cpu()
        Pg_gt  = batch['generator'].y[:, 0].cpu()
        Qg_gt  = batch['generator'].y[:, 1].cpu()
        V_se  += (bus_pred[:, 1] - V_gt ).abs().sum().item()
        th_se += (bus_pred[:, 0] - th_gt).abs().sum().item()
        Pg_se += (gen_pred[:, 0] - Pg_gt).abs().sum().item()
        Qg_se += (gen_pred[:, 1] - Qg_gt).abs().sum().item()
        n_bus += int(bus_pred.size(0))
        n_gen += int(gen_pred.size(0))
        feas_sum += float(torch.sigmoid(out.feas_logit).sum().item())

        c2 = gen_x[:, GEN_CP2_IDX]
        c1 = gen_x[:, GEN_CP1_IDX]
        c0 = gen_x[:, GEN_CP0_IDX]
        Pg_pred = gen_pred[:, 0]
        cost_pred_per_gen = c2 * Pg_pred ** 2 + c1 * Pg_pred + c0
        cost_gt_per_gen   = c2 * Pg_gt   ** 2 + c1 * Pg_gt   + c0
        for cp_g, cg_g in zip(unbatch(cost_pred_per_gen, gen_batch),
                              unbatch(cost_gt_per_gen,   gen_batch)):
            cg = cg_g.sum().item()
            cp = cp_g.sum().item()
            if abs(cg) > 1.0:
                cost_mape_list.append(abs(cp - cg) / abs(cg) * 100.0)

        print(f"  chunk {chunk_start // args.batch_size + 1}/{n_chunks}: "
              f"scenarios [{chunk_start},{chunk_end})  "
              f"elapsed {time.perf_counter() - t0:.1f}s")

    V_mae  = V_se  / n_bus
    th_mae = th_se / n_bus
    Pg_mae = Pg_se / n_gen
    Qg_mae = Qg_se / n_gen
    feas_mean = feas_sum / N

    print(f"\n=== aggregate over {N} scenarios on {args.case} / {args.split} ===")
    print(f"  buses evaluated:      {n_bus:>10d}")
    print(f"  generators evaluated: {n_gen:>10d}")
    print(f"  mean predicted feas:  {feas_mean:.4f}")
    print(f"  V_mae       {V_mae:.4f} pu   ({V_mae*100:.2f}% of nominal voltage / bus)")
    print(f"  theta_mae   {th_mae:.4f} rad ({th_mae*180/math.pi:.2f}° / bus)")
    print(f"  Pg_mae      {Pg_mae:.4f} pu   ({Pg_mae*100:.2f} MW / gen at 100 MVA base)")
    print(f"  Qg_mae      {Qg_mae:.4f} pu   ({Qg_mae*100:.2f} MVAr / gen at 100 MVA base)")
    if cost_mape_list:
        cm = torch.tensor(cost_mape_list)
        print(f"  cost_mape   mean={cm.mean():.2f}%  median={cm.median():.2f}%  "
              f"max={cm.max():.2f}%  (n={len(cost_mape_list)} feasible scenarios)")
    print(f"  total time: {time.perf_counter() - t0:.1f}s on {device}")


if __name__ == "__main__":
    main()
