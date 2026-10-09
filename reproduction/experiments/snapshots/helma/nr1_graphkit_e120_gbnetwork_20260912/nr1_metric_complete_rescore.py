#!/usr/bin/env python3
"""Metric-complete post-hoc scorer for frozen C/CSP16 + one NR step.

It deliberately reuses the cached C/CSP16 states from the validated NR1 run:
there is no neural inference, retraining, data generation, or test-time tuning.
The custom Newton functions and the train-only CSP state are exactly those of
``nr1_multiprocess_baseline.py``.  It writes corrected states, then scores all
four states with the existing complex128 PB convention.
"""
from __future__ import annotations

import argparse, csv, json, math, multiprocessing as mp, os, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from collate_blockdiag_optimized_complex_columns import collate_blockdiag, ybus_matvec
from controlled_error_geometry import pb_per_scenario, split_dataset
from diagnose_residual_distributions import angle_diff

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "ScenarioSynthesis_PPC"))
from newton_raphson_improved import JacobianMatrix3p, VoltageCalculation3p

G = {}


def _init(state_path, s_path, y_path, bt_path):
    global G
    G = {
        "state": np.load(state_path, mmap_mode="r"),
        "S": np.load(s_path, mmap_mode="r"),
        "Y": np.load(y_path),
        "bt": np.load(bt_path),
    }


def _nr_chunk(bounds):
    lo, hi, eta = bounds
    y, bt = G["Y"], G["bt"]
    out = np.empty((hi - lo, y.shape[0], 2), dtype=np.float64)
    failures = 0
    solve_seconds = 0.0
    for j, i in enumerate(range(lo, hi)):
        vth = G["state"][i]
        u0 = np.asarray(vth[:, 0]) * np.exp(1j * np.asarray(vth[:, 1]))
        s = np.asarray(G["S"][i])
        try:
            tic = time.perf_counter()
            jac = JacobianMatrix3p(y, u0)[0]
            u1, _ = VoltageCalculation3p(bt, jac, y, u0, s.real, s.imag)
            solve_seconds += time.perf_counter() - tic
        except np.linalg.LinAlgError:
            failures += 1
            u1 = u0
        a0, a1 = np.angle(u0), np.angle(u1)
        m0, m1 = np.abs(u0), np.abs(u1)
        slack = bt == 1
        pq = (bt != 1) & (bt != 2)
        ang, mag = a0.copy(), m0.copy()
        non_slack = ~slack
        ang[non_slack] += eta * np.angle(np.exp(1j * (a1[non_slack] - a0[non_slack])))
        mag[pq] += eta * (m1[pq] - m0[pq])
        out[j, :, 0], out[j, :, 1] = mag, ang
    return lo, out, failures, solve_seconds


def run_nr(state_path: Path, s_path: Path, y_path: Path, bt_path: Path,
           out_path: Path, eta: float, workers: int, chunk: int):
    n = np.load(state_path, mmap_mode="r").shape[0]
    nbus = np.load(state_path, mmap_mode="r").shape[1]
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype="float64", shape=(n, nbus, 2))
    tasks = [(i, min(i + chunk, n), eta) for i in range(0, n, chunk)]
    ctx = mp.get_context("fork")
    tic = time.perf_counter(); failures = 0; solve_seconds = 0.0
    with ctx.Pool(workers, initializer=_init,
                  initargs=(str(state_path), str(s_path), str(y_path), str(bt_path)),
                  maxtasksperchild=100) as pool:
        for lo, block, bad, sec in pool.imap_unordered(_nr_chunk, tasks, chunksize=1):
            out[lo:lo + len(block)] = block
            failures += bad; solve_seconds += sec
    out.flush()
    return {"wall_seconds": time.perf_counter() - tic,
            "sum_solve_seconds": solve_seconds, "failures": int(failures),
            "n_scenarios": n, "eta": eta}


def _new_record(nbus, device):
    return dict(pb=[], scen_max=[], max_pb=0.0, mag_sse=0.0, ang_sse=0.0, n=0,
                sum_p=torch.zeros(nbus, dtype=torch.float64, device=device),
                sum_r=torch.zeros(nbus, dtype=torch.float64, device=device),
                sum_p2=torch.zeros(nbus, dtype=torch.float64, device=device),
                sum_r2=torch.zeros(nbus, dtype=torch.float64, device=device),
                sum_pr=torch.zeros(nbus, dtype=torch.float64, device=device), n_scen=0)


def score_states(parquet: str, paths: dict[str, Path], batch_size: int, device: torch.device):
    _train, _valid, test = split_dataset(parquet)
    arrays = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
    n = len(test); nbus = arrays["C"].shape[1]
    assert all(x.shape == (n, nbus, 2) for x in arrays.values())
    rec = {name: _new_record(nbus, device) for name in arrays}
    loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_blockdiag)
    pos = 0
    with torch.no_grad():
        for batch in loader:
            ns = len(batch["sizes"]); assert int(batch["sizes"][0]) == nbus
            ref = batch["V_newton"].to(device=device, dtype=torch.float64)[0].reshape(ns, nbus, 2)
            y = batch["Ybus"].to(device=device, dtype=torch.complex128)
            s = batch["S_start"].to(device=device, dtype=torch.complex128)[0]
            bt = batch["bus_type"][0].to(device)
            p_mask, q_mask = bt != 1, (bt != 1) & (bt != 2)
            for name, arr in arrays.items():
                x = torch.from_numpy(np.array(arr[pos:pos + ns], copy=True)).to(device=device, dtype=torch.float64)
                v, theta = x[..., 0], x[..., 1]
                per, mx = pb_per_scenario(v, theta, batch, device, return_global_max=True)
                vc = v.reshape(-1) * torch.exp(1j * theta.reshape(-1))
                sc = vc * ybus_matvec(y, vc).conj()
                dp, dq = s.real - sc.real, s.imag - sc.imag
                rho = torch.sqrt((dp * p_mask) ** 2 + (dq * q_mask) ** 2).reshape(ns, nbus)
                r = rec[name]
                r["pb"].append(per); r["scen_max"].append(rho.max(1).values.cpu().numpy()); r["max_pb"] = max(r["max_pb"], mx)
                r["mag_sse"] += float((v - ref[..., 0]).square().sum()); r["ang_sse"] += float(angle_diff(theta, ref[..., 1]).square().sum()); r["n"] += int(v.numel())
                r["sum_p"] += v.sum(0); r["sum_r"] += ref[..., 0].sum(0); r["sum_p2"] += v.square().sum(0); r["sum_r2"] += ref[..., 0].square().sum(0); r["sum_pr"] += (v * ref[..., 0]).sum(0); r["n_scen"] += ns
            pos += ns
    assert pos == n
    out = {}
    for name, r in rec.items():
        ns = r["n_scen"]
        vp = (r["sum_p2"] - r["sum_p"].square() / ns).sum().item(); vr = (r["sum_r2"] - r["sum_r"].square() / ns).sum().item(); cov = (r["sum_pr"] - r["sum_p"] * r["sum_r"] / ns).sum().item()
        pb = np.concatenate(r["pb"]); sm = np.concatenate(r["scen_max"])
        out[name] = {"vmag_rmse": math.sqrt(r["mag_sse"] / r["n"]), "angle_rmse_deg": math.degrees(math.sqrt(r["ang_sse"] / r["n"])),
                     "within_r2": cov * cov / (vp * vr) if vp > 0 and vr > 0 else 0.0, "within_slope": cov / vr if vr > 0 else float("nan"),
                     "mean_pb": float(pb.mean()), "max_pb": r["max_pb"],
                     "scenario_max_pb_p50": float(np.quantile(sm, .50)), "scenario_max_pb_p95": float(np.quantile(sm, .95)), "scenario_max_pb_p99": float(np.quantile(sm, .99))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--seed", required=True); ap.add_argument("--parquet", required=True)
    ap.add_argument("--cache-dir", required=True); ap.add_argument("--out-dir", required=True); ap.add_argument("--eta", type=float, required=True)
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--chunk", type=int, default=16)
    a = ap.parse_args(); os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    cd, od = Path(a.cache_dir), Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    paths = {"C": cd / "C.npy", "CSP16": cd / "CSP16.npy"}; assert all(p.exists() for p in paths.values())
    common = (cd / "S.npy", cd / "Y.npy", cd / "bt.npy"); assert all(p.exists() for p in common)
    runs = {}
    for src, dst in (("C", "C_NR1"), ("CSP16", "CSP16_NR1")):
        path = od / f"{dst}.npy"; runs[dst] = run_nr(paths[src], *common, path, a.eta, a.workers, a.chunk); paths[dst] = path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda": torch.cuda.synchronize()
    tic = time.perf_counter(); metrics = score_states(a.parquet, paths, a.batch, device)
    if device.type == "cuda": torch.cuda.synchronize()
    result = {"protocol": {"eta": a.eta, "input": "existing frozen C/CSP16 caches; no neural inference", "nr": "one custom Newton correction; PQ magnitude/non-slack angle updates", "pb": "existing complex128 GENCO structural-zero PB", "test_split": "seed=42"},
              "model": a.model, "seed": a.seed, "n_test": int(np.load(paths["C"], mmap_mode="r").shape[0]), "nr_runtime": runs, "metric_scoring_wall_seconds": time.perf_counter() - tic, "metrics": metrics}
    (od / "complete_metrics.json").write_text(json.dumps(result, indent=2))
    with (od / "complete_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "seed", "output"] + list(next(iter(metrics.values())).keys())); w.writeheader()
        for name, row in metrics.items(): w.writerow({"model": a.model, "seed": a.seed, "output": name, **row})


if __name__ == "__main__":
    main()
