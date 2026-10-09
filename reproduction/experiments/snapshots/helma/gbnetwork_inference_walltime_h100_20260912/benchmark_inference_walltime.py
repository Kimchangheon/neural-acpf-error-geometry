#!/usr/bin/env python3
"""H100 inference benchmark for the GBnetwork intervention table.

Dataset loading, checkpoint/model construction, train-only calibration fitting,
and train-only SVD fitting are excluded.  Timed regions include the model
forward and the requested frozen output transform, with CUDA synchronization.
No accuracy or residual metric is evaluated.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from collate_blockdiag_optimized_complex_columns import collate_blockdiag
from controlled_error_geometry import make_model, split_dataset
from diagnose_residual_distributions import fit_per_bus_offset, manifold_basis
from manifold_projection import project_state


def loader(ds, batch):
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0,
                      collate_fn=collate_blockdiag)


def count_parameters(model):
    return {
        "parameters_total": int(sum(p.numel() for p in model.parameters())),
        "parameters_trainable": int(sum(p.numel() for p in model.parameters()
                                        if p.requires_grad)),
        "parameter_bytes": int(sum(p.numel() * p.element_size()
                                   for p in model.parameters())),
    }


def one_forward(forward, batch, mode=None, offset=None, basis=None):
    with torch.inference_mode():
        out = forward(batch)
        if mode is not None:
            out = transform(out, mode, offset, basis)
        # Force materialisation before synchronize, without computing a metric.
        _ = out.shape
    torch.cuda.synchronize()


def try_batch(forward, test, batch_size, mode=None, offset=None, basis=None):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        batch = next(iter(loader(test, batch_size)))
        one_forward(forward, batch, mode, offset, basis)
        return True, {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "actual_batch": int(len(batch["sizes"])),
        }
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        return False, None


def probe(args, model, forward, train, test):
    # First bracket the failure by geometric growth, then find the exact largest
    # integer batch in the bracket.  The cap prevents pathological host-memory
    # pressure while remaining far above all training batches used here.
    cap = min(len(test), args.probe_cap)
    # The benchmark uses one common batch across Raw/C/P16/CSP16.  Probe the
    # highest-memory path, rather than raw forward alone, so a selected batch
    # cannot fail later when double-precision projection is enabled.
    offset = fit_per_bus_offset(model, forward, train, args.fit_batch,
                                torch.device("cuda"))
    Uv, Ut, vbar, tbar, _, _ = manifold_basis(train, args.fit_batch, k=16)
    basis = tuple(x.to(device="cuda", dtype=torch.float64)
                  for x in (Uv[:, :16], Ut[:, :16], vbar, tbar))
    success, failure, success_record, trials = 0, None, None, {}
    candidate = max(1, args.probe_start)
    while candidate <= cap:
        ok, rec = try_batch(forward, test, candidate, "CSP16", offset, basis)
        trials[str(candidate)] = {"success": ok, **(rec or {})}
        print(f"[probe] batch={candidate} success={ok} {rec or ''}", flush=True)
        if not ok:
            failure = candidate
            break
        success, success_record = candidate, rec
        if candidate == cap:
            break
        candidate = min(cap, candidate * 2)
    if success == 0:
        raise RuntimeError("Even batch size 1 failed")
    if failure is not None:
        lo, hi = success + 1, failure - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            ok, rec = try_batch(forward, test, mid, "CSP16", offset, basis)
            trials[str(mid)] = {"success": ok, **(rec or {})}
            print(f"[probe] batch={mid} success={ok} {rec or ''}", flush=True)
            if ok:
                success, success_record, lo = mid, rec, mid + 1
            else:
                hi = mid - 1
    # Do not rerun after an OOM trial: PyTorch's caching allocator may be
    # fragmented even after empty_cache(), yielding a false final failure.
    assert success_record is not None
    return {
        "probe_output": "CSP16",
        "max_successful_batch": success,
        "first_failed_batch": failure,
        "probe_cap": cap,
        "cap_reached_without_oom": failure is None and success == cap,
        "peak_at_selected_batch": success_record,
        "trials": trials,
    }


def transform(raw, mode, offset, basis):
    raw = raw.detach().to(torch.float64)
    sizes = raw.shape
    if mode in ("C", "CSP16"):
        nrep = raw.shape[1] // offset[0].numel()
        out = raw.clone()
        out[0, :, 0] -= offset[0].repeat(nrep)
        out[0, :, 1] = torch.atan2(
            torch.sin(out[0, :, 1] - offset[1].repeat(nrep)),
            torch.cos(out[0, :, 1] - offset[1].repeat(nrep)))
    else:
        out = raw
    if mode in ("P16", "CSP16"):
        nbus = offset[0].numel(); ns = out.shape[1] // nbus
        state = out[0].reshape(ns, nbus, 2)
        vm, th = project_state(state[..., 0], state[..., 1], basis)
        out = torch.stack((vm, th), dim=-1).reshape(*sizes)
    return out


def timed_pass(forward, test, batch_size, mode, offset, basis):
    elapsed = 0.0; scenarios = 0
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch in loader(test, batch_size):
            torch.cuda.synchronize()
            start = time.perf_counter()
            raw = forward(batch)
            if mode != "Raw":
                raw = transform(raw, mode, offset, basis)
            _ = raw.shape
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - start
            scenarios += len(batch["sizes"])
    return elapsed, scenarios, int(torch.cuda.max_memory_allocated()), \
        int(torch.cuda.max_memory_reserved())


def benchmark(args, model, forward, train, test):
    probed_batch = json.loads(Path(args.batch_json).read_text())["probe"]["max_successful_batch"]
    # A single-forward OOM boundary is not necessarily safe for a repeated
    # benchmark with warm-up/allocator history.  Preserve explicit headroom.
    batch_size = max(1, int(probed_batch * args.timing_batch_fraction))
    # These training-derived quantities are prepared before timing.
    offset = fit_per_bus_offset(model, forward, train, min(batch_size, args.fit_batch),
                                torch.device("cuda"))
    Uv, Ut, vbar, tbar, _, _ = manifold_basis(train, args.fit_batch, k=16)
    basis = tuple(x.to(device="cuda", dtype=torch.float64)
                  for x in (Uv[:, :16], Ut[:, :16], vbar, tbar))
    modes = ("Raw", "C", "P16", "CSP16")
    result = {}
    for mode in modes:
        # Mode-specific warm-up is excluded.
        first = next(iter(loader(test, batch_size)))
        with torch.inference_mode():
            warm = forward(first)
            if mode != "Raw": warm = transform(warm, mode, offset, basis)
            _ = warm.shape
        torch.cuda.synchronize()
        del warm, first
        gc.collect(); torch.cuda.empty_cache()
        reps = []
        for rep in range(args.repeats):
            sec, ns, alloc, reserved = timed_pass(
                forward, test, batch_size, mode, offset, basis)
            reps.append(sec)
            print(f"[time] mode={mode} repeat={rep} seconds={sec:.6f}", flush=True)
        mean = sum(reps) / len(reps)
        sd = math.sqrt(sum((x - mean) ** 2 for x in reps) / (len(reps) - 1)) \
            if len(reps) > 1 else 0.0
        result[mode] = {
            "seconds_per_testset_repeats": reps,
            "seconds_per_testset_mean": mean,
            "seconds_per_testset_sample_sd": sd,
            "milliseconds_per_scenario": 1000.0 * mean / ns,
            "scenarios": ns,
            "peak_allocated_bytes": alloc,
            "peak_reserved_bytes": reserved,
        }
    return batch_size, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("probe", "benchmark"), required=True)
    ap.add_argument("--model", choices=("g3", "gridsfm", "gridsfm_mask", "graphkit", "lumina"), required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--lumina-args", default="")
    ap.add_argument("--pignn-batched-armijo", action="store_true",
                    help="Opt in to the packed B=1 Armijo-candidate fast path.")
    ap.add_argument("--probe-start", type=int, default=16)
    ap.add_argument("--probe-cap", type=int, default=2048)
    ap.add_argument("--fit-batch", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--timing-batch-fraction", type=float, default=.90,
                    help="Fraction of the probe's maximum batch used in repeated timing.")
    ap.add_argument("--batch-json", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    train, _, test = split_dataset(args.parquet)
    model, forward = make_model(args.model, args.checkpoint, args.parquet,
                                args.fit_batch, torch.device("cuda"),
                                args.lumina_args,
                                pignn_batched_armijo=args.pignn_batched_armijo)
    meta = {
        "stage": args.stage, "model": args.model, "checkpoint": args.checkpoint,
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "test_scenarios": len(test), **count_parameters(model),
        "timing_scope": "GPU-synchronized model forward plus requested output transform; data loading, model loading, calibration fitting, and SVD fitting excluded",
        "pignn_batched_armijo": bool(args.pignn_batched_armijo),
    }
    if args.stage == "probe":
        meta["probe"] = probe(args, model, forward, train, test)
    else:
        if not args.batch_json: raise ValueError("--batch-json is required")
        bs, values = benchmark(args, model, forward, train, test)
        meta.update(batch_size=bs, repeats=args.repeats, timing=values)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
