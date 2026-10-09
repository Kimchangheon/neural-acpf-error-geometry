#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Find out WHY six grids fall short of 36,000 rows under the backbone preset.

The backbone preset moves the operating point with two independent knobs:

    global load scale   ~ U(0.60, 1.40)     one draw, applied to every load
    per-bus jitter      ~ N(1, sigma)       one draw PER LOAD, sigma_P=0.10, sigma_Q=0.15

Both push the operating point around, but they fail differently, and the
remedy differs too:

  * If yield collapses at the ENDS of the load-scale sweep, the grid is
    genuinely infeasible at those loadings. Oversampling then buys rows only
    from the feasible middle -- it silently truncates the input distribution.

  * If yield collapses as sigma grows at FIXED scale 1.0, the culprit is the
    per-bus draw. That mechanism is size-dependent: with n loads, the worst
    draw in a sample is an order statistic, so E[max] grows like sigma*sqrt(2 ln n).
    A 9,241-bus grid sees ~4.1 sigma excursions every single sample where a
    14-bus grid sees ~2.2 sigma. Same preset, wildly different tail exposure.
    Sign flips are the extreme case: q_mvar goes negative once z < -1/sigma_Q
    = -6.7, which never happens on a small grid and happens routinely on a
    large one.

Condition C therefore also reports how many loads flipped sign, which
distinguishes "hard but physical" from "the perturbation made a load into a
generator".

Run:  python diagnose_yield.py --grids case145,case300 --workers 72
"""
import argparse
import os
import sys
import time
import warnings
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

# The case generator imports pandapower function-locally throughout; keep the
# heavy imports out of module scope here too so worker startup stays cheap.


# Grids under investigation, with their measured backbone yield from the
# _ppNR_ campaign. The two 100% grids are controls: any mechanism proposed for
# the failures must also explain why these are fine.
GRID_YIELD = {
    "case145":        0.219,
    "case300":        0.778,
    "case1354pegase": 0.906,
    "case6470rte":    0.710,
    "case6495rte":    0.648,
    "case6515rte":    0.559,
    "case9241pegase": 0.508,
    # controls
    "case1888rte":    1.000,
    "case2869pegase": 1.000,
}

# Denser between 0.95 and 1.15: the case145 pilot put a hard feasibility cliff
# in there (100% at 1.00, 0% at 1.10), and a coarse grid cannot locate it.
SCALE_GRID = [0.60, 0.70, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.40]

# (jitter_load, jitter_load_q, jitter_gen) -- backbone is the third entry.
JITTER_GRID = [
    (0.00, 0.000, 0.00),
    (0.05, 0.075, 0.025),
    (0.10, 0.150, 0.050),
    (0.15, 0.225, 0.075),
]


def _base_kwargs(grid):
    import pandapower.networks as pn
    return dict(
        case_fn=getattr(pn, grid),
        case_kwargs={},
        ybus_mode="ppcY",
        start_mode="dc_compile",
        scale_gen_with_load=True,
        rand_u_start=True,
        angle_jitter_deg=5.0,
        mag_jitter_pq=0.02,
        line_outage_prob=0.0,
        return_pp_solution=True,
    )


def _run_one(job):
    """Solve one scenario. Returns a dict; never raises.

    Errors are classified rather than swallowed: a structurally singular
    Jacobian is a different finding from an iteration that ran out of steps,
    and collapsing them into 'not converged' is exactly what hid the case145
    story in the first place.
    """
    grid, cond, seed, scale, jl, jq, jg, pv = job
    import case_generator_all_test_cases_pandapower_consider_ppc_branch_row as cg

    # Condition A passes the real (lo, hi) band; B and C pin a single scale.
    rng_range = (tuple(float(x) for x in scale) if isinstance(scale, (tuple, list))
                 else (float(scale), float(scale)))

    kw = _base_kwargs(grid)
    kw.update(
        seed=int(seed),
        load_scale_range=rng_range,
        jitter_load=float(jl),
        jitter_load_q=float(jq),
        jitter_gen=float(jg),
        pv_vset_range=pv,
    )

    rec = dict(grid=grid, cond=cond, scale=scale, jl=jl, jq=jq, jg=jg,
               converged=0, err="", t=0.0)
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = cg.case_generation_pandapower(**kw)
            rec["converged"] = 1 if bool(out[27]) else 0
            rec["connected"] = bool(out[5])
        except Exception as exc:
            rec["err"] = f"{type(exc).__name__}: {exc}"[:120]
        msgs = " | ".join(str(w.message)[:80] for w in caught)
    if "singular" in msgs.lower():
        rec["err"] = rec["err"] or "SINGULAR"
    rec["t"] = time.time() - t0
    return rec


def _sign_flip_stats(grid, sigma_q, n_draw=200, seed=0):
    """How often does per-bus Q jitter flip a load's sign?

    This is pure sampling statistics -- no power flow -- so it is cheap and
    exact. A flip turns a reactive consumer into a reactive source, which is
    not a perturbation of the operating point but a change of what the device
    IS.
    """
    import pandapower.networks as pn
    net = getattr(pn, grid)()
    nload = len(net.load)
    if nload == 0 or sigma_q <= 0:
        return dict(n_load=nload, p_any_flip=0.0, mean_flips=0.0, max_z=0.0)
    rng = np.random.default_rng(seed)
    s = rng.normal(1.0, sigma_q, size=(n_draw, nload))
    flips = (s < 0.0).sum(axis=1)
    return dict(
        n_load=nload,
        p_any_flip=float((flips > 0).mean()),
        mean_flips=float(flips.mean()),
        # typical worst-case excursion in units of sigma, per sample
        max_z=float(np.mean((s.max(axis=1) - 1.0) / sigma_q)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=str, required=True)
    ap.add_argument("--workers", type=int, default=72)
    ap.add_argument("--n_scale", type=int, default=40,
                    help="samples per load-scale bin (condition B)")
    ap.add_argument("--n_jitter", type=int, default=40,
                    help="samples per jitter level (condition C)")
    ap.add_argument("--n_base", type=int, default=200,
                    help="samples for the full backbone baseline (condition A)")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    grids = [g.strip() for g in args.grids.split(",") if g.strip()]
    PV = (0.95, 1.05)

    jobs = []
    rng = np.random.default_rng(12345)

    def seeds(n):
        return rng.integers(0, 2**31 - 1, size=n)

    for g in grids:
        # A: the actual backbone preset, to confirm we reproduce the campaign yield
        for s in seeds(args.n_base):
            jobs.append((g, "A_backbone", s, None, 0.10, 0.15, 0.05, PV))
        # B: load scale swept, ALL jitter off -> isolates global infeasibility
        for sc in SCALE_GRID:
            for s in seeds(args.n_scale):
                jobs.append((g, f"B_scale_{sc:.2f}", s, sc, 0.0, 0.0, 0.0, (1.0, 1.0)))
        # C: jitter swept at scale 1.0 -> isolates the per-bus draw
        for (jl, jq, jg) in JITTER_GRID:
            for s in seeds(args.n_jitter):
                jobs.append((g, f"C_jit_{jl:.2f}", s, 1.00, jl, jq, jg, PV))

    # Condition A uses the real U(0.6,1.4) draw, which _run_one cannot express
    # via a fixed (s,s) range; substitute the true range there.
    jobs = [(g, c, s, (0.60, 1.40) if sc is None else sc, jl, jq, jg, pv)
            for (g, c, s, sc, jl, jq, jg, pv) in jobs]

    print(f"[diag] {len(jobs)} scenarios over {len(grids)} grids, {args.workers} workers",
          flush=True)

    import multiprocessing as mp
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        recs = pool.map(_run_one, jobs, chunksize=1)
    print(f"[diag] done in {time.time()-t0:.0f}s", flush=True)

    agg = defaultdict(lambda: [0, 0, defaultdict(int), 0.0])
    for r in recs:
        k = (r["grid"], r["cond"])
        agg[k][0] += 1
        agg[k][1] += r["converged"]
        if r["err"]:
            agg[k][2][r["err"].split(":")[0]] += 1
        agg[k][3] += r["t"]

    lines = []
    for g in grids:
        lines.append(f"\n{'='*78}\n{g}   (campaign yield {GRID_YIELD.get(g, float('nan')):.1%})\n{'='*78}")
        for sig in (0.10, 0.15):
            st = _sign_flip_stats(g, sig)
            lines.append(f"  sigma_Q={sig}: n_load={st['n_load']:5d}  "
                         f"P(>=1 sign flip per sample)={st['p_any_flip']:.3f}  "
                         f"mean flips={st['mean_flips']:.2f}  "
                         f"mean max-excursion={st['max_z']:.2f} sigma")
        lines.append(f"  {'condition':<18}{'n':>5}{'conv':>7}{'yield':>9}   errors        s/sample")
        for (gg, cond) in sorted(agg):
            if gg != g:
                continue
            n, c, errs, tt = agg[(gg, cond)]
            e = ",".join(f"{k}x{v}" for k, v in sorted(errs.items())) or "-"
            lines.append(f"  {cond:<18}{n:>5}{c:>7}{c/max(n,1):>8.1%}   {e:<14}{tt/max(n,1):>6.2f}")

    txt = "\n".join(lines)
    print(txt, flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(txt + "\n")
        print(f"[diag] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
