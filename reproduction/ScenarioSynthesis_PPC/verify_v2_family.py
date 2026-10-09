#!/usr/bin/env python3
"""Verify the v2 _ppNR_ corpus before the old generations are deleted.

Checks each of the three things v2 changed actually landed in the stored rows,
rather than trusting that the flags were passed:

  clean start      u_start comes straight from a DC solve, which sets every PQ
                   bus magnitude to exactly 1.0 pu. Any jitter would show up as
                   scatter around that. This is an exact check, not a threshold.

  pv_q_from_vstart the PV-bus reactive entry of S_start must equal
                   Im[u_start conj(Y u_start)] to solver precision, with Y
                   rebuilt from the stored branch columns the same way the
                   dataloader does it.

  perturb_sgen     total active injection must actually move across rows. On a
                   grid whose generation is mostly sgen this is the difference
                   between a real +/-40% sweep and a 2.2% one, so the spread is
                   reported per grid rather than asserted.

Plus the basics: row count, converged flag, bus count, and the four column
names the training loader requires.
"""
import glob
import io
import os
import sys

import numpy as np
import pyarrow.parquet as pq

import case_generator_all_test_cases_pandapower_consider_ppc_branch_row as cg

OUT = "/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out"
TAG = "u0clean_sgenpert_pvqvstart"
REQUIRED = ("u_start", "u_newton", "S_start", "S_newton")


def L(b):
    return np.load(io.BytesIO(b), allow_pickle=False)


def check(path, n_rows_probe=3):
    pf = pq.ParquetFile(path)
    md = pf.metadata
    cols = set(pf.schema_arrow.names)
    t = pf.read_row_group(0)

    res = dict(rows=md.num_rows, missing=[c for c in REQUIRED if c not in cols])

    g = lambda c, i=0: L(t[c][i].as_py())
    bt = g("bus_typ")
    res["buses"] = len(bt)

    # --- clean start: PQ magnitudes must be exactly flat 1.0 pu -------------
    us = g("u_start")
    vb = g("vn_kv") * 1e3 if "vn_kv" in cols else None
    if vb is None:
        res["pq_mag_dev"] = float("nan")
    else:
        pqm = np.abs(us[bt == 3]) / vb[bt == 3]
        res["pq_mag_dev"] = float(np.abs(pqm - 1.0).max()) if pqm.size else 0.0

    # --- pv_q_from_vstart ---------------------------------------------------
    Y = cg.reconstruct_Y_pandapower_branchrows_direct_SI(
        N=len(bt),
        Branch_f_bus=g("Branch_f_bus"), Branch_t_bus=g("Branch_t_bus"),
        Branch_status=g("Branch_status"), Branch_tau=g("Branch_tau"),
        Branch_shift_deg=g("Branch_shift_deg"),
        Branch_y_series_from=g("Branch_y_series_from"),
        Branch_y_series_to=g("Branch_y_series_to"),
        Branch_y_series_ft=g("Branch_y_series_ft"),
        Branch_y_shunt_from=g("Branch_y_shunt_from"),
        Branch_y_shunt_to=g("Branch_y_shunt_to"),
        Y_shunt_bus=g("Y_shunt_bus"),
        Vbase_bus=vb,
    )
    ss = g("S_start")
    pv = np.flatnonzero(bt == 2)
    if pv.size:
        q_want = (us * np.conj(Y @ us)).imag[pv]
        denom = max(np.abs(q_want).max(), 1e-30)
        res["pvq_rel"] = float(np.abs(ss.imag[pv] - q_want).max() / denom)
    else:
        res["pvq_rel"] = None          # no PV buses (SimBench)
    res["n_pv"] = int(pv.size)

    # --- perturb_sgen: does total injection actually move? ------------------
    n = min(n_rows_probe * 40, md.num_rows)
    tot = []
    tb = pq.read_table(path, columns=["S_start"]).slice(0, n)
    for i in range(n):
        tot.append(L(tb["S_start"][i].as_py()).real.sum() / 1e6)
    tot = np.array(tot)
    res["p_spread"] = float((tot.max() - tot.min()) / max(abs(tot.mean()), 1e-9))

    conv = np.asarray(pq.read_table(path, columns=["converged"])["converged"].to_pylist())
    res["conv_all"] = bool((conv == 1).all())
    return res


def main():
    files = sorted(glob.glob(f"{OUT}/*{TAG}*directSI.parquet"))
    print(f"{len(files)} files with tag {TAG}\n")
    print(f"{'grid':<24}{'rows':>8}{'bus':>6}{'n_pv':>6}{'PQ|V|dev':>10}"
          f"{'PVQ rel':>10}{'P spread':>10}  conv  cols")
    bad = []
    for f in files:
        grid = os.path.basename(f).split("_ppcY_")[0]
        try:
            r = check(f)
        except Exception as exc:
            print(f"{grid:<24}  ERROR {type(exc).__name__}: {exc}")
            bad.append(grid)
            continue
        pvq = "n/a" if r["pvq_rel"] is None else f"{r['pvq_rel']:.1e}"
        ok_cols = "ok" if not r["missing"] else ",".join(r["missing"])
        flag = ""
        if r["pvq_rel"] is not None and r["pvq_rel"] > 1e-9:
            flag += " PVQ!"
        if r["pq_mag_dev"] > 1e-9:
            flag += " JITTER!"
        if not r["conv_all"] or r["missing"] or r["rows"] < 36000:
            flag += " FAIL!"
        if flag:
            bad.append(grid + flag)
        print(f"{grid:<24}{r['rows']:>8,}{r['buses']:>6}{r['n_pv']:>6}"
              f"{r['pq_mag_dev']:>10.1e}{pvq:>10}{r['p_spread']:>9.1%}"
              f"  {'y' if r['conv_all'] else 'N'}   {ok_cols}{flag}")
    print()
    if bad:
        print("PROBLEMS: " + "; ".join(bad))
        sys.exit(1)
    print(f"all {len(files)} files pass")


if __name__ == "__main__":
    main()
