#!/usr/bin/env python3
"""Build every table of the supplementary material from frozen JSON results.

No metric in the supplement is typed by hand: this script writes LaTeX table
bodies to ``<out>/tab_*.tex`` and the numbers quoted in the prose to
``<out>/summary.json``.  Missing inputs are reported, never imputed.

Inputs (all under --root, a snapshot of the cluster results):
  graphkit_31grid_ranks_20260922/{json,floor}  120-epoch GraphKit, k=16..128
  supp_rescore_20261002/{json,floor}           converged (+200 ep) six grids and
                                               the four GBnetwork checkpoints,
                                               every projection channel
  manifest_ppnr_v2.sh                          grid list with bus counts
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

RANKS = (16, 32, 64, 128)
KRON = {"LVN_heo1": ("kron120_LVN_heo1", "kron_e200_LVN_heo1")}
CONV = ("case89pegase", "case118", "case300", "LVN_heo1", "case1354pegase", "case1888rte")
GB = (("gb_base120", "120 ep (paper, seed 42)"), ("gb_lr1e4", r"+200 ep, LR $10^{-4}$"),
      ("gb_lr3e5", r"+200 ep, LR $3{\times}10^{-5}$"), ("gb_lr1e5", r"+200 ep, LR $10^{-5}$"))
SHORT = {"ENTSO_E_RealGridTest": "ENTSO-E", "case_illinois200": "illinois200",
         "case24_ieee_rts": "case24", "GBreducednetwork": "GBreduced"}


def tex(s):
    return SHORT.get(s, s).replace("_", r"\_")


def dec(x, nd):
    """Paper style: drop the leading zero below one (.07917)."""
    s = f"{x:.{nd}f}"
    return s[1:] if s.startswith("0.") else s


def sci(x):
    m, e = f"{x:.2e}".split("e")
    return rf"${m}{{\times}}10^{{{int(e)}}}$"


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def manifest(path: Path):
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith('"'):
            f = line.split('"')[1].split("|")
            out.append((f[0], int(f[1])))
    return out


def m_of(p: Path):
    d = load(p)
    return None if d is None else d["result"]["metrics"]


def floor_of(p: Path):
    d = load(p)
    return None if d is None else d["metrics"]


# --------------------------------------------------------------------------- #
# S3: cross-grid floor rule and rank behaviour (120 epochs)
# --------------------------------------------------------------------------- #

def crossgrid(root: Path, out: Path, summ: dict):
    rd = root / "graphkit_31grid_ranks_20260922"
    grids = manifest(root / "manifest_ppnr_v2.sh")
    rows, missing = [], []
    for g, n in grids:
        m = {k: m_of(rd / "json" / f"{g}_r{k}.json") for k in RANKS}
        f = {k: floor_of(rd / "floor" / f"{g}_r{k}.json") for k in RANKS}
        if any(v is None for v in m.values()) or any(v is None for v in f.values()):
            missing.append(g)
            continue
        rows.append((g, n, m, f))
    summ["crossgrid_missing"] = missing
    summ["crossgrid_n"] = len(rows)

    # Per-rank summary.  For k >= N the basis spans the whole state space, so
    # P_k is the identity and CSP_k reduces to calibration; those grids are
    # counted separately rather than credited to projection.
    lines, per_rank = [], {}
    for k in RANKS:
        act = [(g, n, m[k], f[k]) for g, n, m, f in rows if k < n]
        pb_imp = v_imp = both = agree = 0
        on_floor = []
        for g, n, mk, fk in act:
            raw, csp = mk["raw"], mk[f"CSP{k}"]
            fl = fk[f"P{k}_truth"]["mean_pb"]
            pi = csp["mean_pb"] < raw["mean_pb"]
            vi = csp["vmag_rmse"] < raw["vmag_rmse"]
            pb_imp += pi; v_imp += vi; both += pi and vi
            agree += (raw["mean_pb"] > fl) == pi
            on_floor.append(csp["mean_pb"] / fl)
        on_floor.sort()
        med = on_floor[len(on_floor) // 2] if on_floor else float("nan")
        idn = sum(1 for _, n, _, _ in rows if k >= n)
        per_rank[k] = dict(active=len(act), identity=idn, pb_improved=pb_imp,
                           v_improved=v_imp, both=both, rule_agree=agree,
                           median_csp_over_floor=med)
        lines.append(rf"{k} & {len(act)} & {idn} & {pb_imp} & {v_imp} & {both} & "
                     rf"{agree}/{len(act)} & {med:.2f} \\")
    # Calibration alone (rank-independent), for the prose.
    cal = [(m[16]["C"], m[16]["raw"]) for _, _, m, _ in rows]
    ratios = sorted(c["max_pb"] / r["max_pb"] for c, r in cal)
    summ["calibration_crossgrid"] = dict(
        n=len(cal), mean_pb_lower=sum(c["mean_pb"] < r["mean_pb"] for c, r in cal),
        vmag_lower=sum(c["vmag_rmse"] < r["vmag_rmse"] for c, r in cal),
        angle_lower=sum(c["angle_rmse_deg"] < r["angle_rmse_deg"] for c, r in cal),
        max_pb_ratio_median=ratios[len(ratios) // 2])
    (out / "tab_floor_rule.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")
    summ["floor_rule_by_rank"] = per_rank

    # Per-grid k=16 table (all scored grids).
    lines = []
    for g, n, m, f in rows:
        raw, csp = m[16]["raw"], m[16]["CSP16"]
        fl = f[16]["P16_truth"]["mean_pb"]
        # Mean PB in units of 1e-3.  N <= 16: P_16 is the identity (floor 0).
        mark = r"$^\ddagger$" if n <= 16 else ("" if raw["mean_pb"] > fl else r"$^\dagger$")
        k3 = lambda x: ("0" if x <= 0 else f"{1e3 * x:,.0f}" if 1e3 * x >= 1000 else f"{1e3 * x:.3g}")  # noqa: E731
        lines.append(rf"{tex(g)}{mark} & {n} & {k3(raw['mean_pb'])} & {k3(fl) if n > 16 else '--'} & "
                     rf"{k3(csp['mean_pb'])} & {csp['mean_pb'] / raw['mean_pb']:.2f} & "
                     rf"{csp['vmag_rmse'] / raw['vmag_rmse']:.2f} \\")
    (out / "tab_crossgrid_k16.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")

    # Large grids: PB/|V| trade-off across ranks (quoted in prose).
    trade = {}
    for g, n, m, f in rows:
        if n >= 1354:
            trade[g] = {k: dict(pb=m[k][f"CSP{k}"]["mean_pb"] / m[k]["raw"]["mean_pb"],
                                v=m[k][f"CSP{k}"]["vmag_rmse"] / m[k]["raw"]["vmag_rmse"])
                        for k in RANKS}
    summ["large_grid_tradeoff"] = trade


# --------------------------------------------------------------------------- #
# S1: converged GraphKit on six grids, and projection channels
# --------------------------------------------------------------------------- #

def converged(root: Path, out: Path, summ: dict):
    base = root / "graphkit_31grid_ranks_20260922"
    sr = root / "supp_rescore_20261002"
    lines_drift, lines_ch, missing, drift = [], [], [], {}
    for g in CONV:
        if g in KRON:
            # Kron-reduced model (exact reduction of the 328 pass-through
            # buses), scored on all buses; floors are the unreduced grid's.
            m120 = m_of(sr / "json" / f"{KRON[g][0]}_r16.json")
            mc = {k: m_of(sr / "json" / f"{KRON[g][1]}_r{k}.json") for k in RANKS}
        else:
            m120 = m_of(base / "json" / f"{g}_r16.json")
            mc = {k: m_of(sr / "json" / f"conv_{g}_r{k}.json") for k in RANKS}
        fc = {k: floor_of(sr / "floor" / f"{g}_r{k}.json") for k in RANKS}
        if m120 is None or any(v is None for v in mc.values()) or any(v is None for v in fc.values()):
            missing.append(g)
            continue
        fl16 = fc[16]["P16_truth"]["mean_pb"]
        drift[g] = {}
        for tag, m in (("120", m120), ("+200", mc[16])):
            raw, c, csp = m["raw"], m["C"], m["CSP16"]
            drift[g][tag] = dict(raw_pb=raw["mean_pb"], c_pb=c["mean_pb"], csp_pb=csp["mean_pb"],
                                 csp_over_raw=csp["mean_pb"] / raw["mean_pb"],
                                 c_over_raw=c["mean_pb"] / raw["mean_pb"],
                                 raw_over_floor=raw["mean_pb"] / fl16,
                                 raw_aw=raw["within_slope"], raw_v=raw["vmag_rmse"],
                                 csp_v=csp["vmag_rmse"])
            first = (tex(g) + (r"$^\ast$" if g in KRON else "")) if tag == "120" else ""
            lines_drift.append(
                rf"{first} & {tag} & {dec(raw['vmag_rmse'], 5)} & {dec(raw['within_slope'], 3)} & "
                rf"{dec(raw['mean_pb'], 4)} & {c['mean_pb'] / raw['mean_pb']:.2f} & "
                rf"{csp['mean_pb'] / raw['mean_pb']:.2f} & {raw['mean_pb'] / fl16:.2f} \\")
        lines_drift.append(r"\addlinespace[1pt]")
        nbus = dict(manifest(root / "manifest_ppnr_v2.sh"))[g]
        khi = 128 if nbus > 128 else 64          # k >= N would make P_k the identity
        for k in (16, khi):
            m, fk = mc[k], fc[k]
            rp = m["raw"]["mean_pb"]
            first = (tex(g) + (r"$^\ast$" if g in KRON else "")) if k == 16 else ""
            cells = [f"{m[s]['mean_pb'] / rp:.2f}" for s in ("C", f"CSP{k}", f"CSPmag{k}", f"CSPang{k}")]
            cells += [f"{fk[s]['mean_pb'] / rp:.2f}" for s in (f"P{k}_truth", f"Pmag{k}_truth", f"Pang{k}_truth")]
            lines_ch.append(rf"{first} & {k} & " + " & ".join(cells) +
                            rf" & {m[f'CSP{k}']['vmag_rmse'] / m['raw']['vmag_rmse']:.2f} \\")
        lines_ch.append(r"\addlinespace[1pt]")
    if lines_drift:
        (out / "tab_converged_drift.tex").write_text("\n".join(lines_drift[:-1]) + "\n\\bottomrule\n")
        (out / "tab_converged_channels.tex").write_text("\n".join(lines_ch[:-1]) + "\n\\bottomrule\n")
    summ["converged_missing"] = missing
    summ["converged_drift"] = drift


# --------------------------------------------------------------------------- #
# S2: GBnetwork convergence
# --------------------------------------------------------------------------- #

def gbnetwork(root: Path, out: Path, summ: dict):
    sr = root / "supp_rescore_20261002"
    ms = {n: {k: m_of(sr / "json" / f"{n}_r{k}.json") for k in RANKS} for n, _ in GB}
    fl = {k: floor_of(sr / "floor" / f"GBnetwork_r{k}.json") for k in RANKS}
    if any(v is None for d in ms.values() for v in d.values()) or any(v is None for v in fl.values()):
        summ["gb_missing"] = True
        return
    summ["gb_missing"] = False
    lines = []
    for n, label in GB:
        m = ms[n][16]
        for i, s in enumerate(("raw", "C", "P16", "CSP16")):
            x = m[s]
            name = {"raw": "Raw", "C": "$C$", "P16": "$P_{16}$", "CSP16": "CSP$_{16}$"}[s]
            first = rf"\multirow{{4}}{{*}}{{\shortstack[l]{{{label}}}}}" if i == 0 else ""
            lines.append(rf"{first} & {name} & {dec(x['vmag_rmse'], 5)} & {dec(x['angle_rmse_deg'], 3)} & "
                         rf"{dec(x['within_r2'], 3)} & {dec(x['within_slope'], 3)} & "
                         rf"{dec(x['mean_pb'], 5)} & {x['max_pb']:.2f} \\")
        lines.append(r"\midrule")
    (out / "tab_gb_full.tex").write_text("\n".join(lines[:-1]) + "\n\\bottomrule\n")

    lines, gbsum = [], {}
    for n, label in GB:
        gbsum[n] = {}
        for k in RANKS:
            m = ms[n][k]
            rp, cp = m["raw"]["mean_pb"], m["C"]["mean_pb"]
            vals = {s: m[f"{s}{k}"]["mean_pb"] for s in ("CSP", "CSPmag", "CSPang")}
            gbsum[n][k] = dict(raw=rp, C=cp, **vals,
                               floor=fl[k][f"P{k}_truth"]["mean_pb"],
                               floor_mag=fl[k][f"Pmag{k}_truth"]["mean_pb"])
            first = rf"\multirow{{4}}{{*}}{{\shortstack[l]{{{label}}}}}" if k == 16 else ""
            best = min(vals, key=vals.get)
            cells = []
            for s in ("CSP", "CSPmag", "CSPang"):
                c = f"{100 * (vals[s] / cp - 1):+.1f}"
                cells.append(rf"\textbf{{{c}}}" if s == best and vals[s] < cp else c)
            lines.append(rf"{first} & {k} & {dec(rp, 4)} & {dec(cp, 4)} & " + " & ".join(cells) +
                         rf" & {dec(fl[k][f'P{k}_truth']['mean_pb'], 4)} & "
                         rf"{dec(fl[k][f'Pmag{k}_truth']['mean_pb'], 4)} \\")
        lines.append(r"\midrule")
    (out / "tab_gb_rank.tex").write_text("\n".join(lines[:-1]) + "\n\\bottomrule\n")
    summ["gb"] = gbsum


def angle_only(root: Path, summ: dict):
    """Angle-only projection (120 ep, grids N >= 300): does it keep the PB gain?"""
    ad = root / "graphkit_angonly_20261002"
    res = {}
    for f in sorted((ad / "json").glob("*_r16.json")):
        g = f.name[:-len("_r16.json")]
        out = {}
        for k in RANKS:
            m = m_of(ad / "json" / f"{g}_r{k}.json"); fl = floor_of(ad / "floor" / f"{g}_r{k}.json")
            if m is None or fl is None:
                break
            rp = m["raw"]["mean_pb"]
            out[k] = dict(C=m["C"]["mean_pb"] / rp, CSP=m[f"CSP{k}"]["mean_pb"] / rp,
                          CSPmag=m[f"CSPmag{k}"]["mean_pb"] / rp, CSPang=m[f"CSPang{k}"]["mean_pb"] / rp,
                          floor=fl[f"P{k}_truth"]["mean_pb"] / rp,
                          floor_ang=fl[f"Pang{k}_truth"]["mean_pb"] / rp,
                          floor_mag=fl[f"Pmag{k}_truth"]["mean_pb"] / rp)
        else:
            res[g] = out
    n = len(res)
    summ["angle_only"] = res
    summ["angle_only_counts"] = {str(k): dict(
        n=n,
        cspang_worse_than_csp=sum(res[g][k]["CSPang"] > res[g][k]["CSP"] for g in res),
        cspang_worse_than_raw=sum(res[g][k]["CSPang"] > 1 for g in res),
        cspang_better_than_C=sum(res[g][k]["CSPang"] < res[g][k]["C"] for g in res),
        ref_angonly_floor_above_full=sum(res[g][k]["floor_ang"] > res[g][k]["floor"] for g in res))
        for k in RANKS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    summ = {}
    crossgrid(a.root, a.out, summ)
    converged(a.root, a.out, summ)
    gbnetwork(a.root, a.out, summ)
    angle_only(a.root, summ)
    (a.out / "summary.json").write_text(json.dumps(summ, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in summ.items() if k in (
        "crossgrid_n", "crossgrid_missing", "converged_missing", "gb_missing")}, indent=1))
    print(json.dumps(summ["angle_only_counts"], indent=1))


if __name__ == "__main__":
    main()
