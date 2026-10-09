#!/usr/bin/env python3
"""Supplementary Fig. S2: CSP rank curves for gridfm-graphkit (120 epochs).

One panel per grid with N >= 145 (for smaller grids k >= N already makes the
projection the identity, so CSP_k reduces to calibration).  Each panel shows,
as ratios to the raw prediction and on a log scale:
  CSP_k mean PB, CSP_k |V| RMSE, the projected-truth floor PB, and (where
  scored) the angle-only variant CSPang_k.

Inputs are the frozen per-grid JSON written by output_intervention_metrics.py
(``<grid>_r<k>.json``) and csp16_truth_floor.py (``floor/<grid>_r<k>.json``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

RANKS = (16, 32, 64, 128)
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"
SHORT = {"ENTSO_E_RealGridTest": "ENTSO-E", "case_illinois200": "illinois200"}


def manifest(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith('"'):
            f = line.split('"')[1].split("|")
            rows.append((f[0], int(f[1])))
    return rows


def metrics(path: Path):
    return json.loads(path.read_text())["result"]["metrics"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--ranks-dir", type=Path, required=True,
                    help="dir with json/<grid>_r<k>.json and floor/<grid>_r<k>.json")
    ap.add_argument("--angonly-dir", type=Path, default=None,
                    help="optional dir whose json carries CSPang_k")
    ap.add_argument("--min-buses", type=int, default=145)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    grids = [(g, n) for g, n in manifest(a.manifest) if n >= a.min_buses
             and all((a.ranks_dir / "json" / f"{g}_r{k}.json").exists() for k in RANKS)]
    grids.sort(key=lambda x: x[1])
    ncol = 5
    nrow = (len(grids) + ncol - 1) // ncol
    plt.rcParams.update({"font.size": 6.5, "axes.edgecolor": "#c9c8c2",
                         "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 1.55 * nrow + 0.45),
                             sharex=True, sharey=True)
    ticks = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 50]
    has_ang = False
    for ax, (g, n) in zip(axes.flat, grids):
        m = {k: metrics(a.ranks_dir / "json" / f"{g}_r{k}.json") for k in RANKS}
        fl = {k: json.loads((a.ranks_dir / "floor" / f"{g}_r{k}.json").read_text())["metrics"]
              for k in RANKS}
        rp, rv = m[16]["raw"]["mean_pb"], m[16]["raw"]["vmag_rmse"]
        ax.axhline(1, color=INK2, lw=0.6, zorder=1)
        ax.plot(RANKS, [fl[k][f"P{k}_truth"]["mean_pb"] / rp for k in RANKS],
                color=GRAY, lw=0.9, ls=(0, (3, 2)), zorder=2)
        ax.plot(RANKS, [m[k][f"CSP{k}"]["vmag_rmse"] / rv for k in RANKS], color=ORANGE,
                lw=1.3, marker="s", ms=2.8, zorder=3)
        ax.plot(RANKS, [m[k][f"CSP{k}"]["mean_pb"] / rp for k in RANKS], color=BLUE,
                lw=1.3, marker="o", ms=3, zorder=4)
        if a.angonly_dir is not None:
            p = {k: a.angonly_dir / "json" / f"{g}_r{k}.json" for k in RANKS}
            if all(x.exists() for x in p.values()):
                has_ang = True
                ma = {k: metrics(p[k]) for k in RANKS}
                ax.plot(RANKS, [ma[k][f"CSPang{k}"]["mean_pb"] / rp for k in RANKS],
                        color=AQUA, lw=1.3, marker="D", ms=2.6, zorder=5)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator(RANKS))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_ylim(0.015, 60)
        ax.grid(True, axis="y", color="#e6e5e0", lw=0.4, zorder=0)
        ax.set_title(f"{SHORT.get(g, g)} ($N$={n:,})", fontsize=6.5, loc="left", pad=2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes.flat[len(grids):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("rank $k$")
    for ax in axes[:, 0]:
        ax.set_ylabel("ratio to raw")
    h = [Line2D([], [], color=BLUE, lw=1.3, marker="o", ms=3, label=r"CSP$_k$ Mean PB"),
         Line2D([], [], color=ORANGE, lw=1.3, marker="s", ms=2.8, label=r"CSP$_k$ $V$ RMSE"),
         Line2D([], [], color=GRAY, lw=0.9, ls=(0, (3, 2)), label=r"floor: $P_k x^\star$ Mean PB")]
    if has_ang:
        h.append(Line2D([], [], color=AQUA, lw=1.3, marker="D", ms=2.6,
                        label=r"angle-only CSP$_{\theta,k}$ Mean PB"))
    fig.legend(handles=h, loc="upper center", ncol=len(h), frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=0.6, w_pad=0.4)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print("wrote", a.out, len(grids), "panels")


if __name__ == "__main__":
    main()
