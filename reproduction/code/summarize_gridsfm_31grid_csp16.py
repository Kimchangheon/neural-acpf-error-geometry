#!/usr/bin/env python3
"""Assemble frozen 31-grid GridSFM raw/CSP16 results into CSV and figures.

This postprocessor never performs inference.  It validates the per-grid JSON
written by ``score_gridsfm_31grid_csp16_helma.sh`` and preserves missing or
failed grids explicitly in the audit CSV rather than imputing values.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_manifest(path: Path):
    rows = []
    text = path.read_text()
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith('"'):
            continue
        value = line.split('"', 2)[1]
        fields = value.split('|')
        if len(fields) == 4:
            rows.append({'grid_name': fields[0], 'number_of_buses': int(fields[1]),
                         'parquet': fields[2]})
    if len(rows) != 31:
        raise RuntimeError(f'Expected 31 manifest rows, found {len(rows)}')
    return rows


def row_from_json(base, json_path: Path, provenance_path: Path):
    row = dict(base)
    row.update(status='missing', reason='No result JSON', checkpoint='', checkpoint_sha256='',
               test_scenarios='', raw_mean_pb='', csp16_mean_pb='', raw_max_pb='',
               csp16_max_pb='', raw_vmag_rmse='', csp16_vmag_rmse='',
               log10_raw_mean_pb='', log10_csp16_mean_pb='')
    if not json_path.exists():
        return row
    try:
        payload = json.loads(json_path.read_text())
        protocol, result = payload['protocol'], payload['result']
        m = result['metrics']
        raw, csp = m['raw'], m['CSP16']
        if protocol.get('rank') != 16 or protocol.get('split_seed') != 42:
            raise ValueError('rank/split protocol mismatch')
        values = [raw['mean_pb'], csp['mean_pb'], raw['max_pb'], csp['max_pb'],
                  raw['vmag_rmse'], csp['vmag_rmse']]
        if not all(math.isfinite(float(x)) and float(x) > 0 for x in values):
            raise ValueError('non-finite or non-positive metric')
        p = {}
        if provenance_path.exists():
            for line in provenance_path.read_text().splitlines():
                if '=' in line:
                    k, v = line.split('=', 1); p[k] = v
        row.update(status='scored', reason='', checkpoint=p.get('checkpoint', ''),
                   checkpoint_sha256=p.get('checkpoint_sha256', ''),
                   test_scenarios=result['n_scenarios'], raw_mean_pb=raw['mean_pb'],
                   csp16_mean_pb=csp['mean_pb'], raw_max_pb=raw['max_pb'],
                   csp16_max_pb=csp['max_pb'], raw_vmag_rmse=raw['vmag_rmse'],
                   csp16_vmag_rmse=csp['vmag_rmse'],
                   log10_raw_mean_pb=math.log10(raw['mean_pb']),
                   log10_csp16_mean_pb=math.log10(csp['mean_pb']))
    except Exception as exc:  # audit failures must remain visible
        row.update(status='invalid', reason=str(exc))
    return row


def plot(rows, output: Path, paired_lines: bool, label: str = 'GridSFM'):
    good = [r for r in rows if r['status'] == 'scored']
    x = np.asarray([float(r['number_of_buses']) for r in good])
    raw = np.asarray([float(r['raw_mean_pb']) for r in good])
    csp = np.asarray([float(r['csp16_mean_pb']) for r in good])
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    if paired_lines:
        for xi, yi, ci in zip(x, raw, csp):
            ax.plot([xi, xi], [yi, ci], color='0.70', linewidth=.7, zorder=1)
    ax.scatter(x, raw, color='#D55E00', marker='o', s=28, label=f'{label} Raw', zorder=3)
    ax.scatter(x, csp, color='#0072B2', marker='s', s=28, label=label + r' CSP$_{16}$', zorder=4)
    ax.set_yscale('log')
    ax.set_xlabel('Number of buses')
    ax.set_ylabel('Mean PB')
    ax.grid(True, which='both', axis='y', color='0.88', linewidth=.6)
    ax.legend(frameon=False, loc='best')
    ax.set_title(f'{label} power-balance consistency across {len(good)} grids')
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(output.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_discrete(rows, output: Path, label: str = 'GridSFM', text_scale: float = 1.0):
    """Categorical 31-grid view: one ordered position per bus-count entry."""
    good = [r for r in rows if r['status'] == 'scored']
    x = np.arange(len(good))
    raw = np.asarray([float(r['raw_mean_pb']) for r in good])
    csp = np.asarray([float(r['csp16_mean_pb']) for r in good])
    labels = [str(r['number_of_buses']) for r in good]
    # Preserve the paired figure's visual language; only the x-coordinate is
    # categorical here, so that unequal bus-count gaps do not compress the
    # smaller grids.
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    # Pair only the two states of the same grid.  Do not connect successive
    # grids: the categorical x-axis encodes an ordered comparison, not a
    # continuous operating curve.
    for xi, yi, ci in zip(x, raw, csp):
        ax.plot([xi, xi], [yi, ci], color='0.70', linewidth=.8, zorder=1)
    ms = 28 * text_scale ** 2
    ax.scatter(x, raw, color='#ff7f0e', marker='s', s=ms,
               label=f'{label} Raw', zorder=3)
    ax.scatter(x, csp, color='#1f77b4', marker='o', s=ms,
               label=label + r' + CSP$_{16}$', zorder=4)
    ax.set_yscale('log')
    fs = 10 * text_scale
    ax.set_xlabel(f'Number of buses ({len(good)} grids, sorted)', fontsize=fs)
    ax.set_ylabel('Mean PB', fontsize=fs)
    ax.set_xticks(x, labels, rotation=55, ha='right', fontsize=7 * text_scale)
    ax.tick_params(axis='y', labelsize=fs)
    ax.grid(True, which='both', axis='y', color='0.88', linewidth=.6)
    # Top-right, as in the published figure; 'best' drops it onto the
    # 29-bus point, which sits highest on the left.
    ax.legend(frameon=False, loc='upper right', fontsize=fs)
    ax.set_title(f'{label} power-balance consistency across {len(good)} grids', fontsize=fs * 1.1)
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(output.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--results-dir', type=Path, required=True)
    ap.add_argument('--model-label', default='GridSFM',
                    help='Name used in legends and titles.')
    ap.add_argument('--prefix', default='gridsfm',
                    help='Figure filename prefix.')
    args = ap.parse_args()
    out = args.results_dir
    rows = [row_from_json(base, out / 'json' / f"{base['grid_name']}.json",
                          out / 'checkpoint_manifest' / f"{base['grid_name']}.txt")
            for base in load_manifest(args.manifest)]
    rows.sort(key=lambda r: (r['number_of_buses'], r['grid_name']))
    fields = ['grid_name', 'number_of_buses', 'parquet', 'status', 'reason', 'checkpoint', 'checkpoint_sha256',
              'test_scenarios', 'raw_mean_pb', 'csp16_mean_pb', 'raw_max_pb', 'csp16_max_pb',
              'raw_vmag_rmse', 'csp16_vmag_rmse', 'log10_raw_mean_pb', 'log10_csp16_mean_pb']
    for name, subset in [('grid_metrics.csv', rows), ('audit.csv', rows)]:
        with (out / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(subset)
    good = [r for r in rows if r['status'] == 'scored']
    if good:
        raw = np.asarray([float(r['raw_mean_pb']) for r in good])
        csp = np.asarray([float(r['csp16_mean_pb']) for r in good])
        summary = {'expected_grids': len(rows), 'scored_grids': len(good),
                   'unscored_grids': [r['grid_name'] for r in rows if r['status'] != 'scored'],
                   'csp_lowers_mean_pb_grids': int(np.sum(csp < raw)),
                   'median_raw_mean_pb': float(np.median(raw)),
                   'median_csp16_mean_pb': float(np.median(csp)),
                   'median_relative_reduction': float(np.median(1.0 - csp / raw)),
                   'geometric_mean_csp16_over_raw': float(np.exp(np.mean(np.log(csp / raw))))}
    else:
        summary = {'expected_grids': len(rows), 'scored_grids': 0,
                   'unscored_grids': [r['grid_name'] for r in rows]}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if good:
        pre, lab = args.prefix, args.model_label
        plot(rows, out / f'{pre}_raw_vs_csp16_paired', paired_lines=True, label=lab)
        plot(rows, out / f'{pre}_raw_vs_csp16_unpaired', paired_lines=False, label=lab)
        plot_discrete(rows, out / f'{pre}_raw_vs_csp16_discrete', label=lab)
        plot_discrete(rows, out / f'{pre}_raw_vs_csp16_discrete_larger_text',
                      label=lab, text_scale=1.6)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
