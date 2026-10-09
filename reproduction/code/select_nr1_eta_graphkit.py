#!/usr/bin/env python3
"""Freeze the NR1 damping value from GraphKit validation results only."""
import argparse
import glob
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--glob", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--expected-count", type=int, default=3,
                help="Number of validation JSON files required by the frozen protocol.")
a = ap.parse_args()

files = sorted(glob.glob(a.glob))
assert len(files) == a.expected_count, (
    f"expected {a.expected_count} validation files, got {len(files)}")
rows = [json.load(open(path)) for path in files]
etas = rows[0]["etas"]
means = {}
for eta in etas:
    values = [row["score"][state][str(eta)]["mean_pb"]
              for row in rows for state in ("C", "CSP16")]
    means[str(eta)] = sum(values) / len(values)
selected = min(etas, key=lambda eta: means[str(eta)])
Path(a.out).write_text(json.dumps({
    "rule": "minimize unweighted mean validation Mean PB over C+NR1 and CSP16+NR1 across the frozen validation runs",
    "candidates": means,
    "selected_eta": selected,
    "inputs": files,
}, indent=2))
