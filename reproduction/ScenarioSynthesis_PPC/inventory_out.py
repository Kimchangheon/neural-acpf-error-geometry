#!/usr/bin/env python3
"""Group everything in the out/ directory into human-readable categories.

The directory holds several years of campaigns under one flat namespace, so
"what is in here" is not answerable by ls. This classifies every file and
directory by what it actually is, totals the size per group, and flags which
groups are live, superseded, or the only copy of something.
"""
import os
import re
from collections import defaultdict

OUT = "/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out"


def size(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    return sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(p) for f in fs)


def classify(name, isdir):
    n = name
    if isdir:
        m = re.match(r"^(.*)_chunk_\d+$", n)
        if m:
            fam = m.group(1)
            if "cnr" in fam.lower():
                return "CHUNK: custom-NR (cNR) leftovers"
            if fam.endswith("_A") or "_A_" in fam:
                return "CHUNK: 'A' preset campaign"
            if "snapshot" in fam.lower():
                return "CHUNK: snapshot-envelope experiments"
            return "CHUNK: other"
        if "smoke" in n.lower() or "debug" in n.lower():
            return "scratch: smoke / debug runs"
        return "dir: other"

    # ---- files -----------------------------------------------------------
    if not n.endswith(".parquet"):
        return "non-parquet"
    if "u0clean_sgenpert_pvqvstart" in n:
        return "*** v2 corpus (LIVE) ***"
    if "_OPF" in n or "ACOPF" in n:
        return "OPF datasets"
    if n.endswith("_rg20.parquet"):
        return "rg20 re-encodings of superseded v1 files"
    if "_u0jit5deg-0.02_" in n or ("_u0clean_siNR_" in n and
                                   ("case118" in n or "case300" in n)):
        return "u_start A/B pair (kept deliberately)"
    if "_cNR_" in n:
        return "custom-NR (cNR) merged"
    if "snapshot_envelope" in n:
        return "snapshot-envelope experiments"
    if re.search(r"_ppcY_A_", n):
        return "'A' preset campaign (merged)"
    if "dc_init" in n or "flat_start" in n:
        return "ENTSO-E start-mode variants"
    if "_ppNR_" in n:
        return "leftover v1 ppNR"
    return "other parquet"


def main():
    groups = defaultdict(lambda: [0, 0])      # label -> [bytes, count]
    examples = defaultdict(list)
    for e in os.listdir(OUT):
        p = os.path.join(OUT, e)
        isdir = os.path.isdir(p)
        lab = classify(e, isdir)
        s = size(p)
        groups[lab][0] += s
        groups[lab][1] += 1
        if len(examples[lab]) < 3:
            examples[lab].append((s, e))

    total = sum(v[0] for v in groups.values())
    print(f"{'group':<46}{'items':>7}{'GB':>10}{'share':>8}")
    print("-" * 71)
    for lab, (b, c) in sorted(groups.items(), key=lambda kv: -kv[1][0]):
        print(f"{lab:<46}{c:>7}{b/1e9:>10.1f}{b/total:>8.1%}")
    print("-" * 71)
    print(f"{'TOTAL':<46}{sum(v[1] for v in groups.values()):>7}{total/1e9:>10.1f}")

    print("\n\nlargest example per group:")
    for lab, (b, c) in sorted(groups.items(), key=lambda kv: -kv[1][0]):
        s, e = max(examples[lab])
        print(f"  [{lab}]\n    {e[:96]}  ({s/1e9:.1f} GB)")


if __name__ == "__main__":
    main()
