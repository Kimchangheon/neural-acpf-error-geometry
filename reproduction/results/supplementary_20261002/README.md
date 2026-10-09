# Supplementary-material result snapshot (2026-10-02)

Frozen JSON copied from the Fritz cluster (`PIGNN-Attn-LS-PPC/results/`).
Every table of `ICASSP2027_manuscript/neural_pf_csp_supplementary.tex` is
rebuilt from this directory by

    python reproduction/code/build_supplementary.py \
        --root reproduction/results/supplementary_20261002 \
        --out ICASSP2027_manuscript/supplementary/generated

and the figures by `plot_supplementary_rank_curves.py` and
`summarize_gridsfm_31grid_csp16.py --model-label gridfm-graphkit --prefix graphkit`.

| Directory | Content |
|---|---|
| `graphkit_31grid_csp16_20260921/` | 120-epoch gridfm-graphkit, raw/C/P16/CSP16, 30 of 31 grids (+ checkpoint manifests) |
| `graphkit_31grid_ranks_20260922/` | same checkpoints, k = 16/32/64/128, with magnitude-only states and NR floors |
| `graphkit_angonly_20261002/` | 12 grids with N >= 300, adds angle-only CSP and floors |
| `graphkit_ext3e5_csp16_20260929/` | converged (+200 ep, LR 3e-5) six grids, k = 16 (superseded by supp_rescore) |
| `gb_rank_sweep_20260922/` | GBnetwork ext checkpoints, k = 16..128, magnitude-only (superseded by supp_rescore) |
| `supp_rescore_20261002/` | converged six grids and four GBnetwork checkpoints, k = 16..128, all channels + floors; `*.sha256` = checkpoint hashes; `sbatch/supp_rescore.sh` = job script (helma job 925415) |

## Scorer protocol check
The paper's Table 1 gridfm-graphkit rows were scored with full-state
projection and **no** `restore_known`
(`graphkit_e120_fullstate_csp_k16_alex_20260911`). Rescoring the same seed-42
checkpoint `k08_s42_e120_best.pt` (sha256 prefix `42bcd9d9f4211c36bb22`) with
the scorer used here reproduces all mean metrics of raw/C/P16/CSP16 to
~1e-7 relative; Max PB of raw/C differs by 2e-4 relative (GPU nondeterminism).

## Pending
case9241pegase: retraining (39 epochs at LR 1e-4, then 40+41 at 3e-5, helma
jobs 921710 -> 921711 -> CSP16 921712). Add its rank and angle-only sweeps,
then rerun the builder.
