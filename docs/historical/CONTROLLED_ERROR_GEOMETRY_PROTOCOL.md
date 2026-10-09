# Frozen protocol: controlled error-geometry experiment

Frozen before validation execution on 2026-09-05.

## Common data and scoring

- Dataset: GBnetwork ppNR-v2 parquet, fixed topology, 37,022 converged rows.
- Split: `random_split`, seed 42, `int(0.3333*n)` train, validation, then test.
- Models: 40-epoch-campaign best-validation checkpoints for PIGNN-GC/G3,
  GridSFM, GridFM-GraphKit, and LUMINA.
- Each model receives one forward pass over training data to fit its per-bus
  magnitude offset and circular angle offset. Each evaluated split receives
  exactly one neural forward pass; calibrated predictions and references are
  cached before rank-wise scoring.
- The magnitude SVD basis is fitted once from training reference solutions only.
- Physical score: complex128 GENCO-style PB, averaged over all bus--scenario
  pairs; PQ uses `sqrt(dP^2+dQ^2)`, PV uses `abs(dP)`, slack contributes zero.

## Stage 1: validation only

Candidate ranks are `4, 8, 16, 32, 64`. For each model, compute calibrated
validation magnitude RMSE/Mean PB and calibrated-plus-projection (CSP) RMSE/
Mean PB at each candidate rank. Let `q_m(k)=MeanPB_m^CSP(k)/MeanPB_m^C`.

A rank is admissible only when `RMSE_m^CSP(k) <= RMSE_m^C` for every model.
Among admissible ranks, select the one minimizing `mean_m q_m(k)`. If no rank
is admissible, Stage 2 is not run and no post-hoc fallback is introduced.

Stage 1 must not load, score, summarize, or save predictions from the test split.

## Stage 2: test after frozen rank selection

Only after the selected rank is written to the Stage-1 result file, evaluate the
test split. Let `e=v^C-v*`, `e_parallel=P_k e`, and `e_perp=(I-P_k)e`. For
alpha in `{0,.25,.5,.75,1}`, construct the fixed-norm off- and in-subspace
counterfactuals specified by the manuscript plan, retain calibrated phase
angles, and recompute PB. No clipping is allowed. Record zero denominators,
fixed-norm error, alpha=1 reproduction error, per-scenario PB, paired bootstrap
intervals, and fractions of scenarios with lower PB.

The counterfactual uses `v*`; it is a diagnostic mechanism experiment, not a
deployable correction.

## Frozen Stage-1 selection (2026-09-06)

Stage 1 completed before any test-split evaluation. Every candidate met the
per-model validation RMSE guardrail. The mean of the four pre-specified
normalized validation PB values was: `k=4: 0.272423`, `k=8: 0.272030`,
`k=16: 0.270474`, `k=32: 0.271460`, and `k=64: 0.271975`. Therefore the
frozen common rank for Stage 2 is **`k*=16`**. This decision uses only the
four validation JSON files in `results/controlled_geometry_20260905/json/`.
