# From each result in the paper back to the code that made it

Every row below was traced by reading the batch script that ran the job and, for
the main table, by recomputing the published numbers from the stored JSON. Where
a number could not be reproduced, that is said explicitly.

`$CODE` = `reproduction/code/`, `$JOBS` = `reproduction/experiments/historical_local/`,
`$RES` = `reproduction/results/`.

---

## Table `tab:intervention` — GBnetwork, k = 16, four models

Raw / C / P₁₆ / CSP₁₆ for each model, three seeds, plus per-scenario inference
time.

| Model | Job script | Script | Results |
|---|---|---|---|
| PIGNN-GC, GridSFM | `$JOBS/gbnetwork_known_state_csp_k16_helma_20260909.sh` | `output_intervention_metrics.py` | `$RES/helma/gbnetwork_known_state_csp_k16_20260909/` |
| GridFM-GraphKit | `$JOBS/score_graphkit_e120_fullstate_k16_helma_20260911.sh` | same | `$RES/helma/graphkit_e120_fullstate_csp_k16_20260911/` |
| LUMINA | `$JOBS/score_lumina_gbv06z_h100_3seed_fullstate_k16_20260914.sh` | same | `$RES/helma/lumina_gbv06z_h100_3seed_fullstate_csp_k16_20260914/` |
| Inference time | `$JOBS/benchmark_gbnetwork_inference_h100_20260912.sh` | `benchmark_inference_walltime.py` | `$RES/helma/gbnetwork_inference_walltime_h100_20260912/` |

Checkpoints for each seed are listed in `DATA_AND_CHECKPOINTS.md`.

### Verification

Recomputing mean ± sd of `vmag_rmse` over the three seed JSONs reproduces the
published values exactly:

| Model | Variant read | Raw | C | P₁₆ | CSP₁₆ |
|---|---|---|---|---|---|
| PIGNN-GC | `restore_known` | .02849 | .01393 | .01624 | .01221 |
| GridSFM | `restore_known` | .02868 | .01086 | .01814 | .00782 |
| GraphKit | `full` | .00892 | .00753 | .00617 | .00606 |
| LUMINA | `full` | .02653 | .01780 | .01898 | .01570 |

All agree with the manuscript to the printed precision.

### One inconsistency in the scripts, which turns out not to matter

The manuscript states that "prescribed PV/slack magnitudes and the slack angle
are restored before" scoring. The scripts do not apply one policy uniformly:

- PIGNN-GC and GridSFM were scored with `--projection-variants restore_known
  unknown_only`, and the published rows come from **`restore_known`**.
- GraphKit and LUMINA were scored with `--projection-variants full`, and the
  published rows come from **`full`** — nominally the path that does *not*
  restore.

Each set of numbers matches its own JSON exactly, so the two model pairs really
were scored under different flags. A dedicated comparison run settles whether
that changes anything:

`$JOBS/csp_fullstate_vs_restore_gbnetwork_helma_20260910.sh` →
`$RES/helma/csp_fullstate_vs_restore_gbnetwork_20260910/`

| Model (seed 42) | Variant | Raw | C | P16 | CSP16 |
|---|---|---|---|---|---|
| PIGNN-GC | `full` | .029351 | .014213 | .017191 | .012872 |
| PIGNN-GC | `restore_known` | .029351 | .014213 | .017191 | .012872 |
| GridSFM | `full` | .031862 | .011789 | .020269 | .008572 |
| GridSFM | `restore_known` | .031862 | .011789 | .020269 | .008572 |

**`full` and `restore_known` are identical to every printed digit**, including
for GridSFM, the one model that does not pin the known entries inside its own
forward pass. The projection does not move the prescribed quantities, so
restoring them afterwards has nothing to do.

The third policy is the one that differs. `unknown_only` restricts the
projection to the unknown block, and for GridSFM it gives P16 .01939 and CSP16
.00878 against .01814 and .00782 — it was not used for any published row.

So the manuscript sentence is correct in effect; the flags are inconsistent in
form only. Worth tidying before release so a reader reproducing from the scripts
is not misled, but no number changes.

---

## Figure `fig:error_geometry` — fixed-norm directional intervention

| Job script | Script | Results |
|---|---|---|
| `$JOBS/controlled_geometry_stage1_{helma,alex_20260906}.sh`, `…_stage2_alex_20260906.sh`, `…_summary_k16_alex_20260907.sh` | `controlled_error_geometry.py` | `$RES/helma/controlled_geometry_20260905/`, `$RES/local/controlled_geometry_20260905/` |
| `$JOBS/controlled_geometry_graphkit_e120_alex_20260911.sh` | same | `$RES/helma/fixed_norm_directional_graphkit_e120_20260912/` |
| (LUMINA arm) | same | `$RES/alex/fixed_norm_lumina_gbv06z_20260913/` |

The frozen protocol for this campaign is
`docs/historical/CONTROLLED_ERROR_GEOMETRY_PROTOCOL.md`, and the code snapshot it
ran against is `reproduction/experiments/snapshots/helma/controlled_geometry_20260905/`.

---

## Table `tab:basis_control` — basis controls

Random, graph-spectral and smoothing bases against the training-solution basis.

| Job script | Script | Results |
|---|---|---|
| `$JOBS/pignn_g3_basis_controls_gbnetwork_{helma,alex2_a40}_20260909.sh` | `compare_solution_basis_controls.py` | `$RES/helma/pignn_g3_basis_controls_20260909_alex2/` |
| `$JOBS/basis_reconstruction_matched_controls_helma_20260914.sh` | `basis_reconstruction_matched_controls.py` | `$RES/helma/basis_reconstruction_matched_20260914/` |
| `$JOBS/pignn_g3_shrinkage_stage1_{helma,alex2_a40}_20260909.sh` | `evaluate_scalar_shrinkage.py` | `$RES/helma/pignn_g3_shrinkage_20260909_alex2/` |

---

## Section "Off-subspace error and directional sensitivity"

| Job script | Script | Results |
|---|---|---|
| `$JOBS/off_subspace_conditional_association_4model_seed42_helma_20260914.sh` | `off_subspace_conditional_association.py` | `$RES/helma/off_subspace_conditional_association_20260914/` |

Supporting Jacobian evidence:

| Job script | Script | Results |
|---|---|---|
| `$JOBS/jacobian_subspace_alignment_gbnetwork_{helma,alex2_a40}_20260909.sh` | `jacobian_subspace_alignment.py` | — |
| `$JOBS/jacobian_reproducibility_gbnetwork_helma_20260910.sh` | `jacobian_subspace_reproducibility.py` | `$RES/helma/jacobian_reproducibility_gbnetwork_20260910/` |
| `$JOBS/jacobian_reproducibility_graphkit_e120_helma_20260912.sh` | same | `$RES/helma/jacobian_graphkit_e120_20260912/` |
| `$JOBS/jacobian_reproducibility_lumina_gbv06z_alex_a40_20260913.sh` | same | `$RES/alex/jacobian_lumina_gbv06z_20260913/` |

---

## Section "Post-training CSP correction"

| Job script | Script | Results |
|---|---|---|
| `$JOBS/csp_posthoc_diagnostics_gbnetwork_helma_20260910.sh` | `csp_posthoc_diagnostics.py` | `$RES/helma/csp_posthoc_diagnostics_gbnetwork_20260910/` |
| `$JOBS/csp_posthoc_diagnostics_graphkit_e120_helma_20260912.sh` | same | `$RES/helma/csp_posthoc_diagnostics_graphkit_e120_20260912/` |
| `$JOBS/csp_posthoc_diagnostics_lumina_gbv06z_alex_a40_20260913.sh` | same | `$RES/alex/csp_posthoc_lumina_gbv06z_20260913/` |
| `$JOBS/csp_projection_block_ablation_*.sh` | `csp_projection_block_ablation.py` | `$RES/helma/csp_projection_block_ablation_*`, `$RES/alex/csp_blocks_lumina_gbv06z_20260913/` |

---

## Table `tab:nr1` — one Newton update after C or CSP₁₆

Run in three stages so the damping factor is chosen on validation only and never
on test.

1. **Validation** — `nr1_*_validation_*.sh` → `nr1_multiprocess_baseline.py`
2. **Select η** — `nr1_*_select_eta_*.sh` → `select_nr1_eta.py` /
   `select_nr1_eta_graphkit.py`
3. **Test** — `nr1_*_test_*.sh` → `nr1_multiprocess_baseline.py`, then
   `nr1_*_complete_*.sh` → `nr1_metric_complete_rescore.py`

Results: `$RES/helma/nr1_multiproc_gbnetwork_20260910/`,
`$RES/helma/nr1_graphkit_e120_gbnetwork{,_a40}_20260912/`,
`$RES/helma/nr1_lumina_gbv06z_3seed_20260914/`,
`$RES/alex/nr1_lumina_gbv06z_20260913/`.

---

## Figure `fig:crossgrid_csp` — 31 grids, GridSFM before and after CSP₁₆

| Job script | Script | Results |
|---|---|---|
| `$JOBS/score_gridsfm_31grid_csp16_{helma,alex}.sh` | `output_intervention_metrics.py` | `$RES/helma/gridsfm_31grid_csp16_20260907/` |
| — | `summarize_gridsfm_31grid_csp16.py` | assembles the CSV and the figure |

---

## Table `tab:shift` — topology and operating-distribution shift

Two shift arms, and for each a transfer (T) and a refit (R) variant.

| Arm | Corpus | Job script | Script | Results |
|---|---|---|---|---|
| N-1 topology | `mix_n1_20260905/arm{A,B}*.parquet` | (campaign snapshot) | `train_valid_test.py` + `output_intervention_metrics.py` | `$RES/helma/mix_n1_20260905/` |
| Correlated tilt | `control.parquet`, `gbcorr.parquet` | `$JOBS/gbcorr_csp_posthoc_alex2_20260910.sh` | `gbcorr_csp_posthoc.py` | `$RES/alex/gbcorr_csp_posthoc_20260910/`, `$RES/local/gbcorr_csp_posthoc_20260910/` |

The N-1 arm ran against the frozen code in
`reproduction/experiments/snapshots/helma/mix_n1_20260905/`, which contains its own copies of
`train_valid_test.py` and `GNSMsg_SelfAttention_armijo.py`.

---

## Model seed replicates

The three-seed sets for GridSFM, GraphKit and LUMINA were trained by
`$JOBS/gbnetwork_model_seed_replicates_a40_20260907.sh` (drivers
`train_valid_test_gridsfm.py`, `train_valid_test_gridfm.py`,
`train_valid_test_lumina.py`) and scored by
`$JOBS/gbnetwork_model_seed_metrics_a40_20260907.sh`.

Note what "seed" means here: `--split_seed 42` is held fixed, so all replicates
see the identical train/validation/test partition. Only `--seed_value` changes,
which moves model initialisation and the optimisation stream. The ± in the
tables is therefore sensitivity to initialisation and optimisation randomness,
not to the data split.

One caveat carried over from the campaign: for PIGNN-GC and GridSFM the third
"seed" is an older frozen baseline from a different campaign rather than a
matched replicate (see `DATA_AND_CHECKPOINTS.md`). GraphKit's three are matched.
