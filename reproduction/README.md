# The research archive

This directory is what actually ran on the clusters: four training drivers, the
diagnostics that produced each figure, the batch scripts, and the result
artifacts. It is kept close to verbatim so the published numbers stay traceable.

**If you want to use CSP, you want [`../csp/`](../csp/), not this.** The method
is re-implemented there as a small, dependency-light package, and
`tests/test_equivalence_with_paper_code.py` checks that the rewrite computes
what the code here computed. This archive is for reproducing the paper and for
reading how the experiments were actually run.

Nothing here is importable as a library. Modules assume sibling files on
`sys.path`, cluster paths, and a parquet pipeline that is not shipped.

```
reproduction/
├── code/                  every module the paper's numbers depend on
├── ScenarioSynthesis_PPC/ the corpus generator and its solver
├── experiments/
│   ├── historical_local/    job scripts as kept in the working repo
│   ├── historical_helma/    job scripts as kept on the cluster
│   └── snapshots/           frozen per-campaign code copies
├── results/                 small artifacts (JSON/CSV/figures), by host
├── reports/                 longer internal reports on datasets and models
└── provenance/              what was copied here, from where, with checksums
```

**On `snapshots/`:** each campaign ran against a frozen copy of the driver and
model files rather than the live repository, so later edits could not silently
change a published number. If a result and `code/` disagree, the snapshot is
what produced the result.

---

## `code/` — what each module is for

### The four surrogates

Each model has its own driver. They share the data pipeline and the metric code
and differ in how the model is constructed and called.

| Module | Role |
|---|---|
| `train_valid_test.py` | PIGNN-GC. Newton-like iteration with an Armijo line search; output heads are zero-initialised, so the first forward pass reproduces `V_start`. |
| `train_valid_test_gridfm.py` | GridFM-GraphKit; also hosts our `GridFMHeteroSurrogate` re-implementation behind `--gridfm_impl`. |
| `train_valid_test_gridsfm.py` | GridSFM, wrapping the released `GridTransformerBackbone`. |
| `train_valid_test_lumina.py` | LUMINA-2M, wrapping the released HGT. |

Supporting model code:

| Module | Role |
|---|---|
| `GNSMsg_SelfAttention_armijo.py` | PIGNN backbone: message passing with self-attention and the Armijo-damped update. |
| `GNSMsg_SelfAttention_armijo_khop.py`, `GNSMsg_armijo.py` | k-hop and plain variants. |
| `gridfm_graphkit_adapter.py` | Builds the released GraphKit heterogeneous batch and runs its forward pass. |
| `gridfm_projected_dpf_hybrid.py` | Projected-DPF hybrid inside the released GraphKit processor. |
| `genco_ls.py` | Heterogeneous GNS with a state-space incremental decoder and line search. |
| `known_operator_pf.py`, `helm_known_operator_pf.py` | Differentiable known-operator corrections, Newton and HELM style. |

### CSP, as the paper ran it

| Module | Role |
|---|---|
| `manifold_projection.py` | The basis and the projection. One implementation, shared by the training path and the scorer, so the two cannot drift apart. |
| `output_intervention_metrics.py` | The main table: one forward pass, then Raw / C / P_k / CSP_k scored from it, under any of the three projection policies. |

These two are the originals that `../csp/` was extracted from.

### AC-PF adaptation of the OPF surrogates

GridSFM and LUMINA are released as AC-OPF models. These supply what power flow
specifies and the OPF formulation does not.

| Module | Role |
|---|---|
| `pf_known_mask.py` | Holds the specified quantities: \|V\| at PV and slack, angle at slack. |
| `pf_anchors.py` | Output-side anchors: angle on the DC start, magnitude on a prior. |
| `vmag_prior.py` | A chord (modified-Newton) step with the Jacobian frozen at the flat point, LU-factorised once per grid, used as a magnitude prior. |
| `supervised_voltage_loss.py` | Train-only channel normalisation for the supervised voltage loss. |

### Data pipeline

| Module | Role |
|---|---|
| `Dataset_optimized_complex_columns.py` | Parquet-backed dataset; lazy row groups, shared grid tensors, complex128 decode. |
| `collate_blockdiag_optimized_complex_columns.py` | Block-diagonal collation of a scenario batch into one graph. |
| `preload_dataset.py` | Decodes a split once into RAM so epochs stop re-reading parquet. |
| `read_npy_columns_optimized.py` | Low-level column reader. |
| `opfdata_pipeline.py`, `opf_task.py` | The second (OPFData) pipeline and the shared OPF task pieces. |

### Training infrastructure

| Module | Role |
|---|---|
| `ddp_utils.py` | Multi-node DDP launched by `srun` rather than `torchrun`. |
| `grad_accum.py` | Gradient accumulation, so the effective batch stays fixed across grids of very different size. |
| `helper.py` | Small shared utilities. |

### Diagnostics and analysis

| Module | Produces |
|---|---|
| `controlled_error_geometry.py` | Fixed-norm directional intervention (`fig:error_geometry`). |
| `csp_posthoc_diagnostics.py` | Post-hoc CSP diagnostics against the frozen implementation. |
| `csp_projection_block_ablation.py` | Magnitude/angle-block ablation of the projection. |
| `jacobian_subspace_alignment.py` | Direct Jacobian alignment with the train-solution subspace. |
| `jacobian_subspace_reproducibility.py` | Frozen random-JVP reproduction plus audited model-error JVPs. |
| `off_subspace_conditional_association.py` | Off-subspace error and conditional association. |
| `compare_solution_basis_controls.py` | Random / graph-spectral / smoothing basis controls (`tab:basis_control`). |
| `basis_reconstruction_matched_controls.py` | Reconstruction-matched controls for the same table. |
| `evaluate_scalar_shrinkage.py` | Validation-selected scalar shrinkage baseline. |
| `nr1_multiprocess_baseline.py` | One-step Newton refinement from cached predictions (`tab:nr1`). |
| `nr1_metric_complete_rescore.py` | Metric-complete rescoring of frozen C/CSP16 + one NR step. |
| `select_nr1_eta.py`, `select_nr1_eta_graphkit.py` | Freeze the NR damping factor from validation only. |
| `gbcorr_csp_posthoc.py` | Correlated-tilt shift audit (`tab:shift`). |
| `summarize_gridsfm_31grid_csp16.py` | 31-grid GridSFM raw vs CSP16 (`fig:crossgrid_csp`). |
| `benchmark_inference_walltime.py` | The inference-time column of `tab:intervention`. |
| `summarize_gk_both_3seed.py` | Aggregates the three-seed GraphKit rescores. |
| `rescore_slope_r2.py` | Pooled slope and R² for any saved checkpoint of any of the four models. |
| `prediction_diagnostics.py` | Whether a surrogate tracks the solver or collapses to a constant. |
| `diagnose_residual_distributions.py` | Paired AC-residual distribution diagnostics. |

---

## `ScenarioSynthesis_PPC/`

The corpus generator. It builds each grid as a pandapower network — synthetic
cases directly, CGMES cases through pandapower's CIM converter — compiles the
internal PPC and its admittance matrix, perturbs loads and generation, and
labels each scenario with a converged power-flow solution.

`main_datagen_multiproc_improved.py` is the entry point. A `ppNR` tag in a
corpus filename means the label is pandapower's Newton–Raphson solution
(`algorithm="nr"`, `tolerance_mva=1e-8`, at most 30 iterations, reactive limits
not enforced); `cNR` means the in-repo custom solver.

## `results/`

Small artifacts only: JSON metrics, CSV summaries and generated figures, split
by the cluster account that produced them, because the two accounts are separate
filesystems and a few campaigns ran on both. Checkpoints and corpora are not
here — see [`../docs/DATA_AND_CHECKPOINTS.md`](../docs/DATA_AND_CHECKPOINTS.md).

## `provenance/`

`local_copy_manifest.json` records every file copied in, with source path and
SHA-256. `remote_transfer_log.json` records each rsync from a cluster with its
exit status. Both are written by `../tools/assemble_release.py`.
