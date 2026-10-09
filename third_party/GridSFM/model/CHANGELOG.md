# Changelog

Versioning convention: the code's minor version tracks the checkpoint
release it primarily targets. `gridsfm` `1.1.x` pairs with model
`gridsfm_open_v1.1.pt`; older `1.0.x` pairs with `gridsfm_open_v1.0.pt`.
Patch versions iterate the loader / eval / training code independently
of the checkpoint.

## 1.1.0

### Added

* **Fine-tuning support** (new public API). The package now ships the
  training loss + synthetic-infeasibility data wrapper used to fine-tune
  the released checkpoint on OPFData:

  | symbol | purpose |
  |---|---|
  | `gridsfm.compute_loss` | 13-component training loss (θ / V / Pg / Qg tanh-capped squared error + cost log-MSE + feas BCE + stress-feas + log1p-compressed KCL P/Q + log1p-compressed branch flow P/Q + thermal loading + thermal-limit barrier) |
  | `gridsfm.eval_pass` | held-out eval helper returning average `compute_loss`, per-element MAE on θ/V/Pg/Qg (θ uses circular distance), per-graph cost MAPE, and feas-head binary accuracy |
  | `gridsfm.finetune_opfdata` | standard AdamW + grad-clip training loop |
  | `gridsfm.SyntheticMixedDataset` | per-graph perturbation wrapper (4 modes: voltage_squeeze, thermal_bottleneck, angle_tighten, capacity_aware_spike) |
  | `gridsfm.OPFDataAdapterDataset` | PyG `OPFDataset` adapter with train/val/test splits |

  See `examples/finetune_opfdata_case6470.ipynb` for a few-shot study
  fine-tuning the v1.1 release on `pglib_opf_case6470_rte`.

  **Fine-tuning is supported on the v1.1 checkpoint only, and on the
  OPFData dataset format only.** The synthetic perturbations and loss
  column conventions are pinned to OPFData's schema; custom HeteroData
  scenarios from other sources are inference-only (use `predict()` /
  `model(batch)`).

  Loading a v1.0 weight via the cross-version adapter explicitly zeroes
  the fusion `W_global` max-pool channels (see the v1.0 → v1.1 section
  under `Changed` for the column layout); gradients will move them, but
  the resulting checkpoint is no longer v1.0-shaped.

### Changed

* **Fusion layer `W_global` width: 4d → 8d** (`blocks.py:FusionLayer`).
  Per-graph global readout now concatenates mean **and** max pool of
  each of the four node-type buckets (bus / branch_ac / branch_tr /
  cycle). v1.0 weights for this layer use mean-only; the v1.1
  checkpoint includes the additional max-pool projections.

* **`load_model` is now cross-version (v1.0 → v1.1 only)**. The loader
  verifies the SHA-256 `metadata.hash` strictly (corruption /
  partial-download still raises), then adapts the one tensor whose
  layout differs between releases: `fusion.W_global.weight`. The
  adapter does a column-block permutation: a v1.0 weight's four
  `d`-wide mean column blocks land in the v1.1 layout's even (mean)
  slots; the odd (max) slots are zeroed (NOT left at the `nn.Linear`
  Kaiming init; the adapter explicitly writes 0 into them so the v1.0
  mean pathway dominates at load). Any other shape mismatch, plus
  any ckpt-key not present in the model OR any model parameter not
  present in the ckpt, raises `RuntimeError`. The loader does NOT
  silently zero-pad / crop / discard / random-init, since each of
  those would mask an architecture mismatch slipping past the hash
  check. The reverse direction (v1.1 → a v1.0-arch backbone) is not
  supported.

* **`compute_physics_stress(...)` signature change** (breaking).
  `n_graphs` is now a required keyword-only argument. The previous
  auto-derive from `data.num_graphs` was fragile on single-graph
  forwards that didn't set the attribute; callers must pass it
  explicitly. All in-tree callers updated.

### Deprecated

* **`gridsfm_open_v1.0.pt` is deprecated** and will be removed in a
  future minor release. New users should adopt `gridsfm_open_v1.1.pt`
  (recommended for inference, required for fine-tuning).

## 1.0.0

Initial public release of the inference package.

* `GridTransformerBackbone` + `load_model` + `predict` API.
* Hodge PE + cycle-basis preprocessing pipeline.
* HuggingFace loader (`load_from_hf`, `gridsfm.hf_util.GridSFM_PG_Loader`).
* OPFData example (`examples/opfdata.py`) + 53 shipped sample scenarios
  (`examples/infer_samples.py`).
