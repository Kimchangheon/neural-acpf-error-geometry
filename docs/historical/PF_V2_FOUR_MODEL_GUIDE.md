# Four AC power-flow surrogates on the ppNR v2 corpus

A working guide for running, testing and extending the four PF surrogates —
**PIGNN-Attn-LS**, **GridFM/graphkit**, **GridSFM** and **LUMINA** — on the
regenerated 31-grid corpus.

Companion to `GRIDFM_GRIDSFM_WINDOWS_GPU_RUN.md`, which covers the older
SimBench/LVN/ENTSO-E work. This document is about power flow only, on the v2
data, for all four models.

---

## 0. Read this first: what is and is not settled

The campaign is **not finished**. Numbers below are current, not final.

| Model | Trained | Usable | Missing |
|---|---|---|---|
| `graphkit` | 30/31 | 30 | case9241pegase |
| `lumina` | 30/31 | 30 | case9241pegase |
| `gridsfm` | 26/31 | 26 | the 5 largest grids |
| `pignn` | 26/31 | **22** | the 5 largest + 4 diverged |

"Usable" excludes runs whose validation RMSE never moved. Two failure modes are
known and documented in §7 — read that section before trusting any large-grid
cell, and before assuming a flat metric means the model is bad.

**One habit worth adopting from this project:** every number here was checked
against something independent before being believed. Re-scoring a checkpoint
must reproduce the training log's RMSE; a fixed input must be verified by
reconstructing it from the graph the model actually receives. Several
conclusions in earlier reports were wrong until that check was run.

---

## 1. Where everything lives

### Reports

| File | What it answers |
|---|---|
| `latex/01_data/ppnr_v2_dataset_overview.tex` | how the v2 corpus was generated, and what changed from v1 |
| `latex/02_results/pf_v2_results.pdf` | current four-model comparison on v2 |
| `latex/02_results/pf_v1_vs_v2.pdf` | whether the regenerated corpus changed each model's accuracy |
| `latex/02_results/pf_six_model_comparison.pdf` | the older six-model study on v1 (includes the GridFM mirrors) |
| `latex/02_results/pf_gridfm_report.pdf` | GridFM-family deep dive: pinning, collapse diagnostic |
| `latex/02_results/lvn_heo1_experiment_report.pdf` | LVN Heo1 history, including the Armijo/detach diagnosis |
| `latex/03_reference/model_loss_functions.tex` | the loss each model is trained against |

### Upstream model packages

| Model | Package | Note |
|---|---|---|
| `pignn` | in-repo `GNSMsg_SelfAttention_armijo.py` | ours; unrolled solver |
| `graphkit` | `gridfm_graphkit` (IBM, released) | no released weights |
| `gridsfm` | `GridSFM/model/gridsfm` (released) | no released weights loaded here |
| `lumina` | `lumina-sdk` → `lumina_inference` | needs `lumina_config.json` even from scratch |

All four are trained **from scratch**. Nothing in this campaign measures a
pretrained model.

### Data

31 parquet files, one per grid, 111 GiB total.

```
/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/
```

**Select v2 by its configuration tag, not by the solver tag:**

```bash
ls *_u0clean_sgenpert_pvqvstart_*directSI.parquet   # exactly 31 files
```

`_ppNR_` alone now matches several generations and will silently mix corpora.
`manifest_ppnr_v2.sh` in the repo carries the authoritative list with bus counts
and byte sizes.

Filename fields, e.g.
`case118_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37000_NR_branchrows_directSI.parquet`:

| Field | Meaning |
|---|---|
| `ppcY` | Y-bus taken from the pandapower PPC |
| `backbone` | scenario preset |
| `dc_compile` | initial voltage from a DC solve |
| `ppNR` | labels solved by pandapower Newton–Raphson |
| `ls0.60-1.40` | global load scale ±40% |
| `u0clean` | **no start-point jitter** (v2 change) |
| `sgenpert` | `net.sgen` perturbed with the load scale (v2 change) |
| `pvqvstart` | PV-bus Q taken from `u_start` (v2 change) |
| `37000` | true row count — trustworthy in v2, was attempts in v1 |

---

## 2. Environment

Python 3.12.

```bash
pip install torch==2.7.1 torch-geometric==2.8.0 numpy pandas pyarrow scipy
```

Per model, additionally:

```bash
# graphkit
pip install --no-deps -e /path/to/gridfm-graphkit
pip install opt_einsum lightning matplotlib torch_scatter
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.7.1+cu126.html

# gridsfm
pip install --no-deps -e /path/to/GridSFM/model

# lumina
pip install --no-deps -e /path/to/lumina-sdk
```

Check the import before queuing anything:

```bash
python -c "import gridfm_graphkit, gridsfm, lumina_inference; print('ok')"
```

### The data pipeline is shared and must not be changed per model

```
Dataset_optimized_complex_columns.py            parquet -> tensors, per-unit rebase
collate_blockdiag_optimized_complex_columns.py  batching, sparse block-diagonal Y-bus
```

Every run uses `--PER_UNIT --target_S_base 1e8 --dataset_complex_dtype complex128`.
Changing any of these makes results incomparable across models.

---

## 3. Training

One dispatcher covers all four models and handles cluster placement, batch
sizing and staging:

```bash
MODEL=pignn TIER=A bash dispatch_pf_v2.sh
```

`TIER` selects the hardware and the grid set:

| Tier | Grids | Host | GPU |
|---|---|---|---|
| A | 10 smallest (case4gs … case33bw) | alex | a40 |
| B | case39 … case_illinois200 | alex2 | a100 |
| C | case300 … case1888rte | alex | a100 + `--constraint=a100_80` |
| D | GBnetwork … case9241pegase | helma | h100 |

Useful overrides: `GRIDS="case14 case118"`, `EPOCHS=20`, `LR=1e-4`,
`PARTITION_OVERRIDE=preempt`, `BN_BUDGET=...`.

### Running one grid by hand

```bash
python train_valid_test_gridfm.py \
  --PARQUET case118_..._directSI.parquet --task pf --gridfm_impl graphkit \
  --run_name my_run --log_to_file --log_dir results/logs/x --ckpt_dir results/ckpt/x \
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 \
  --BATCH 64 --EPOCHS 40 --LR 5e-4 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 \
  --hidden_size 48 --num_layers 12 --n_heads 8 \
  --zero_init_head --vn_feature_mode log --feature_transform signed_log \
  --mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh
```

Per-model driver and the flags that define each model:

| Model | Driver | Model-defining flags | LR |
|---|---|---|---|
| `pignn` | `train_valid_test.py` | `--BLOCK_DIAG --PINN --d 4 --d_hi 24 --n_heads 8 --num_attn_layers 8 --K 40 --use_armijo --armijo_mode geometric_safe` | 1e-4 |
| `graphkit` | `train_valid_test_gridfm.py` | `--task pf --gridfm_impl graphkit --hidden_size 48 --num_layers 12 --n_heads 8 --zero_init_head` | 5e-4 |
| `gridsfm` | `train_valid_test_gridsfm.py` | `--task pf --init_mode scratch` | 1e-4 |
| `lumina` | `train_valid_test_lumina.py` | `--task pf --init_mode scratch --model_config <path>/lumina_config.json` | 1e-4 |

**`--armijo_mode` is not optional for PIGNN.** Its default is `fixed`, which
discards the solver step when the line search fails, leaving the output equal to
the constant input and the loss detached from the parameters. That silently
killed 11 of 31 runs in the v1 campaign — no error, no warning beyond a flood of
`[warn] physics loss detached`. Always pass `geometric_safe`.

**LUMINA needs its config file even when training from scratch.** It reads the
architecture from the released `lumina_config.json`, and compute nodes have no
route to the HuggingFace Hub. Ten runs died on `FileNotFoundError` because the
file existed under one cluster account and not the other.

### Batch size

Memory per bus-sample was measured, not guessed:

| Model | KiB per bus-sample | Budget on a 48 GiB a40 |
|---|---|---|
| `gridsfm` | ~1040 | B·N ≤ 30,000 |
| `graphkit` | ~479 | B·N ≤ 60,000 |
| `lumina` | ~145 | B·N ≤ 200,000 |
| `pignn` | K=40 unrolled | B·N ≤ 27,000 |

The dispatcher scales these by the card's memory. GridSFM is **seven times**
heavier per bus-sample than LUMINA, so a single shared rule is wrong for one of
them. If you change the model shape, re-measure:

```bash
python probe_batch_memory.py --model gridsfm --PARQUET big.parquet --batches 1,2,4,8,16
```

Do not extrapolate from a job that OOM'd — the allocator's ceiling at failure is
not the model's demand. Estimating that way once gave 166 KiB/node against a
true 479, and cost two rounds of jobs.

---

## 4. Checkpoints

```
results/ckpt/pfv2_<model>_<tier>_<stamp>/
```

Naming differs by driver:

| Model | Filename |
|---|---|
| `pignn` | `pfv2_pignn_<grid>_b<batch>_<epochs>_best_model.ckpt` |
| `graphkit` / `gridsfm` / `lumina` | `pfv2_<model>_<grid>_b<batch>_best.pt` |

**A checkpoint exists from the first improving epoch onward.** Its presence does
not mean the run finished. Gate on the training log's test block instead:

```bash
grep -l 'Final test-set RMSE\|Test physics-loss' results/logs/pfv2_*/*_training_log.txt | wc -l
```

Scoring a mid-training snapshot silently produces numbers that read as final.
This has bitten this project more than once.

---

## 5. Testing a trained model

### Voltage error and the collapse diagnostic

```bash
python rescore_pf.py --PARQUET grid.parquet \
  --ckpt results/ckpt/.../pfv2_graphkit_case118_b64_best.pt \
  --grid case118 --impl graphkit --BATCH 32 --json_out out.json
```

For PIGNN use `rescore_pignn.py` (same interface, different model construction).

**Always check that the re-score reproduces the training log's RMSE.** That
agreement is what proves the split and the checkpoint line up. For the GridFM
family all 93 re-scores matched to better than 2e-4 relative; for PIGNN, which
runs 40 float32 solver iterations, agreement is ~0.7% and that looseness is
structural, not a bug.

### What the metrics mean

| Metric | Reads |
|---|---|
| `rmse_vmag_pu`, `rmse_theta_deg` | voltage state error |
| `dPinf`, `dQinf` | worst-bus power-balance residual |
| `mean|dP|`, `p95|dP|` | residual distribution — report both |
| `slope_vmag_pq`, `R2_vmag_pq` | **collapse detector**, PQ buses only |

**Why the slope matters.** |V| in pu spans only ~0.95–1.10, so a flat predictor
of V = 1.0 posts a small MAE while carrying no information. Slope 1 means the
model tracks the reference spread one-for-one; well below 1 means it is
collapsing toward the mean. This caught a *negative* slope on case145 (prediction
anti-correlated with the reference) that MAE alone rated merely "poor".

Use the **PQ-restricted** columns for comparisons. Models that copy |V| through
at PV and slack buses are exact there by construction, which flatters a pooled
number for reasons unrelated to modelling.

### Free-standing inference

`pf_predict.py` loads a checkpoint and prints pooled and PQ-only errors for the
GridFM family. `predict_pignn.py` does the same for PIGNN — but note it omits
`n_nodes_per_graph`, so figures from it on a block-diagonal batch should be
re-checked against a training log first.

---

## 6. Reproducibility rules

- Split is `random_split` with lengths from `train_ratio`/`valid_ratio` and
  `--seed_value 42`. Every driver does this identically, so the held-out rows
  are the same across models. **Do not change the ratios if you want your run to
  be comparable to the tables.**
- Bus-type numbering in this pipeline is **1 = slack, 2 = PV, everything else
  PQ** — verified on case118 (1 slack, 53 PV, 64 PQ). This is *not* MATPOWER
  numbering, which the OPFData path uses. Getting it backwards once produced a
  PQ metric that selected the slack bus alone and reported exactly 0.000e+00.
- `--PER_UNIT` means branch admittances are already converted. Applying the base
  again is wrong by `Vbase²/S_base` and will not raise — it just trains on a
  Y-bus that is off by a factor of ~750.
- Residual masking: P is not scored at the slack, Q is not scored at slack or
  PV, because there the injection is a free variable.

---

## 7. Known failure modes

### PIGNN: gradient detached (v1, fixed)

`--armijo_mode fixed` discarded the update whenever neither the line search nor
its fallback reduced the mismatch. Over K=40 steps the returned voltage was the
constant input, so the loss carried no gradient and the optimiser step was a
no-op. Signature: `[warn] physics loss detached` on 96–99.6% of batches, and
validation RMSE taking one or two distinct values across all 41 epochs.

Measured directly with `probe_armijo_detach.py`:

```
[fixed         ] output attached to params: False   grad norm: DETACHED
[geometric_safe] output attached to params: True    grad norm: 2.707e-01
```

Fixed by `--armijo_mode geometric_safe`.

### PIGNN: divergence to NaN (v2, open)

On LVN_heo1, case1888rte, GBnetwork and case2848rte the v2 runs produce a
constant validation RMSE — but the guard fired **zero** times, so this is not
the v1 defect. The gradient exists and is too large:

```
Epoch  0 | train loss 1.1116e+04  rmse 1.0297e+00  (dPinf 1.126e+05 pu)
Epoch 40 | train loss nan         rmse nan         (dPinf nan)
```

The reported RMSE is constant because the best checkpoint never moves past
epoch 0. **This is an open problem and a good first experiment for a student:**
gradient clipping or a lower LR on these grids, neither of which has been tried.

### GridSFM / LUMINA: injection clamped away (fixed)

Both put the bus injection on a load node as `clamp(-S, min=0)` and gave a
generator node only to slack and PV buses, so a PQ bus with net generation
reached the model as a zero load — 99.7% of the active-power magnitude on
SimBench's PQ buses — while the physics loss still scored against the true S.

Fixed by `--pf_injection signed` (now the default). Verify with:

```bash
python probe_graph_inputs.py --model gridsfm --PARQUET g.parquet --mode signed
```

Note the honest outcome: fixing this did **not** improve either model
materially. It was a real defect, but not the reason those two trail.

### Wall clock on the largest grids

An epoch costs 2900–6700 s on the five biggest grids, so 40 epochs does not fit
a 24 h limit. helma's `preempt` partition allows 48 h — but in practice all 12
jobs sent there were **preempted within ~3 hours**, and the drivers have no
resume, so the work was lost. If you need those grids, either cut epochs to fit
24 h (and say so when reporting) or add resume to the driver.

---

## 8. Current results

Medians over the 22 grids where all four models produced a usable run:

| | **pignn** | graphkit | gridsfm | lumina |
|---|---|---|---|---|
| \|V\| RMSE [pu] | **1.87e-04** | 4.87e-03 | 1.07e-02 | 2.70e-02 |
| θ RMSE [deg] | **0.078** | 0.368 | 5.84 | 3.68 |
| ΔP∞ [pu] | **1.60e-02** | 2.34e-01 | 8.32e-01 | 3.01 |
| mean\|ΔP\| [pu] | **1.75e-03** | 3.90e-02 | 9.99e-02 | 4.00e-01 |
| p95\|ΔP\| [pu] | **4.40e-03** | 1.22e-01 | 4.38e-01 | 1.38 |

The order `pignn < graphkit < gridsfm < lumina` holds on all eight metric
families with no crossing, on both v1 and v2.

**This is not a like-for-like architecture comparison.** PIGNN runs 40
physics-informed solver steps with 66k parameters — closer to a learned Newton
iteration — and pays for it at inference. GridSFM and LUMINA are OPF
architectures asked to do PF from scratch, with hyperparameters carried over
rather than tuned. Their poor showing is a fact about this setup, not a verdict
on the models.

---

## 9. Suggested experiments

Ordered by how much they would settle:

1. **Fix PIGNN's divergence.** Gradient clipping or LR on the four NaN grids.
   Directly unblocks 4 of 31 cells.
2. **Tune LR and epochs for GridSFM and LUMINA.** The strongest remaining
   explanation for their gap is that 40 epochs at LR 1e-4 is simply the wrong
   budget for them; it has never been tested on this corpus.
3. **Add resume to the drivers.** `train_valid_test_gridfm.py` already has
   `--resume_state_dict`; `train_valid_test.py` does not. This is what makes the
   five largest grids reachable at full epoch budget.
4. **Multi-seed.** Every number is one run. The same configuration has given
   1.040e-03 and 4.300e-03 on case14 across clusters, so differences below a
   factor of two should not be read as real. `SEEDS="42 43 44"` exists in the
   dispatchers and has never been used.
5. **Slope/R² for GridSFM and LUMINA.** They have no re-scoring pass, so the
   collapse diagnostic is blank for them.

---

## 10. File map

| File | Role |
|---|---|
| `dispatch_pf_v2.sh` | training dispatcher, all four models, tier placement |
| `manifest_ppnr_v2.sh` | the 31 v2 files with bus counts and sizes |
| `train_valid_test.py` | PIGNN driver |
| `train_valid_test_gridfm.py` | GridFM/graphkit driver |
| `train_valid_test_gridsfm.py` | GridSFM driver |
| `train_valid_test_lumina.py` | LUMINA driver |
| `Dataset_optimized_complex_columns.py` | parquet loader, per-unit, Y-bus |
| `collate_blockdiag_optimized_complex_columns.py` | batching, sparse Y-bus |
| `rescore_pf.py` / `rescore_pignn.py` | re-score a checkpoint, with diagnostics |
| `pf_predict.py` / `predict_pignn.py` | standalone inference |
| `prediction_diagnostics.py` | MAE, regression slope/R² |
| `probe_batch_memory.py` | measure per-bus-sample memory |
| `probe_graph_inputs.py` | verify the injection reaches the model |
| `probe_armijo_detach.py` | verify PIGNN gradients are attached |
| `probe_injection_clipping.py` | quantify injection lost to clamping |
