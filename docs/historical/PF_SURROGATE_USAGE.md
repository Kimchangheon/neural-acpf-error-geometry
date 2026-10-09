# Using the trained AC power-flow surrogates

Three PF surrogates were trained on the Phase-0 backbone corpus, one model per
grid. This is how to take one of them — SimBench is the worked example — and run
it on your own machine.

Companion to `GRIDFM_GRIDSFM_WINDOWS_GPU_RUN.md`, which covers the older
GridFM / GridSFM / LUMINA OPF work. This document is only about power flow.

---

## 0. Status

All three campaigns are complete: 31 grids x 3 implementations, 93 training runs
and 93 validation passes, every one finished.

| Variant | Training | SimBench checkpoint |
|---|---|---|
| `mirror` | 31/31 | final |
| `mirrorpin` | 31/31 | final |
| `graphkit` | 31/31 | final |

An earlier version of this bundle shipped `mirrorpin` as a mid-training snapshot
named `..._PROVISIONAL_epoch29of40.pt`. **That file is superseded and has been
removed.** If you still have a copy, delete it and use
`pf_mirrorpin_SimBench_b64_best.pt` (md5 `260dae25a75ff936f8432fc8cd9d53c9`);
the provisional numbers were worse and should not be quoted.

## 1. What the three variants are

All three predict the same thing — the bus voltage state \|V\|, θ given the
injections — and all three are **trained from scratch**. `gridfm_graphkit` ships
no weights, so none of these is "pretrained" and none should be described that
way.

| Variant | Model | \|V\| at PV/slack, θ at slack |
|---|---|---|
| `mirror` | `GridFMHeteroSurrogate` (ours, GridFM-style) | **predicted** |
| `mirrorpin` | the same model, `--pin_known` | **held to the given input** |
| `graphkit` | IBM `gridfm_graphkit` `GNS_heterogeneous` | held to the given input |

### Why `mirrorpin` exists

In AC power flow \|V\| is an *input* at PV and slack buses, and the angle is an
input at the slack; only the rest is unknown. `mirror` ignores this and predicts
a correction at every bus, so it can — and does — walk away from values it was
handed. Measured on case118, where 46% of buses have \|V\| given:

```
mean |V| moved away from the given value : 2.253e-03 pu
mean |V| error against the solver        : 2.253e-03 pu   <- identical
max                                      : 1.648e-02 pu
angle error at the slack bus             : 0.150 deg
```

The first two lines agreeing to every digit is the whole point: at those buses
the given value *is* the answer, so whatever the model moves is exactly what it
gets wrong. `mirrorpin` fixes it; a verification run confirms the shift becomes
exactly `0.000e+00` at PV/slack while PQ buses keep their full freedom.

**`mirror` is kept only for reproducibility of the first campaign.** For new
work use `mirrorpin` or `graphkit`.

### A consequence for comparisons

`graphkit` and `mirrorpin` are exact by construction at the buses they pin, so a
pooled \|V\| error compares them favourably against `mirror` for a reason that
has nothing to do with modelling quality. Compare on PQ buses only —
`pf_predict.py` prints that line separately.

---

## 2. Get the files

Everything is on the transfer server under `~/PIGNN-Attn-LS/pf_release/`:

```bash
scp -r chkim@131.188.35.62:~/PIGNN-Attn-LS/pf_release .
```

PowerShell is the same command. Contents:

```text
pf_release/
├── ckpt/
│   ├── pf_mirror_SimBench_b64_best.pt      77 MB
│   ├── pf_mirrorpin_SimBench_b64_best.pt   77 MB
│   └── pf_graphkit_SimBench_b64_best.pt    77 MB
├── data/
│   └── SimBench_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_siNR_36000_NR_branchrows_directSI.parquet   393 MB
├── logs/            training logs for all three runs
└── scripts/         everything needed to load and run a checkpoint
```

### Which parquet

The filename carries the provenance and it matters:

- `_ppNR_` — solved by **pandapower** Newton-Raphson. This is the Phase-0
  backbone corpus. `LVN_heo1` instead carries `_cNR_` because pandapower cannot
  solve that grid and it was produced with the custom NR oracle.
- Do **not** select files by size or by the `_A_` tag; `_A_` is a different,
  older scenario set. Select on the solver tag.
- `_rg20` means the file was rewritten to 20 rows per parquet row group.
  Training and evaluation shuffle, so a file written with 3000–6000 rows per
  group decodes a whole group to serve one sample — measured at 3191 ms/batch
  against 176 ms/batch, an 18× penalty. SimBench was already at 20 and has no
  suffix.

---

## 3. Environment

Python 3.12, and the loader needs the same numerical setup the models were
trained with:

```bash
pip install torch==2.7.1 torch-geometric==2.8.0 numpy pandas pyarrow scipy
```

For the `graphkit` variant only, additionally:

```bash
pip install --no-deps -e /path/to/gridfm-graphkit
pip install opt_einsum lightning matplotlib torch_scatter
```

`torch_scatter` must match your torch/CUDA build:

```bash
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.7.1+cu126.html
```

CPU works; a GPU is only needed for speed. On an H100 the SimBench test split
runs in well under a minute.

---

## 4. Run it

```bash
cd pf_release/scripts
export PYTHONPATH=$PWD

python pf_predict.py \
  --PARQUET ../data/SimBench_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_siNR_36000_NR_branchrows_directSI.parquet \
  --ckpt   ../ckpt/pf_mirror_SimBench_b64_best.pt \
  --variant mirror
```

Output:

```text
[model] mirror  params=20,049,458  device=cuda
[error] |V|  MAE 8.3092e-04  RMSE 1.0483e-03 pu
[error] theta MAE 2.1671e-02  RMSE 3.3881e-02 deg
[error] |V| at PQ only: MAE 8.5579e-04  RMSE 1.0653e-03 pu
```

The other two, same parquet:

```bash
python pf_predict.py --PARQUET ../data/SimBench_...parquet \
  --ckpt ../ckpt/pf_graphkit_SimBench_b64_best.pt --variant graphkit

python pf_predict.py --PARQUET ../data/SimBench_...parquet \
  --ckpt ../ckpt/pf_mirrorpin_SimBench_b64_best.pt --variant mirrorpin
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--n 2000` | cap the number of scenarios; leave off for the whole test split |
| `--split all` | score every scenario instead of the held-out third |
| `--save_npz out.npz` | write predictions and references for your own analysis |
| `--device cpu` | force CPU |
| `--BATCH 32` | batch size; see §7 on memory |

### `--variant` has to be right

It is not stored in the checkpoint. `graphkit` and `mirror` have different
parameter names, so a mismatch fails loudly at `load_state_dict`. But `mirror`
and `mirrorpin` **share identical weights** and differ only in whether known
quantities are pinned — a mismatch there loads cleanly and silently changes what
is predicted. Match the variant to the filename.

---

## 5. Reference numbers, SimBench

Held-out test split, all three final checkpoints:

| Variant | RMSE | \|V\| MAE | \|V\| RMSE | θ RMSE | slope \|V\| | R² \|V\| |
|---|---|---|---|---|---|---|
| `mirror` | 1.203e-03 | 8.297e-04 | 1.048e-03 | 0.034° | 0.998 | 0.9986 |
| `mirrorpin` | 1.491e-03 | 1.078e-03 | 1.319e-03 | 0.040° | 0.998 | 0.9977 |
| `graphkit` | 3.885e-03 | 1.279e-03 | 1.584e-03 | 0.203° | 0.992 | 0.9970 |

If your run reproduces the \|V\| RMSE above, your environment is right. Each
shipped checkpoint was run through `pf_predict.py` and reproduces its training
run's RMSE exactly.

**SimBench is not the grid that separates these variants.** It is 97% PQ buses,
so pinning applies to almost nothing and all three land within a factor of 1.5.
`mirror` posting the best number here is not evidence that it is the better
model — across the full 31-grid corpus, restricted to PQ buses where all three
are genuinely predicting, `mirrorpin` wins 16 grids, `graphkit` 12 and `mirror`
3. The split is mostly by size: `graphkit` wins 12 of the 15 grids at or below
118 buses, `mirrorpin` 14 of the 16 above.

**On `slope` and `R²`:** these come from a pooled regression of predicted
against reference values, and they answer a question MAE cannot. |V| in pu only
spans about 0.95–1.10, so a flat predictor of `V = 1.0` posts a small MAE while
carrying no information at all. Slope 1 means the model tracks the reference
spread one-for-one; slope well below 1 means it is collapsing toward the mean.
SimBench is healthy at 0.998. Much of the corpus is not: on PQ buses, 17 of 31
grids fall below 0.9 under `mirror` and 7 below 0.5, and `graphkit` reaches a
*negative* slope on case145 — its prediction is anti-correlated with the
reference there. Check the slope before trusting a small MAE.

## 6. Other grids

31 grids were trained, 4 to 9241 buses. Checkpoints follow the same naming:

```text
results/ckpt/<group>/pf_<variant>_<grid>_b<batch>_best.pt
```

with these groups on helma:

| Variant | Group |
|---|---|
| `mirror` | `pf_gridfm_h100_20260810_221252` |
| `graphkit` | `pf_gridfm_h100_20260811_135024` |
| `mirrorpin` | `pf_gridfm_h100_20260812_120439` |

The batch in the filename is what that grid was trained with, and it varies with
grid size (64 down to 12) — see §7. Ask for a specific grid to be packaged, or
copy it yourself from
`/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/`.

Two grids need a caveat before their numbers are used:

- **case145** has 7885 scenarios where every other grid has ~36000 — 22% of the
  data. It is the worst grid on every metric, and a ±20% regeneration is
  outstanding. Do not compare it against the others.
- Scenario counts are not uniform elsewhere either: case9241pegase 18280,
  case6515rte 20135, case6495rte 23332, case6470rte 25551, case300 27994.

---

## 7. Memory, if you retrain

Two things bound the batch size, and only one of them is the GPU.

**The Y-bus used to be the wall.** The collate built a *dense* block-diagonal
Y-bus, costing `(B·N)² × 16` bytes — quadratic in batch times buses. case2848rte
at batch 32 asked for 124 GB and every large grid was OOM-killed. It is now
assembled as sparse COO: on case2848rte at batch 8 that is **7921 MB → 1.8 MB**,
verified numerically identical (the implied injection agrees to 1.1e-11
absolute, i.e. complex128 rounding).

**What binds now is activation memory**, measured directly on case9241pegase:

| batch | B·N | peak GPU |
|---|---|---|
| 1 | 9,241 | 4.3 GB |
| 8 | 73,928 | 33.6 GB |
| 16 | 147,856 | 67.0 GB |
| 24 | 221,784 | OOM (93.3 GB usable) |

Linear at ~479 KB per node for this model shape (hidden 48, 12 layers, 8 heads).
The dispatcher picks the batch from a `B·N ≤ 120000` budget, which is ~54 GB.
Re-measure if you change the model shape — the constant belongs to the
architecture, not the data.

Do not extrapolate from the point where a job *failed*: the memory reported at
an OOM is the allocator's ceiling, not the model's demand. Estimating that way
gave 166 KB/node against a true 479 KB/node and cost two failed campaigns.

---

## 8. Retraining from scratch

```bash
python train_valid_test_gridfm.py \
  --PARQUET grid.parquet --task pf \
  --gridfm_impl mirror \        # or: graphkit
  --pin_known \                 # mirror only; omit for the original behaviour
  --run_name my_run \
  --log_to_file --log_dir results/logs/my_group --ckpt_dir results/ckpt/my_group \
  --PER_UNIT --target_S_base 1e8 --share_grid --lazy_parquet --row_group_cache_size 4 \
  --dataset_complex_dtype complex128 \
  --BATCH 64 --EPOCHS 40 --LR 5e-4 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 \
  --mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh \
  --hidden_size 48 --num_layers 12 --n_heads 8 \
  --zero_init_head --vn_feature_mode log --feature_transform signed_log
```

`--task pf` is the default. On a cluster, `dispatch_pf_gridfm.sh` handles grid
selection, batch sizing and staging; `IMPL=graphkit` and `PIN=1` select the
variant, and `SEEDS="42 43 44"` runs multiple seeds per grid.

### Multiple seeds

Single runs are not enough to rank models here. The same configuration gave
1.040e-3 on one cluster and 4.300e-3 on another for case14, and two GridSFM
scratch runs swapped rank between clusters. PFDelta's benchmark trains every
model three times and reports the standard deviation; `SEEDS` implements that,
and `summarize_rescore.py` reports mean ± sd when it finds more than one seed.
**The results in §5 are single-seed and carry no error bar.**

---

## 9. Scripts

| File | Role |
|---|---|
| `pf_predict.py` | load a checkpoint and predict — the entry point |
| `train_valid_test_gridfm.py` | training driver, all three variants |
| `Dataset_optimized_complex_columns.py` | parquet loader, per-unit conversion, Y-bus |
| `collate_blockdiag_optimized_complex_columns.py` | batching, sparse block-diagonal Y-bus, `ybus_matvec` |
| `gridfm_graphkit_adapter.py` | `graphkit` only: model construction, edge attributes, masks |
| `known_operator_pf.py` | imported by the training driver |
| `prediction_diagnostics.py` | MAE, median, regression slope/R/R² |
| `rescore_pf.py` | re-score a saved checkpoint without retraining |
| `summarize_rescore.py` | aggregate re-scores, mean ± sd across seeds |
| `rewrite_rowgroups.py` | fix parquet row-group layout (`--scan` to inspect only) |
| `dispatch_pf_gridfm.sh` | cluster dispatcher: grids, batch sizing, staging |
| `dispatch_rescore_pf.sh` | cluster dispatcher for re-scoring |

`pf_predict.py` needs the first six. The rest are for retraining or analysis.

---

## 10. Two conventions that will bite you

**Bus type numbering differs between the two pipelines.** In the parquet
pipeline used here it is **1 = slack, 2 = PV, everything else = PQ** — verified
on case118, which has 1 slack, 53 PV and 64 PQ. The OPFData pipeline elsewhere
in this repo uses MATPOWER numbering (1 = PQ, 2 = PV, 3 = ref). Getting this
backwards produced a PQ-restricted metric that selected the slack bus alone and
reported an error of exactly `0.000e+00`, which looks like a perfect model.

**`--PER_UNIT` means the branch admittances are already converted.** The loader
has applied the local bus bases before the model sees them. Applying them again
— the natural thing to do, since the SI reconstruction path works in SI — is
wrong by `Vbase²/S_base`. An early version of the graphkit edge adapter did
exactly that and missed the Y-bus by a relative 7.5e2. It did not raise; it
trained and produced plausible-looking numbers. `validate_edge_attr_parquet`
exists to catch it and passes at machine precision (rel_err 1.4e-16 … 2.6e-16
across case118, SimBench, LVN_heo1, GBnetwork).
