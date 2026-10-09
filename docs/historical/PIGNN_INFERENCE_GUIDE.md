# Using a trained PIGNN-Attn-LS power-flow surrogate

How to take a checkpoint trained on one grid (SimBench here) and run it on
scenarios for that grid. Written for someone who did not train the model.

The model replaces the Newton-Raphson iteration: given the bus injections it
predicts the voltage state directly, in milliseconds rather than solver
iterations.

---

## 1. What you need

| | |
|---|---|
| `predict_pignn.py` | the inference entry point |
| `GNSMsg_SelfAttention_armijo.py` | the model |
| `Dataset_optimized_complex_columns.py` | parquet loader and per-unit conversion |
| `collate_blockdiag_optimized_complex_columns.py` | batching and the sparse Y-bus |
| `<grid>_best_model.ckpt` | trained weights |
| `<grid>_*.parquet` | scenarios to run |

All five python files must sit in the same directory; `predict_pignn.py` adds
its own directory to `sys.path`, so no installation step is needed.

### Environment

```bash
conda create -n pignn python=3.12
conda activate pignn
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.7.1+cu126.html
pip install pyarrow pandas numpy scipy
```

`torch_scatter` must match your torch/CUDA build — that is what the index URL is
for. The model imports it directly, so a mismatch fails at import.

CPU-only works (`--device cpu`) and is fine for a few hundred scenarios; it is
roughly an order of magnitude slower.

---

## 2. One command

```bash
python predict_pignn.py \
  --ckpt   SimBench_best_model.ckpt \
  --parquet SimBench_scenarios.parquet
```

Output:

```
[data] SimBench_scenarios.parquet: 36000 scenarios, 94 buses; scoring 400
[model] 65,896 parameters, K=40, d_hi=24, heads=8, layers=8, armijo=True, device=cuda

[accuracy vs the Newton reference]  n=400 scenarios
  |V| RMSE            1.1762e-03 pu
  theta RMSE          6.2692e-02 deg
  combined RMSE       1.6064e-03

[power-balance residual of the predicted state]  pu on 100 MVA
  dP_inf 1.2832e-02   dQ_inf 1.2981e-02
  mean|dP| 1.3956e-03   mean|dQ| 1.5305e-03
  p95|dP|  3.7052e-03   p95|dQ|  4.2540e-03

[speed] 15.6s total, 39.0 ms/scenario
```

Useful flags:

| flag | meaning |
|---|---|
| `--limit 500` | score only the first 500 scenarios (quick check) |
| `--batch 8` | scenarios per forward pass; see §5 |
| `--device cpu` | no GPU |
| `--save-csv out.csv` | per-scenario errors, one row per scenario |

---

## 3. Reading the two blocks

They answer different questions and can disagree.

**Accuracy vs the Newton reference** is the distance to the solution the solver
found. It is the natural "is the model right" number, but power flow can have
multiple valid solutions, so it carries a bias toward the particular solution
the data generator landed on.

**Power-balance residual** is how badly the predicted state violates the network
equations, computed from `S = V ⊙ conj(YV)`. It needs no reference solution and
is what determines whether the operating point is usable.

Active power is not scored at the slack bus and reactive power is not scored at
slack or PV buses: there the injection is a free variable that the power flow
determines, not an input the model has to match. Scoring them would charge
correct behaviour as error.

Both a mean and a max are printed because the residual distribution is heavily
skewed on most grids — a handful of buses can carry residuals orders of
magnitude above the rest. On some grids in this corpus the mean exceeds the 95th
percentile. A mean alone hides that; a max alone makes a whole grid look bad
when a few buses are responsible.

All power quantities are per-unit on a 100 MVA base, so `1e-2 pu = 1 MW/MVAr`.

---

## 4. The checkpoint does not carry its architecture

`*.ckpt` is a plain `state_dict` — weights only. `predict_pignn.py` defaults to
the configuration the released checkpoints were trained with:

```
d=4  d_hi=24  n_heads=8  num_attn_layers=8  K=40  armijo=on
```

If you load a checkpoint trained differently, pass the matching flags. A
mismatch prints a `state_dict mismatch` warning listing missing keys — **do not
ignore it**, the numbers after it are meaningless.

Armijo matters more than it looks. It is a line search on each of the K
correction steps, and on this corpus the model does not train without it at
case300 scale and above (measured: loss 1.14 with it, 505 without, and getting
worse with more epochs). Checkpoints trained with it must be run with it.

---

## 5. Choosing `--batch`

Memory scales with `batch × buses`. Measured on an A40 (48 GB) at K=40,
d_hi=24, 8 heads, 8 layers: **about 1.24 MB per bus-sample**.

| grid | buses | batch that fits comfortably |
|---|---|---|
| SimBench, case118, case300 | ≤ 300 | 32 |
| LVN_heo1 (722) | 722 | 32 |
| case1354pegase | 1354 | 20 |
| case2848rte | 2848 | 9 |
| case9241pegase | 9241 | 3 |

Inference needs less than training (no backward pass), so these are
conservative. If you hit `CUDA out of memory`, halve `--batch`; it changes
throughput, not results.

---

## 6. Bringing your own scenarios

The parquet must be in the PPC branch-row direct-SI schema the loader expects —
`bus_number`, `branch_number`, `bus_typ`, `Branch_f_bus`, `Branch_y_series_*`,
`u_start`, `u_newton` and the rest. Generate it with the pipeline in
`ScenarioSynthesis_PPC/` rather than assembling it by hand.

Two things to get right:

**The grid must match the checkpoint.** These are per-grid models: a SimBench
checkpoint on case118 scenarios will load (the architecture is
topology-agnostic) and produce nonsense. There is no guard against this.

**Row groups.** If you generate new files, write them with ~20 rows per group.
The loader reads randomly, and parquet decodes a whole row group to serve any
row in it; files written with 3000–6000 rows per group measured 3191 ms/batch
against 176 ms/batch, an 18x penalty. `rewrite_rowgroups.py` fixes existing
files and `--scan DIR` reports the layout of a directory.

---

## 7. What `u_start` is, and why the model needs it

The model does not start from nothing: it starts from the DC power-flow
initialisation stored in the parquet as `u_start` (magnitudes near nominal,
angles from the DC solution) and applies K learned correction steps. That is why
the parquet needs a start state as well as the scenario.

A consequence worth knowing: if the corrections collapse to zero, the model
outputs approximately the DC initialisation rather than garbage. So a bad model
looks like the DC baseline, not like noise. If the reported error is close to
what the DC start already gives, the model is not contributing.

---

## 8. Troubleshooting

| symptom | cause |
|---|---|
| `No module named 'torch_scatter'` | install it from the PyG index matching your torch build |
| `state_dict mismatch — missing N` | architecture flags do not match the checkpoint (§4) |
| `CUDA out of memory` | lower `--batch` (§5) |
| `NotImplementedError: aten::as_strided ... SparseCUDA` | something called `.squeeze()`/`.unsqueeze()` on the sparse Y-bus; route every `Y @ V` through `ybus_matvec` |
| very slow, low GPU use | the parquet has large row groups (§6) |
| plausible but poor numbers | wrong grid for the checkpoint (§6), or a `_A_`-tagged parquet where a `_ppNR_` one was intended |

---

## 9. Provenance of the bundled checkpoint

* Trained on the Phase-0 backbone corpus, `_ppNR_` scenario set (pandapower
  Newton oracle), 36,000 scenarios, split 1/3 train, 1/3 valid, 1/3 test,
  seed 42.
* 40 epochs, AdamW at LR 1e-4, loss `MSE(V) + 1e-2 · physics` (log-cosh),
  Armijo line search on.
* `d=4, d_hi=24, heads=8, layers=8, K=40` — the Pareto-optimal point of the
  capacity sweep in `latex/results.pdf` §4.4. Note that sweep also found
  `d_hi=64` diverging outright, so this is not a configuration to scale up
  casually.
* The checkpoint is the best-validation epoch, not the last.

**The bundled SimBench checkpoint is from epoch 12 of 40** -- the campaign was
still running when it was packaged. The numbers above are what it gives today;
replace the file from the run directory once training finishes and they will
improve.

There are no published weights for this model family anywhere; every checkpoint
is trained from scratch.
