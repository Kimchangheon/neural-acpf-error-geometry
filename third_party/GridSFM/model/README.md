# GridSFM Model - AC-OPF Inference

Minimal package for running the released **GridSFM-Open** AC-OPF foundation model.

> **Supported grid size: ≥500 buses.** GridSFM-Open is trained on grids with at least 500 buses. Smaller cases (`case14_ieee`, `case30_ieee`, `case57_ieee`, `case118_ieee`) are out of distribution and will produce meaningless results.

## Layout

```
model/
├── gridsfm/        # inference package
├── samples/        # 53 base scenarios (.pyg.json); see samples/README.md
├── examples/       # infer_samples, opfdata, finetune_opfdata_case6470, predict_to_viewer
├── tests/          # pytest suite
├── checkpoints/    # local cache for downloaded ckpts (gitignored)
├── pyproject.toml
└── README.md
```

License: top-level `LICENSE` covers this package.

## Install

Tested on Ubuntu 22.04 / 24.04 and macOS 14+. Python 3.10+, primary CI on 3.12.

```bash
cd model
python -m venv .venv
source .venv/bin/activate
pip install -e .            # package + runtime deps
pip install -e ".[test]"    # also pytest
```

Dependency pins live in `pyproject.toml` (`torch >=2.6,<2.9`, `torch_geometric >=2.5,<3`, `numpy <3`, `scipy <2`).

## Run the tests

```bash
cd model
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
python -m pytest -v
```

Always invoke as `python -m pytest`, not bare `pytest` - a user-local `~/.local/bin/pytest` on `$PATH` will shadow the venv's. Tests that need the checkpoint `pytest.skip` if it's absent.

## Get the checkpoint

```python
from gridsfm import load_from_hf
model = load_from_hf("microsoft/GridSFM_Open")
```

Or download once and load locally:

```bash
hf download microsoft/GridSFM_Open gridsfm_open_v1.1.pt --local-dir checkpoints
```

```python
import torch
from gridsfm import load_model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = load_model("checkpoints/gridsfm_open_v1.1.pt", device=device)
```

### Checkpoint versions

The package supports **both** released checkpoints out of the box:

| ckpt | status | notes |
|---|---|---|
| `gridsfm_open_v1.1.pt` | **recommended** | current backbone (fusion uses mean + max pool per node type, `W_global` is `nn.Linear(8d, d)`). Required for fine-tuning. |
| `gridsfm_open_v1.0.pt` | supported, deprecated | older backbone (mean-only pool, `W_global` is `nn.Linear(4d, d)`). `load_model` permutes the v1.0 mean columns into the v1.1 layout's mean slots and explicitly zeroes the max slots (NOT the `nn.Linear` Kaiming init). Will be removed in a future release. |

`load_model` verifies the SHA-256 hash baked into `metadata.hash` before adapting any shapes, so cross-version load can't silently mask a corrupted file. Fine-tuning is only validated on v1.1; the v1.0 → v1.1 column-permutation load is intended for inference workflows.

## Quickstart

Run the model on all 53 shipped samples in one batched forward and print per-case predictions:

```bash
cd model
python examples/infer_samples.py              # CPU, default ckpt
python examples/infer_samples.py --gpu 0      # GPU 0
python examples/infer_samples.py path/to/your_ckpt.pt
```

Run on the [OPFData](https://arxiv.org/abs/2406.07234) dataset via `torch_geometric.datasets.OPFDataset` (auto-downloads on first use, then cached):

```bash
python examples/opfdata.py --case pglib_opf_case500_goc --root ~/.cache/opfdata --gpu 0
```

Iterates the full test split in chunks of `--batch-size 128` and reports per-bus / per-generator MAE + cost MAPE vs the OPF-solved ground truth.

### In your own code

Single-graph:

```python
import torch
from gridsfm import load_model, predict

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = load_model("checkpoints/gridsfm_open_v1.1.pt", device=device)
out = predict(model, "samples/case500_goc.pyg.json")
print(out["V"], out["theta"], out["Pg"], out["Qg"], out["feas"])
```

Batched (mixed topology):

```python
import torch
from pathlib import Path
from torch_geometric.utils import unbatch
from gridsfm import load_model, batch_data_list, load_pyg_json, prepare_for_inference

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = load_model("checkpoints/gridsfm_open_v1.1.pt", device=device)

scenario_paths = sorted(Path("samples").glob("*.pyg.json"))
prepared = [prepare_for_inference(load_pyg_json(p)) for p in scenario_paths]
batch = batch_data_list(prepared).to(device)
with torch.no_grad():
    out = model(batch)

# Split per-graph predictions using torch_geometric.utils.unbatch:
V_per_graph  = unbatch(out["bus"].pred[:, 1],       out["bus"].batch)
Pg_per_graph = unbatch(out["generator"].pred[:, 0], out["generator"].batch)
for p, V_i, Pg_i in zip(scenario_paths, V_per_graph, Pg_per_graph):
    print(f"{p.name}: V=[{V_i.min():.3f},{V_i.max():.3f}]  Pg_sum={Pg_i.sum():.2f}")
```

## Output schema

`predict()` is single-graph only; returns a dict of CPU tensors / floats:

| key | shape | units |
|---|---|---|
| `V`          | `[n_bus]`     | voltage magnitude (per-unit) |
| `theta`      | `[n_bus]`     | voltage angle (radians) |
| `Pg`         | `[n_gen]`     | generator active power (per-unit) |
| `Qg`         | `[n_gen]`     | generator reactive power (per-unit) |
| `Pij,Qij,Pji,Qji` | `[n_flow_edges]` | branch flows concatenated across families (per-unit) |
| `flow_edge_types`  | `list[str]` | which families contributed flows, in order |
| `flow_edge_counts` | `list[int]` | row count per family |
| `feas`       | `float`       | scenario feasibility probability `[0,1]` |
| `feas_logit` | `float`       | raw feas-head logit |

Split the flat flows back per-family with `flow_edge_counts`:

```python
out = predict(model, "samples/case500_goc.pyg.json")
n_ac = out["flow_edge_counts"][0] if "ac_line" in out["flow_edge_types"] else 0
Pij_ac = out["Pij"][:n_ac]       # aligns with data[("bus","ac_line","bus")].edge_index
Pij_tr = out["Pij"][n_ac:]       # aligns with data[("bus","transformer","bus")].edge_index
```

For batched inference (`model(batch)` directly), `feas_logit` becomes `[n_graphs]` and `out["bus"].pred` / `out["generator"].pred` are concatenated across graphs.

## Inputs

Two formats:

- **`.pyg.json`** (native) - `{grid: {nodes: {bus, generator, load, shunt}, edges: {ac_line, transformer, generator_link, load_link, shunt_link}}}`. See `samples/case500_goc.pyg.json`.
- **OPFData JSON** (DeepMind release) - same layout; pass `fmt="opfdata"`.

Both loaders require a non-empty `bus` block and matching `senders`/`receivers` lengths per edge type. `load_opfdata` additionally requires EXACT column widths (4 / 11 / 9 / 11 for bus / generator / ac_line / transformer).

Column ordering is defined in `gridsfm/schema.py`:

| node/edge | columns |
|---|---|
| bus | `[base_kV, type, Vmin, Vmax]` |
| generator | `[mbase, _, Pmin, Pmax, _, Qmin, Qmax, Vg, cp2, cp1, cp0]` |
| load | `[Pd, Qd, ...]` |
| shunt | `[Bs, Gs, ...]` |
| ac_line edge_attr | `[angmin, angmax, b_fr, b_to, r, x, rate_a, ...]` |
| transformer edge_attr | `[angmin, angmax, r, x, rate_a, _, _, tap, shift, b_fr, b_to]` |

`prepare_for_inference(data)` mutates the input (attaches PE features, adds `branch_ac` / `branch_tr` / `cycle` node types, widens `bus.x` from 4 to 16 columns). `predict(model, scenario)` deep-copies a user-passed `HeteroData` before mutating.

## Caches

Three per-topology caches keyed by SHA-1 over topology + relevant `edge_attr` bytes, so impedance perturbations (line derating) and topology perturbations (line outages, N-1) both invalidate correctly.

- **`CycleBasisCache`** (`cycle_basis.py`): in-memory LRU **128**, optional disk persistence at `$XDG_CACHE_HOME/gridsfm/cycle_basis/`. Disk cache is schema-versioned.
- **`LaplacianFactorizationCache`** (`pe_features.py`): in-memory LRU **16**. SciPy SuperLU closures, not picklable.
- **`DCPriorCache`** (`dc_prior.py`): in-memory LRU **64**, same picklability constraint. Keyed on topology + `b = 1/x` + slack index.

Defaults are module-level singletons (`DEFAULT_CYCLE_CACHE`, `DEFAULT_LAPLACIAN_CACHE`, plus `model._dc_cache`). To customize sizes or disk paths for large N-1 / N-k sweeps, replace them before first use:

```python
import gridsfm.cycle_basis as cb, gridsfm.pe_features as pe
cb.DEFAULT_CYCLE_CACHE = cb.CycleBasisCache(max_cache=2048, cache_dir="/data/grid_cache")
pe.DEFAULT_LAPLACIAN_CACHE = pe.LaplacianFactorizationCache(max_cache=128)

from gridsfm import load_model
from gridsfm.dc_prior import DCPriorCache
model = load_model("checkpoints/gridsfm_open_v1.1.pt")
model._dc_cache = DCPriorCache(max_cache=512)
```

## Fine-tuning (v1.1 only, OPFData only)

The package ships fine-tuning support: `compute_loss`, `eval_pass`, `finetune_opfdata`, `SyntheticMixedDataset`, `OPFDataAdapterDataset`. See [`examples/finetune_opfdata_case6470.ipynb`](examples/finetune_opfdata_case6470.ipynb) for a few-shot study fine-tuning the v1.1 release on `pglib_opf_case6470_rte`.

**Data format**: fine-tuning is supported on the [OPFData](https://arxiv.org/abs/2406.07234) dataset format only (PyG's `torch_geometric.datasets.OPFDataset`). The synthetic perturbation modes in `SyntheticMixedDataset` and the column-index conventions in `compute_loss` are pinned to OPFData's schema for `bus.x`, `generator.x`, `load.x`, `shunt.x`, and the `ac_line` / `transformer` `edge_attr` / `edge_label` tensors. Custom HeteroData scenarios from other sources are NOT supported on the FT path; use `predict()` / `model(batch)` for inference on the `.pyg.json` schema instead.

```python
from torch_geometric.loader import DataLoader
from gridsfm import (
    load_model, OPFDataAdapterDataset, SyntheticMixedDataset,
    finetune_opfdata,
)
from gridsfm.cycle_basis import CycleBasisCache, prepare_for_grid_transformer_
from gridsfm.pe_features import LaplacianFactorizationCache, attach_pe_features_

model = load_model("checkpoints/gridsfm_open_v1.1.pt", device="cuda:0")

# REQUIRED transform: cycle basis + Hodge PE. Without this the model's
# input projections receive no `cycle` / `branch_*` features and the
# forward silently degrades (HodgePE no-ops, GNN conv → zero output).
# Caches are shared so we factor the Laplacian once per unique topology.
cb_cache = CycleBasisCache()
pe_cache = LaplacianFactorizationCache()

def transform(data):
    prepare_for_grid_transformer_(data, cache=cb_cache)
    attach_pe_features_(data, cache=pe_cache)
    return data

from pathlib import Path
opfdata_root = str(Path("~/.cache/opfdata").expanduser())  # OPFData cache; edit to taste
base = OPFDataAdapterDataset(
    root=opfdata_root, case_name="pglib_opf_case500_goc",
    variant="fulltop", split="train", n_graphs=1000,
)
train_ds = SyntheticMixedDataset(base, infeas_prob=0.3, seed=42, transform=transform)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
log = finetune_opfdata(model, train_loader, epochs=10, lr=1e-4)
```

**Fine-tuning is supported on v1.1 only.** The v1.0 → v1.1 load adapter (`load_model`) is intended for inference. A v1.0 ckpt loaded that way starts with zeroed `W_global` max-pool channels; gradient descent will move them, but the resulting FT'd ckpt is no longer v1.0-shape and must be saved/distributed as v1.1.

**Notebook dependencies**: the example notebook also uses `matplotlib`. Install with `pip install "gridsfm[notebook]"` or just `pip install matplotlib` alongside the core package.

### Quick fine-tune example

[`examples/finetune_opfdata_case6470.ipynb`](examples/finetune_opfdata_case6470.ipynb) walks the full few-shot FT recipe on `pglib_opf_case6470_rte`:

1. **0-shot eval** of the released v1.1 ckpt on the fulltop + n-1 OPFData test splits to establish the starting accuracy.
2. **Three independent FT rounds** with `n_samples = 16 / 104 / 1000` training graphs (multiples of `batch_size=8`), each starting fresh from the release weights, AdamW @ `lr=1e-4`, 10 epochs. Each round wraps the training set in `SyntheticMixedDataset(infeas_prob=0.3)` so ~30% of training graphs are perturbed into infeasible scenarios that train the feasibility classifier alongside the regression heads.
3. **Per-epoch train/val loss curves** plotted side-by-side across the three rounds.
4. **Final eval** of each FT'd model on the held-out fulltop + n-1 test splits, reporting cost MAPE, per-element MAE on θ / V / Pg / Qg, branch flow P/Q MAE, KCL P/Q residual, and thermal loading.
5. **Summary table** comparing 0-shot vs FT n=16 / n=104 / n=1000 across all metrics on both splits.

The whole notebook runs end-to-end on a single GPU in roughly an hour on `case6470_rte` — the n=1000 round dominates at ~3.4 min/epoch × 10 epochs ≈ 35 min, with the rest split across 0-shot eval, the small FT rounds, and the per-round held-out evals. The n=16 round finishes in a couple of minutes and is a useful smoke test that FT is wired up correctly.

## Citation

A reference will be added once the accompanying paper is public. Until then, please cite `microsoft/GridSFM`.
