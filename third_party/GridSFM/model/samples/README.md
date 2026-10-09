# Sample scenarios

53 unperturbed AC-OPF scenarios in the `.pyg.json` schema. Each file holds the raw topology + load + cost data plus a `solution`
block containing the AC-OPF dispatch produced by the data-generation
solver, and is suitable for both inference (`predict()` /
`infer_samples.py`) and ground-truth comparison.

| count | prefix | source |
|---|---|---|
| 23 | `case*`  | [pglib-opf](https://github.com/power-grid-lib/pglib-opf) base cases (`pglib_opf_<name>.m`), unperturbed |
| 30 | `msr_*`  | [`microsoft/GridSFM_US_power_grid`](https://huggingface.co/datasets/microsoft/GridSFM_US_power_grid) 16h peak-demand snapshots, unperturbed |

The MSR samples are derived from open data (OpenStreetMap topology +
state-level demand); see the
[`microsoft/gridsfm`](https://huggingface.co/collections/microsoft/gridsfm)
Hugging Face collection and the [`power_grid/`](https://github.com/microsoft/GridSFM/tree/main/power_grid)
pipeline for how they are built and how to download the full
multi-snapshot dataset.

The samples ship as base cases only — none of the perturbations the model
was trained against (load scaling, cost shuffling, generator outages,
line derating, voltage squeezing) are applied here. To run inference on
perturbed scenarios, use the topology-solver pipeline in
[`power_grid/US/topology_solver_pipeline/`](https://github.com/microsoft/GridSFM/tree/main/power_grid/US/topology_solver_pipeline)
to emit `.pyg.json` per scenario.
