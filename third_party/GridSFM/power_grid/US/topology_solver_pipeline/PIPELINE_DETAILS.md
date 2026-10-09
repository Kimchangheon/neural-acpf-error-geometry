# GridSFM data pipeline — from raw topology to solved scenarios

This document provides detailed documentation of each file in the package,
the four pipeline stages, and the data formats they produce. For setup and
high-level commands, see the [README](README.md).

The full pipeline that turns a raw grid topology into GridSFM-capable
`.pyg.json` input files has **four distinct stages**. They are owned by
separate components, and it is important to keep them straight:

```
 stage 1                stage 2                stage 3                stage 4
 ───────                ───────                ───────                ───────

 gridsfm_topo           topology_solver_pipeline  scenario generator     GridSFM
 (upstream package)     (this directory)       (this directory)       (downstream)
 ──────────────         ─────────────────      ──────────────────     ─────────
 build PowerModels ──►  apply the minimum ──►  perturb the solvable ──►  train /
 topology JSONs         parameter             base grid and solve       evaluate on
 from raw grid          relaxations needed    strict AC-OPF to emit     the resulting
 sources (OSM,          to make the topo      one .pyg.json per         .pyg.json
 utility feeds,         AC-OPF-solvable       scenario (5 modes ×       files
 open datasets)         cold-strict           N + 1 unperturbed)

 output:                output:                output:                consumes:
 raw topology           .solvable.json         per-grid folder of     .pyg.json via
 JSON (not always       (canonical, cold-     .pyg.json files with    build_hetero_
 AC-OPF-solvable)       strict solvable)      AC-OPF solution +       data_from_json
                                              duals + perturbation
                                              metadata
```

**Important — what `topology_solver_pipeline` actually is**: stage 2 only. Its job
is narrow: take one raw topology JSON and produce one cold-strict solvable
version (`.solvable.json`). It does **not** generate scenarios, and it does
**not** write GridSFM's `.pyg.json` schema. That is stage 3.

Stage 3 — the scenario generator — is currently housed in the same
directory for convenience (it needs the `.solvable.json` from stage 2 as
input and lives right next to it), but it is a logically separate
component: it consumes `.solvable.json` files and emits `.pyg.json` files
by applying controlled perturbations and solving strict AC-OPF. The files
belonging to each stage are clearly delineated below.

---

## Files, by stage

### Stage 2 — `topology_solver_pipeline` proper (raw topo  →  .solvable.json)

**`run_opf_relaxation.jl`** — the relaxation engine. Holds the logic for
running AC / DC / SOC OPF with optional escalation across relaxation
**levels** (L0 means "solve as-is", L1..L5 progressively widen thermal
ratings / Q limits / voltage bounds / Pmin / load cap, and AC1 injects
DC-derived shunt compensation). Also handles generator de-commitment,
impedance-consistency fixes, and bounded shunt injection. It is never run
directly in this pipeline; it's included by the driver below.

**`solve_topo_json.jl`** — the public entry point for stage 2. Given a raw
topology JSON, it iterates **L0 → AC1 → L1 → L2 → L3 → L4 → L5**, writes the
mutated data after each level to a tmp file, and verifies cold-strict
solvability by reloading that tmp file, zeroing every warm-start field, and
running strict AC-OPF. First level that converges wins and is written to
`<input>.solvable.json`.

```bash
julia --project=. solve_topo_json.jl <input.json> <output.solvable.json>
```

That is the full extent of stage 2. The output `.solvable.json` is a plain
PowerModels-native JSON — any downstream tool that uses PowerModels can
load it directly.

### Stage 3 — scenario generator (.solvable.json  →  .pyg.json files)

This stage happens to live in the same directory, but it is a separate
concern: it reads `.solvable.json` base grids from stage 2 and writes
GridSFM-schema `.pyg.json` files.

**`export_gridsfm_data.jl`** — solves one `.solvable.json` with strict AC-OPF
and writes one `.pyg.json` in the gridSFM schema (grid topology +
AC-OPF solution + duals + metadata). Useful for inspecting a single base
case or building a tiny reference set. The schema itself is documented in
the file header. Its `build_gridsfm_data` function is the single source of
truth for that schema and is reused by the bulk generator below.

```bash
julia --project=. export_gridsfm_data.jl <input.solvable.json> <output.pyg.json>
```

**`gen_perturbed_data.jl`** — the bulk scenario generator. Reads a list file
naming one or more `.solvable.json` base grids and how many scenarios per
mode to produce for each, then spawns worker processes via
`Distributed.pmap` and generates, in parallel across the whole pool:

- one **`base_unperturbed.pyg.json`** per grid (the solvable topology as-is),
  plus
- **N scenarios per mode × 5 modes** per grid, where each scenario applies
  exactly one of the five pure perturbation modes:

  | Mode       | What it does                                                                                   |
  |------------|------------------------------------------------------------------------------------------------|
  | `loads`    | Pick `sf ∈ [0.8, 1.5]`, then `pd,qd *= sf · (0.9 + 0.2·rand)` (±10% per-load jitter)           |
  | `costs`    | Shuffle cost coefficients among ~40% of active gens, within same-`ncost` pools                 |
  | `killgen`  | Flip `gen_status=0` on 1 / 2 / 3 gens (probabilities 0.7 / 0.2 / 0.1), preserving ≥2 active    |
  | `derate`   | On ~10% of active branches, scale `rate_a/b/c` by a uniform factor ∈ [0.7, 0.95]               |
  | `vsqueeze` | On ~10% of buses, `vmin += rand·0.01` and `vmax -= rand·0.01` (per-boundary, independent)      |

Each scenario applies one perturbation, solves strict AC-OPF, and exports a
`.pyg.json` tagged by `perturb_mode` in metadata. The pure-mode split — as
opposed to random combinations of all five — keeps the per-mode signal
uncorrelated. Output layout:

```
<out_root>/<case>/base_unperturbed.pyg.json
<out_root>/<case>/loads_0001.pyg.json     ... loads_NNNN.pyg.json
<out_root>/<case>/costs_0001.pyg.json     ... costs_NNNN.pyg.json
<out_root>/<case>/killgen_0001.pyg.json   ... killgen_NNNN.pyg.json
<out_root>/<case>/derate_0001.pyg.json    ... derate_NNNN.pyg.json
<out_root>/<case>/vsqueeze_0001.pyg.json  ... vsqueeze_NNNN.pyg.json
```

Grid-list format (one line per base grid, `#` for comments):

```
# <solvable_json_path>  <n_per_mode>
<solvable_out_dir>/alabama.solvable.json  400
<solvable_out_dir>/montana.solvable.json  400
```

Total scenarios per grid = **1 + 5 × n_per_mode**. The pmap is global across
all listed grids so workers never idle between cases, and the generator is
resume-friendly — existing `.pyg.json` files are skipped on restart.

**`run_gen_gridsfm_data.sh`** — convenience driver for
`gen_perturbed_data.jl`. Every tunable is a positional CLI arg with a
default; leave later args unset to keep their defaults.

```bash
bash run_gen_gridsfm_data.sh [n_proc] [out_root] [grid_list] [cpu_range]
```

| Arg         | Default                              | Meaning                                                    |
|-------------|--------------------------------------|------------------------------------------------------------|
| `n_proc`    | `51`                                 | worker processes                                           |
| `out_root`  | `./out`        | per-grid output folders land here                          |
| `grid_list` | `$SCRIPT_DIR/grids_solvable.txt`     | one line per grid: `<solvable.json_path>  <n_per_mode>`    |
| `cpu_range` | `77-127`                             | `taskset -c` mask (51 cores by default)                    |

### Stage-3 verification — reconstruct-and-resolve a `.pyg.json`

**`solve_pyg_json.jl`** — answers the question *"is this `.pyg.json` a
usable input?"* by reconstructing the exact PowerModels data the scenario
represents, solving strict AC-OPF, and comparing the result against
`metadata.objective`.

Given a solvable base + a perturbed scenario, it starts from the
`.solvable.json` (which has every field PowerModels needs) and overlays
the perturbed values from the `.pyg.json`:

| Perturbation | Source in `.pyg.json`                                  |
|-------------|---------------------------------------------------------|
| loads       | `grid.nodes.load[:, 1:2]` → `pd`, `qd`                  |
| killgen     | gens absent from `grid.nodes.generator` → `gen_status=0`|
| derate      | `grid.edges.{ac_line,transformer}.features` → rate_a/b/c |
| vsqueeze    | `grid.nodes.bus[:, 3:4]` → `vmin`, `vmax`               |
| costs       | `grid.nodes.generator[:, 9:11]` → `cp2`, `cp1`, `cp0`   |

Warm-start (`va/vm/pg/qg`) comes from `solution.nodes.{bus,generator}`.

```bash
julia --project=. solve_pyg_json.jl <solvable.json> <scenario.pyg.json>
```

Prints `RESOLVE ok obj=<x> expected=<y> Δ=<pct>%` and exits 0 iff the
re-solve converges AND the objective matches within 0.1%; exits 1
otherwise. This is the true integrity check for stage-3 outputs — if it
passes, the `.pyg.json` contained enough information to exactly reproduce
the solve that produced it.

### End-to-end integration test (crosses stages 2 + 3)

**`integration_test_all_components.sh`** — full-pipeline smoke test.
A bash script that orchestrates the four Julia CLI binaries in this
directory; shell is the natural fit for multi-process orchestration.

1. **Stage 2 — solve**: `solve_topo_json.jl` → `raw.solvable.json`.
2. **Stage 3 — export**: `export_gridsfm_data.jl` → `raw.pyg.json`.
3. **Stage 2 — resolve**: strict AC-OPF on the solvable; objective must
   match step 2 within 0.01%.
4. **Stage 3 — perturb+resolve**: `gen_perturbed_data.jl` with
   `n_per_mode=1`, then `solve_pyg_json.jl` on each of the six outputs
   (`base_unperturbed` + one per mode). Each scenario must re-solve with
   Δ < 0.1% vs its recorded `metadata.objective`.

A Python sanity step (skipped gracefully if `gridfm` isn't importable on
the host's `python3` / `$GRIDFM_PYTHON`) confirms
`build_hetero_data_from_json` parses the `.pyg.json` cleanly.

```bash
bash integration_test_all_components.sh <raw_input.json> [out_dir=/tmp/topo_solver_pipe_test]
```

Exits 0 on all checks green, non-zero otherwise. Use it as the single CI
gate for changes to any file in this directory.

---

## End-to-end example

This directory is **self-contained**: its own `Project.toml` + `Manifest.toml`
live alongside the scripts, so every Julia invocation uses `--project=.`
and no parent-project lookup is needed. On first run, Julia will
instantiate the environment (package download + precompile) from the
pinned Manifest:

```bash
cd <wherever you put topology_solver_pipeline/>
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Assuming `<topo_data_path>` is wherever `gridsfm_topo` wrote its raw
topology JSONs and `<solvable_out_dir>` is wherever you want the stage-2
outputs to land:

```bash
# 1. Make a raw topology cold-strict solvable.
#    <topo_data_path>/alabama.json        — raw topology JSON (input)
#    <solvable_out_dir>/alabama.solvable.json — solvable output (stage 2)
julia --project=. solve_topo_json.jl \
    <topo_data_path>/alabama.json \
    <solvable_out_dir>/alabama.solvable.json

# 2. Generate GridSFM-capable .pyg.json files from solvable grids.
#    grids_solvable.txt — list file: each line is "<solvable.json path>  <n_per_mode>"
#    n_per_mode (400)   — number of perturbed scenarios per perturbation mode
#    run_gen_gridsfm_data.sh args: <n_workers> <out_root>
#      51   — number of parallel Julia worker processes
#      ./out — output directory (one subfolder per grid)
cat > grids_solvable.txt <<EOF
<solvable_out_dir>/alabama.solvable.json  400
<solvable_out_dir>/montana.solvable.json  400
EOF
bash run_gen_gridsfm_data.sh 51 ./out

# 3. Smoke-test the full pipeline end-to-end on one raw file.
#    Runs stages 2 + 3 + verification in sequence; exits 0 if all checks pass.
bash integration_test_all_components.sh <topo_data_path>/alabama.json

# 4. Verify one specific perturbed scenario stands on its own.
#    Re-solves the scenario from its .solvable.json base and checks that
#    the objective matches the recorded value within 0.1%.
julia --project=. solve_pyg_json.jl \
    <solvable_out_dir>/alabama.solvable.json \
    ./out/alabama/loads_0001.pyg.json
```

The resulting `<out_root>/<case>/*.pyg.json` files drop straight into the
existing GridSFM data-loading path — no schema adapter needed, same layout
as the rest of the GridSFM-capable inputs.
