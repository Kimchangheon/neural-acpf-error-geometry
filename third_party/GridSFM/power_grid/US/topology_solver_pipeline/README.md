# topology_solver_pipeline

Pipeline for converting raw power grid topology JSONs into OPF-solved GridSFM
scenarios. The Makefile commands below are convenience wrappers around the
Julia scripts documented in [PIPELINE_DETAILS.md](PIPELINE_DETAILS.md),
which describes each file, the four pipeline stages, and the data formats
in detail.

Supports two execution modes:

- **Docker** — self-contained image with Julia 1.11, PowerModels.jl, Ipopt, and
  all dependencies pre-installed. No host setup required beyond Docker.
- **Local** — run directly on your machine with a native Julia installation.
  Faster iteration, no container overhead.

Both modes use the same Makefile. Docker targets have no prefix; local targets
are prefixed with `local-`.

## Prerequisites

### For Docker mode

- Docker 20.10+
- GNU Make

### For local mode

- **Julia 1.11+** — install from [julialang.org](https://julialang.org/downloads/)
  or via juliaup:
  ```bash
  curl -fsSL https://install.julialang.org | sh
  ```
  After installation, restart your terminal and verify:
  ```bash
  julia --version
  # Should show: julia version 1.11.x
  ```

- **Python 3** — needed only for the integration test's optional
  `build_hetero_data_from_json` sanity check (skipped gracefully if not available).

- **GNU Make**

### Input data

PowerModels-compatible topology JSON files from the
[GridSFM US Power Grid data release](../../../README.md). The data directory
should contain `04h/` and `16h/` subdirectories with `*_model.json` files.

---

## Quick Start

```bash
cd power_grid/US/topology_solver_pipeline
make help
```

### Docker

```bash
make build
make run STATE=alabama DATA_DIR=/path/to/GridSFM_US_power_grid
```

### Local

```bash
make local-install                    # one-time: download + precompile Julia packages
make local-run STATE=alabama DATA_DIR=/path/to/GridSFM_US_power_grid
```

---

## 1. Setup

### Docker: Build the image

```bash
make build          # cached build
make rebuild        # full rebuild (no cache)
```

### Local: Install Julia packages

```bash
make local-install  # one-time: instantiate from Manifest.toml + precompile
make local-check    # verify Julia and all packages load correctly
```

## 2. Run the Full Pipeline for a State

Runs all stages: patch → solve → export → perturb → verify.

### Docker

```bash
make run STATE=alabama DATA_DIR=/path/to/GridSFM_US_power_grid
make run STATE=montana DATA_DIR=/path/to/GridSFM_US_power_grid HOUR=04h
make run STATE=alabama DATA_DIR=/path/to/GridSFM_US_power_grid N_PER_MODE=5
```

Shortcuts:
```bash
make run-alabama DATA_DIR=...
make run-montana DATA_DIR=...
make run-both DATA_DIR=...
```

### Local

```bash
make local-run STATE=alabama DATA_DIR=/path/to/GridSFM_US_power_grid
make local-run STATE=montana DATA_DIR=/path/to/GridSFM_US_power_grid
```

Shortcuts:
```bash
make local-run-alabama DATA_DIR=...
make local-run-montana DATA_DIR=...
make local-run-both DATA_DIR=...
```

## 3. Run Individual Stages

Each stage can be run independently. Stages that operate only on pipeline
outputs (`export`, `perturb`, `verify`, `verify-one`) do **not** require
`DATA_DIR`.

| Stage | Docker | Local |
|-------|--------|-------|
| Patch model for compatibility | `make patch` | `make local-patch` |
| Solve (raw → solvable) | `make solve` | `make local-solve` |
| Export (solvable → .pyg.json) | `make export` | `make local-export` |
| Perturb (generate scenarios) | `make perturb` | `make local-perturb` |
| Verify all scenarios | `make verify` | `make local-verify` |
| Verify one scenario | `make verify-one` | `make local-verify-one` |

Examples:
```bash
# Docker
make solve STATE=alabama DATA_DIR=...
make export STATE=alabama                    # no DATA_DIR needed
make perturb STATE=alabama N_PER_MODE=5
make verify STATE=alabama
make verify-one STATE=alabama PYG_FILE=alabama_16h/scenarios/alabama_model/costs_0001.pyg.json

# Local
make local-solve STATE=alabama DATA_DIR=...
make local-export STATE=alabama
make local-perturb STATE=alabama N_PER_MODE=5
make local-verify STATE=alabama
make local-verify-one STATE=alabama PYG_FILE=alabama_16h/scenarios/alabama_model/costs_0001.pyg.json
```

## 4. Bulk Multi-Grid Scenario Generation

Generate scenarios for multiple grids in one run. Create a grid list file with
one `<solvable.json> <n_per_mode>` per line:

```bash
cat > grids.txt <<EOF
/output/alabama_16h/alabama_model.solvable.json 400
/output/montana_16h/montana_model.solvable.json 400
EOF
```

**Docker** (paths in the grid list are container paths under `/output`):
```bash
make gen-bulk GRID_LIST=./grids.txt N_WORKERS=8
```

**Local** (paths in the grid list are host filesystem paths):
```bash
cat > grids.txt <<EOF
$(pwd)/output/alabama_16h/alabama_model.solvable.json 400
$(pwd)/output/montana_16h/montana_model.solvable.json 400
EOF
make local-gen-bulk GRID_LIST=./grids.txt N_WORKERS=8
```

## 5. Integration Test

Full end-to-end smoke test: solve → export → re-solve → perturb → round-trip
verify with objective consistency checks.

```bash
# Docker
make integration-test STATE=alabama DATA_DIR=...

# Local
make local-integration-test STATE=alabama DATA_DIR=...
```

## 6. Interactive Access

```bash
# Docker: bash shell
make shell DATA_DIR=...

# Docker: Julia REPL
make julia-repl DATA_DIR=...

# Local: Julia REPL with project environment
make local-julia-repl
```

## 7. Inspect Results

These targets work the same for both Docker and local:

```bash
make list-states DATA_DIR=...    # list available states in data directory
make list-output                 # list all output artifacts
make output-size                 # show output directory sizes
```

## 8. Cleanup

```bash
make clean-state STATE=alabama HOUR=16h   # remove output for one state
make clean                                # remove all output
make clean-image                          # remove Docker image
make clean-all                            # remove output + Docker image
```

---

## Configuration Variables

All variables can be overridden on the command line:

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE` | `topo_solver_pipe` | Docker image name |
| `DATA_DIR` | *(required for solve/run)* | Path to `GridSFM_US_power_grid/` directory |
| `OUTPUT_DIR` | `./output` | Host directory for output artifacts |
| `STATE` | `alabama` | State name (e.g., `montana`, `texas`) |
| `HOUR` | `16h` | Operating hour (`16h` peak or `04h` off-peak) |
| `N_PER_MODE` | `1` | Scenarios per perturbation mode |
| `N_WORKERS` | `2` | Julia worker processes for scenario generation |
| `CPU_RANGE` | *(auto: `0-(N-1)` where N = number of logical CPUs)* | CPU core range for `taskset` pinning in `local-gen-bulk` (e.g. `0-7`). Linux-only; ignored on macOS (no `taskset`). |
| `GRID_LIST` | *(required for gen-bulk)* | Grid list file for bulk generation |
| `PYG_FILE` | *(required for verify-one)* | Scenario file path relative to `OUTPUT_DIR` |

Example:
```bash
make local-run STATE=texas DATA_DIR=/mnt/data/GridSFM_US_power_grid OUTPUT_DIR=/mnt/results N_PER_MODE=10
```

---

## Makefile → Julia Script Mapping

Every Makefile target is a convenience wrapper around a Julia script
documented in [PIPELINE_DETAILS.md](PIPELINE_DETAILS.md):

| Julia script / command | Docker target | Local target |
|---|---|---|
| `Pkg.instantiate()` | `make build` | `make local-install` |
| `solve_topo_json.jl` | `make solve` | `make local-solve` |
| `export_gridsfm_data.jl` | `make export` | `make local-export` |
| `gen_perturbed_data.jl` | `make perturb` | `make local-perturb` |
| `run_gen_gridsfm_data.sh` | `make gen-bulk` | `make local-gen-bulk` |
| `solve_pyg_json.jl` (one) | `make verify-one` | `make local-verify-one` |
| `solve_pyg_json.jl` (all) | `make verify` | `make local-verify` |
| `integration_test_all_components.sh` | `make integration-test` | `make local-integration-test` |

---

## Pipeline Stages Explained

```
 Stage 2a               Stage 2b               Stage 3
 ────────               ────────               ───────
 solve_topo_json.jl     export_gridsfm_data.jl gen_perturbed_data.jl
 ─────────────────      ────────────────────── ──────────────────────
 raw topology JSON  ──► .solvable.json     ──► .pyg.json files
 (possibly not          (cold-strict            (per-mode perturbed
  AC-OPF solvable)       AC-OPF solvable)       scenarios + base)
```

### Perturbation Modes (Stage 3)

| Mode       | Description |
|------------|-------------|
| `loads`    | Scale system load by factor ∈ [0.8, 1.5] with ±10% per-load jitter |
| `costs`    | Shuffle cost coefficients among ~40% of active generators |
| `killgen`  | Deactivate 1-3 generators (preserving ≥2 active) |
| `derate`   | Scale branch ratings on ~10% of branches by [0.7, 0.95] |
| `vsqueeze` | Shrink voltage bands on ~10% of buses by up to 0.01 pu |

### Relaxation Levels (Stage 2a)

| Level | Name | Description |
|-------|------|-------------|
| L0 | Strict | Model as-is |
| AC1 | V + Q relax | Voltage [0.90,1.10], Q limits ×1.5 |
| L1 | Widen angles | Branch angles ±60° |
| L2 | Thermal headroom | Ratings ×1.2, angles ±60° |
| L3 | Aggressive | Ratings ×1.5, angles ±90°, pmin ×0.5 |
| L4 | Load shedding | Load capped 70%, ratings ×1.5, angles ±90°, pmin=0 |
| L5 | Full relaxation | No thermal limits, V [0.85,1.15], Q ×2.0 |

## Output Structure

After running the full pipeline for a state:

```
output/alabama_16h/
├── alabama_model.solvable.json         # Cold-strict solvable topology
├── alabama_model.pyg.json              # Base gridSFM export
├── grids_solvable.txt                  # Grid list used for scenario generation
└── scenarios/
    └── alabama_model/
        ├── base_unperturbed.pyg.json   # Unperturbed base case
        ├── loads_0001.pyg.json         # Load perturbation scenario
        ├── costs_0001.pyg.json         # Cost perturbation scenario
        ├── killgen_0001.pyg.json       # Generator trip scenario
        ├── derate_0001.pyg.json        # Branch derating scenario
        └── vsqueeze_0001.pyg.json      # Voltage squeeze scenario
```

## Troubleshooting

### Julia not found (local mode)
Ensure Julia 1.11+ is on your PATH:
```bash
julia --version
```
If not installed, see [julialang.org/downloads](https://julialang.org/downloads/).

### Package instantiation fails (local mode)
The `Manifest.toml` pins exact versions for Julia 1.11. If you have a different
Julia version, you may need to resolve:
```bash
cd power_grid/US/topology_solver_pipeline
julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
```

### Build fails on package instantiation (Docker)
Ensure Docker has internet access for downloading packages from the Julia
package registry.

### Solver is slow
Large state models (e.g., Texas, California) may take 10+ minutes per
relaxation level. First runs in local mode include Julia compilation time
(~30-60s). Subsequent runs are faster.

### Permission errors on output (Docker)
Docker writes files as root. The `clean` / `clean-state` targets handle this
automatically by using Docker to remove root-owned files. If you need manual
cleanup:
```bash
docker run --rm -v $(pwd)/output:/output topo_solver_pipe -c "rm -rf /output/*"
```
