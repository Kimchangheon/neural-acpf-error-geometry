#!/usr/bin/env bash
#SBATCH --job-name=gb_e2e_nr32
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --array=0-10
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr_warmstart_gbnetwork_20260912/logs/e2e_%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr_warmstart_gbnetwork_20260912/logs/e2e_%A_%a.err

# One uninterrupted job per checkpoint: GPU output generation -> prescribed
# coordinate restoration -> local node cache -> 32-core CPU NR.  The cache is
# local scratch, not a reused prior result, so this measures an actual pipeline.
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY="$BASE/overlays/gbnetwork_inference_walltime_h100_20260912"
RESULT="$BASE/results/nr_warmstart_gbnetwork_20260912/e2e32"
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
GPU_PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
CPU_PY=python3.9
SPEC="$BASE/nr_warmstart_e2e_specs_safe_20260913.tsv"

mapfile -t ROWS < <(grep -v '^#' "$SPEC")
IFS=$'\t' read -r LABEL KIND RELCKPT BATCH <<< "${ROWS[$SLURM_ARRAY_TASK_ID]}"
CKPT="$BASE/$RELCKPT"
test -r "$DATA"; test -r "$CKPT"
mkdir -p "$RESULT" "$BASE/results/nr_warmstart_gbnetwork_20260912/logs"

TMP_ROOT="${TMPDIR:-/tmp}/gb_e2e_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMP_ROOT"
cleanup() { rm -rf "$TMP_ROOT"; }
trap cleanup EXIT

LUM_ARGS=()
if [[ "$KIND" == "lumina" ]]; then
  LUM_ARGS=(--lumina-args "--model_config $BASE/lumina_ckpt/lumina_config_do0.json --mask_known_v --bus_physics_features --theta_anchor start --init_recipe sd002_zerohead")
fi
export PYTHONPATH="$OVERLAY:$BASE:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

JOB_START_NS=$(date +%s%N)
cd "$OVERLAY"
"$GPU_PY" -u "$OVERLAY/cache_nr_warmstarts.py" \
  --model "$KIND" --checkpoint "$CKPT" --parquet "$DATA" \
  --out "$TMP_ROOT/cache" --batch "$BATCH" --rank 16 "${LUM_ARGS[@]}"
CACHE_END_NS=$(date +%s%N)

cd "$BASE"
for SOLVER in custom pandapower_core; do
  for START in raw csp16; do
    "$CPU_PY" -u benchmark_nr_from_cached_starts.py \
      --cache "$TMP_ROOT/cache" --start "$START" --solver "$SOLVER" \
      --workers 32 --chunksize 1 --tolerance-pu 1e-8 --max-iterations 40 \
      --out "$TMP_ROOT/${SOLVER}_${START}.json"
  done
done
JOB_END_NS=$(date +%s%N)

export E2E_LABEL="$LABEL" E2E_KIND="$KIND" E2E_CHECKPOINT="$CKPT"
export E2E_CACHE="$TMP_ROOT/cache/metadata.json" E2E_WORK="$TMP_ROOT"
export E2E_JOB_START_NS="$JOB_START_NS" E2E_CACHE_END_NS="$CACHE_END_NS" \
  E2E_JOB_END_NS="$JOB_END_NS" E2E_RESULT="$RESULT/${LABEL}.json"
"$CPU_PY" - <<'PY'
import json, os
from pathlib import Path
cache = json.loads(Path(os.environ['E2E_CACHE']).read_text())
root = Path(os.environ['E2E_WORK'])
online = float(cache['cache_wall_seconds'])
rows = {}
for solver in ('custom', 'pandapower_core'):
    for start in ('raw', 'csp16'):
        r = json.loads((root / f'{solver}_{start}.json').read_text())
        # Deployable end-to-end: pre-fitted calibration/PCA artifacts are
        # assumed loaded; cache_wall is the held-out output/restore/materialize
        # stage.  The enclosing job wall also records the conservative path
        # which refits those train-only artifacts for audit reproducibility.
        r['e2e_seconds_online_artifacts_prefit'] = online + r['end_to_end_seconds_excluding_gpu_cache']
        rows[f'{solver}_{start}'] = r
out = {
    'label': os.environ['E2E_LABEL'], 'kind': os.environ['E2E_KIND'],
    'checkpoint': os.environ['E2E_CHECKPOINT'], 'workers': 32,
    'test_scenarios': cache['test_scenarios'], 'cache_metadata': cache,
    'online_inference_restore_materialize_seconds': online,
    'cache_process_wall_with_train_stat_refit_seconds': (int(os.environ['E2E_CACHE_END_NS']) - int(os.environ['E2E_JOB_START_NS'])) / 1e9,
    'full_job_wall_all_four_solver_start_pairs_seconds': (int(os.environ['E2E_JOB_END_NS']) - int(os.environ['E2E_JOB_START_NS'])) / 1e9,
    'nr': rows,
}
Path(os.environ['E2E_RESULT']).write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
PY
