#!/bin/bash -l
# Full CPU NR timing from one already-complete GPU warm-start cache.
# A 128-core allocation is intentional: one BLAS thread per process and all
# CPU cores granted by the H100 node are used for each solver/start condition.
#SBATCH --job-name=nr_solve_gb
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr_warmstart_gbnetwork_20260912/logs/solve_%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr_warmstart_gbnetwork_20260912/logs/solve_%A_%a.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OUT=$BASE/results/nr_warmstart_gbnetwork_20260912
SPEC=$BASE/nr_warmstart_specs_20260912.tsv
mkdir -p "$OUT/logs" "$OUT/nr"
mapfile -t ROWS < <(grep -v '^#' "$SPEC")
IFS=$'\t' read -r LABEL KIND RELCKPT BATCH <<< "${ROWS[$SLURM_ARRAY_TASK_ID]}"
CACHE=$OUT/cache/$LABEL
test -r "$CACHE/metadata.json"; test -r "$CACHE/raw_restored.npy"; test -r "$CACHE/csp16_restored.npy"
export PYTHONPATH=$BASE:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$BASE"
for SOLVER in custom pandapower_core; do
  for START in cold raw csp16; do
    srun python3.9 -u benchmark_nr_from_cached_starts.py --cache "$CACHE" \
      --start "$START" --solver "$SOLVER" --workers 128 --chunksize 1 \
      --tolerance-pu 1e-8 --max-iterations 40 \
      --out "$OUT/nr/${LABEL}_${SOLVER}_${START}.json"
  done
done
