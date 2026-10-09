#!/bin/bash -l
# Immediate snapshot scoring while the source training jobs continue.
# The checkpoint copies were made before this job was submitted.
#SBATCH --job-name=g3_lr1e5_snap16
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/pignn_gc_lr1e5_provisional_fullstate_csp_k16_20260912/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/pignn_gc_lr1e5_provisional_fullstate_csp_k16_20260912/logs/%A_%a.err

set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/pignn_gc_lr1e5_fullstate_csp_k16_20260912
OUT=${BASE}/results/pignn_gc_lr1e5_provisional_fullstate_csp_k16_20260912
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
seeds=(41 42 43)
seed=${seeds[${SLURM_ARRAY_TASK_ID:?}]}
CKPT=${OUT}/ckpt/pignn_gc_GBnetwork_s${seed}_best_snapshot.ckpt
mkdir -p "${OUT}/json" "${OUT}/logs"
test -r "${DATA}"; test -r "${CKPT}"; test -r "${OVERLAY}/output_intervention_metrics.py"
export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${BASE}"
echo "[snapshot checkpoint] PIGNN-GC G3 seed=${seed}: ${CKPT}"
echo "[protocol] seed-42 split; train-only calibration/bases; k=16; full-state only"
srun "${PY}" -u "${OVERLAY}/output_intervention_metrics.py" \
  --model g3 --checkpoint "${CKPT}" --parquet "${DATA}" \
  --rank 16 --batch 16 --projection-variants full \
  --out "${OUT}/json/pignn_gc_lr1e5_s${seed}_snapshot_fullstate_csp_k16.json"
