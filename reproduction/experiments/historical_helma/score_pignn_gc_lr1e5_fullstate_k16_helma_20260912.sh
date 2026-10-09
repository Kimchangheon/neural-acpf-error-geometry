#!/bin/bash -l
# Test-only table scoring for the three G3/PIGNN-GC lr=1e-5 seeds.
# This job deliberately waits for the training jobs and then reads their
# validation-selected best checkpoints.  It performs no optimisation.
# Projection is the manuscript's affine full-state P16/CSP16 policy: no
# prescribed-variable restoration and no unknown-coordinate restriction.
#SBATCH --job-name=g3_lr1e5_full16
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/pignn_gc_lr1e5_fullstate_csp_k16_20260912/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/pignn_gc_lr1e5_fullstate_csp_k16_20260912/logs/%A_%a.err

set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/pignn_gc_lr1e5_fullstate_csp_k16_20260912
OUT=${BASE}/results/pignn_gc_lr1e5_fullstate_csp_k16_20260912
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python

seeds=(41 42 43)
seed=${seeds[${SLURM_ARRAY_TASK_ID:?}]}
CKPT=${BASE}/results/pignn_gc_e120_lr1e5_20260911/ckpt/pignn_gc_GBnetwork_s${seed}_e120_lr1e5_120_best_model.ckpt

mkdir -p "${OUT}/json" "${OUT}/logs"
test -r "${DATA}"; test -r "${CKPT}"; test -r "${OVERLAY}/output_intervention_metrics.py"
export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${BASE}"
echo "[checkpoint] PIGNN-GC G3 lr=1e-5 seed=${seed}: ${CKPT}"
echo "[protocol] fixed seed-42 split; train-only calibration and |V|/angle bases; k=16"
echo "[PB] final complex128 voltage, GENCO structural-zero PB; full-state projection only"
srun "${PY}" -u "${OVERLAY}/output_intervention_metrics.py" \
  --model g3 --checkpoint "${CKPT}" --parquet "${DATA}" \
  --rank 16 --batch 16 --projection-variants full \
  --out "${OUT}/json/pignn_gc_lr1e5_s${seed}_fullstate_csp_k16.json"
