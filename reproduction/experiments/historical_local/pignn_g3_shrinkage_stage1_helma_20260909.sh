#!/bin/bash -l
# Validation-only scalar-shrinkage lambda selection.  Does not score test.
#SBATCH --job-name=g3_shrink_val
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/pignn_g3_shrinkage_20260909/logs/%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/pignn_g3_shrinkage_20260909/logs/%j.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/pignn_g3_shrinkage_20260909
OUT=${BASE}/results/pignn_g3_shrinkage_20260909
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
CKPT=${BASE}/overlays/residual_split_20260829/pignn_global_GBnetwork_g3_s42_40_best_model.ckpt
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
mkdir -p "${OUT}/json" "${OUT}/logs"
test -r "${DATA}"; test -r "${CKPT}"; test -r "${OVERLAY}/evaluate_scalar_shrinkage.py"
export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
cd "${BASE}"
srun "${PY}" -u "${OVERLAY}/evaluate_scalar_shrinkage.py" --stage validation \
 --checkpoint "${CKPT}" --parquet "${DATA}" --batch 16 \
 --out "${OUT}/json/validation.json"
