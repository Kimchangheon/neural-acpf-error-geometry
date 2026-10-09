#!/bin/bash -l
# Frozen post-training basis-control experiment for GBnetwork PIGNN-GC/G3.
#SBATCH --job-name=g3_basis_ctrl
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/pignn_g3_basis_controls_20260909/logs/%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/pignn_g3_basis_controls_20260909/logs/%j.err

set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/pignn_g3_basis_controls_20260909
OUT=${BASE}/results/pignn_g3_basis_controls_20260909
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
CKPT=${BASE}/overlays/residual_split_20260829/pignn_global_GBnetwork_g3_s42_40_best_model.ckpt
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
mkdir -p "${OUT}/json" "${OUT}/logs"
test -r "${DATA}"; test -r "${CKPT}"; test -r "${OVERLAY}/compare_solution_basis_controls.py"
export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
cd "${BASE}"
srun "${PY}" -u "${OVERLAY}/compare_solution_basis_controls.py" \
  --checkpoint "${CKPT}" --parquet "${DATA}" --rank 16 --batch 16 \
  --random-draws 8 --random-seed 20260909 \
  --out "${OUT}/json/pignn_g3_basis_controls_k16.json"
