#!/bin/bash -l
#SBATCH --job-name=basis_match
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/basis_reconstruction_matched_20260914/logs/%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/basis_reconstruction_matched_20260914/logs/%j.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/basis_reconstruction_matched_20260914
OUT=${BASE}/results/basis_reconstruction_matched_20260914
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
CKPT=${BASE}/overlays/residual_split_20260829/pignn_global_GBnetwork_g3_s42_40_best_model.ckpt
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
mkdir -p "$OUT/logs" "$OUT/json"
export PYTHONPATH="$OVERLAY:$BASE:${PYTHONPATH:-}"
cd "$BASE"
srun "$PY" -u "$OVERLAY/basis_reconstruction_matched_controls.py" \
  --parquet "$DATA" --checkpoint "$CKPT" --rank 16 --batch 16 --random-draws 8 \
  --out "$OUT/json/pignn_g3_s42_basis_reconstruction_matched.json"
