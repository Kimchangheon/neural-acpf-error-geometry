#!/bin/bash -l
# Fixed-norm directional diagnostic for GraphKit E120 seed 42 only, matching
# the single-checkpoint Figure-2 protocol used for PIGNN-GC and GridSFM.
#SBATCH --job-name=gk120_fixednorm
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/fixed_norm_directional_graphkit_e120_20260912/logs/%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/fixed_norm_directional_graphkit_e120_20260912/logs/%j.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
O=$BASE/overlays/fixed_norm_directional_graphkit_e120_20260912
R=$BASE/results/fixed_norm_directional_graphkit_e120_20260912
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
GK=$BASE/results/ckpt/gk_e120_20260910/k08_s42_e120_best.pt
mkdir -p "$R/logs" "$R/json"
test -r "$DATA"; test -r "$GK"
export PYTHONPATH="$O:$BASE:${PYTHONPATH:-}" OMP_NUM_THREADS=16
cd "$BASE"
echo "[protocol] GraphKit E120 seed 42; test split; train-only calibration/SVD; k=16; per-scenario fixed magnitude-error norm; calibrated angle fixed"
srun "$PY" -u "$O/controlled_error_geometry.py" --parquet "$DATA" --out "$R/json/fixed_norm_directional_graphkit_e120_s42_k16.json" --phase test --ranks 16 --batch 16 --models graphkit --pignn "$GK" --gridsfm "$GK" --graphkit "$GK" --lumina "$GK"
