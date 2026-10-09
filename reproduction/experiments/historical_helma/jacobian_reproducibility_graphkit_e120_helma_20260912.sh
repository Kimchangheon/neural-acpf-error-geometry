#!/bin/bash -l
# Frozen 128 x 32 Jacobian-gain reproduction plus GraphKit E120 actual-error JVPs.
#SBATCH --job-name=gk120_jac_audit
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/jacobian_graphkit_e120_20260912/logs/%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/jacobian_graphkit_e120_20260912/logs/%j.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
O=$BASE/overlays/jacobian_graphkit_e120_20260912
R=$BASE/results/jacobian_graphkit_e120_20260912
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
GK=$BASE/results/ckpt/gk_e120_20260910/k08_s42_e120_best.pt
mkdir -p "$R/logs" "$R/json"
test -r "$DATA"; test -r "$GK"
LOCAL_DATA=${TMPDIR:?}/GBnetwork.parquet
cp "$DATA" "$LOCAL_DATA"
test "$(stat -c %s "$DATA")" = "$(stat -c %s "$LOCAL_DATA")"
export PYTHONPATH="$O:$BASE:${PYTHONPATH:-}" OMP_NUM_THREADS=24 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$O"
echo "[protocol] frozen rank=16; 128 held-out seed-42-test scenarios; 32 Gaussian directions/scenario; RNG=20260909"
echo "[checkpoint] $GK (GraphKit E120 seed 42; sd=0.02 / flat-head family)"
srun "$PY" -u "$O/jacobian_subspace_reproducibility.py" --parquet "$LOCAL_DATA" --graphkit "$GK" --rank 16 --scenarios 128 --directions 32 --seed 20260909 --basis-batch 16 --out-json "$R/json/jacobian_graphkit_e120_s42_k16.json" --out-csv "$R/json/jacobian_graphkit_e120_s42_k16.csv" --out-npz "$R/json/jacobian_random_gains_k16.npz"
