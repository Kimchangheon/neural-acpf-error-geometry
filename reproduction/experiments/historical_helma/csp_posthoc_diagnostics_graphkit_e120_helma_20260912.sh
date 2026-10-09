#!/bin/bash -l
# GBnetwork GraphKit E120 CSP post-hoc diagnostic; no training or data creation.
#SBATCH --job-name=gk120_csp_posthoc
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/csp_posthoc_diagnostics_graphkit_e120_20260912/logs/%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/csp_posthoc_diagnostics_graphkit_e120_20260912/logs/%j.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
O=$BASE/overlays/csp_posthoc_diagnostics_graphkit_e120_20260912
R=$BASE/results/csp_posthoc_diagnostics_graphkit_e120_20260912
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
GK=$BASE/results/ckpt/gk_e120_20260910/k08_s42_e120_best.pt
mkdir -p "$R/logs" "$R/json"
test -r "$DATA"; test -r "$GK"
LOCAL_DATA=${TMPDIR:?}/GBnetwork.parquet
cp "$DATA" "$LOCAL_DATA"
test "$(stat -c %s "$DATA")" = "$(stat -c %s "$LOCAL_DATA")"
export PYTHONPATH="$O:$BASE:${PYTHONPATH:-}" OMP_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$O"
echo "[protocol] GraphKit E120 seed 42; seed-42 split; train-only calibration/basis; k=16; complex128 PB"
echo "[identity] pure affine projection is checked separately from restore_known operational CSP"
srun "$PY" -u "$O/csp_posthoc_diagnostics.py" --parquet "$LOCAL_DATA" --graphkit "$GK" --rank 16 --batch 16 --out-json "$R/json/gbnetwork_graphkit_e120_csp_posthoc_k16.json" --out-csv "$R/json/gbnetwork_graphkit_e120_csp_posthoc_k16.csv"
