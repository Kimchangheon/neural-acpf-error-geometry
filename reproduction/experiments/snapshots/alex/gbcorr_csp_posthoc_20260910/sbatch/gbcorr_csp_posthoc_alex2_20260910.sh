#!/bin/bash -l
# Frozen G3-s42 GBcorr calibration x basis CSP audit; no training/data creation.
#SBATCH --job-name=gbcorr_csp_audit
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/gbcorr_csp_posthoc_20260910/logs/%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/gbcorr_csp_posthoc_20260910/logs/%j.err
set -euo pipefail
BASE=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
O=$BASE/overlays/gbcorr_csp_posthoc_20260910
R=$BASE/results/gbcorr_csp_posthoc_20260910
PY=/home/hpc/iwi5/iwi5295h/conda-envs/gridfm-py312/bin/python
CONTROL=$R/data/control.parquet
GBCORR=$R/data/gbcorr.parquet
CKPT=$R/checkpoints/pignn_global_GBnetwork_g3_s42_40_best_model.ckpt
mkdir -p "$R/json" "$R/logs"
test -r "$CONTROL"; test -r "$GBCORR"; test -r "$CKPT"
export PYTHONPATH=$O:$BASE:${PYTHONPATH:-}
export OMP_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$O"
srun "$PY" -u "$O/gbcorr_csp_posthoc.py" --control "$CONTROL" --gbcorr "$GBCORR" --checkpoint "$CKPT" --rank 16 --batch 8 --out-json "$R/json/gbcorr_csp_posthoc_k16.json" --out-csv "$R/json/gbcorr_csp_posthoc_k16.csv"
