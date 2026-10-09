#!/bin/bash -l
#SBATCH --job-name=nr_probe_gsfm_mask
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr_warmstart_gbnetwork_20260912/logs/probe_mask_%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr_warmstart_gbnetwork_20260912/logs/probe_mask_%j.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=$BASE/overlays/gbnetwork_inference_walltime_h100_20260912
OUT=$BASE/results/nr_warmstart_gbnetwork_20260912
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
CKPT=$BASE/results/ckpt/gsfm_mask_e120_20260910/gsfm_gb_mask_s42_e120_best.pt
mkdir -p "$OUT/logs" "$OUT/probe"
export PYTHONPATH=$OVERLAY:$BASE:${PYTHONPATH:-} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$OVERLAY"
srun "$PY" -u "$OVERLAY/benchmark_inference_walltime.py" --stage probe --model gridsfm_mask --checkpoint "$CKPT" --parquet "$DATA" --probe-start 16 --probe-cap 4096 --fit-batch 16 --out "$OUT/probe/gridsfm_masked_s42_e120.json"
