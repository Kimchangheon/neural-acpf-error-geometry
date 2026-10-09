#!/bin/bash -l
# Metric-complete rescore from frozen LUMINA seed 41/43 C/CSP16 caches.
#SBATCH --job-name=lum_nr1_s4143_c
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr1_lumina_gbv06z_3seed_20260914/logs/complete_%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr1_lumina_gbv06z_3seed_20260914/logs/complete_%A_%a.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=$BASE/overlays/nr1_lumina_gbv06z_3seed_20260914
RESULT=$BASE/results/nr1_lumina_gbv06z_3seed_20260914
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
seeds=(41 43)
seed=${seeds[${SLURM_ARRAY_TASK_ID:?}]}
export PYTHONPATH="$OVERLAY:$BASE:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "[frozen protocol] seed=$seed eta=1.0; no neural inference"
srun "$PY" -u "$OVERLAY/nr1_metric_complete_rescore.py" \
  --model lumina --seed "$seed" --parquet "$DATA" --eta 1.0 \
  --workers 16 --batch 16 --chunk 16 \
  --cache-dir "$RESULT/cache/lumina_s${seed}_test" \
  --out-dir "$RESULT/complete/lumina_s${seed}"
