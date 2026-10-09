#!/bin/bash -l
# Numerical-compatibility pilot for the reusable pandapower Newton core.
# This is deliberately separate from the full wall-time benchmark.
#SBATCH --job-name=ppnr_gb_pilot
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr_warmstart_gbnetwork_20260912/logs/pilot_%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr_warmstart_gbnetwork_20260912/logs/pilot_%j.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OUT=$BASE/results/nr_warmstart_gbnetwork_20260912
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
mkdir -p "$OUT/logs"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$BASE:${PYTHONPATH:-}"
cd "$BASE"
python3.9 -u nr_warmstart_pandapower_pilot.py --parquet "$DATA" | tee "$OUT/pilot.json"
