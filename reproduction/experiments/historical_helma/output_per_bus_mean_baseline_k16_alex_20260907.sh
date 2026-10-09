#!/bin/bash -l
# Training-split per-bus mean voltage/angle baseline under the same scorer as Table 1.
#SBATCH --job-name=meanpb16
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/controlled_geometry_20260905/logs/%x_%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/controlled_geometry_20260905/logs/%x_%j.err

set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/controlled_geometry_20260905
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
OUT=${BASE}/results/controlled_geometry_20260905/json/per_bus_mean_baseline_output_interventions_k16.json
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
cd "${BASE}"
srun "${PY}" -u "${OVERLAY}/output_intervention_metrics.py" \
  --model constant --parquet "${DATA}" --rank 16 --batch 16 --out "${OUT}"
