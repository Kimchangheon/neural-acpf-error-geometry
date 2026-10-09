#!/bin/bash -l
# Comprehensive Raw/C/P16/CSP16 table after frozen validation rank selection.
#SBATCH --job-name=outmet16
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/controlled_geometry_20260905/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/controlled_geometry_20260905/logs/%A_%a.err

set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/controlled_geometry_20260905
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
OUT=${BASE}/results/controlled_geometry_20260905/json
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
CKPT=${BASE}/overlays/residual_split_20260829
models=(g3 gridsfm graphkit lumina)
checkpoints=(
  "${CKPT}/pignn_global_GBnetwork_g3_s42_40_best_model.ckpt"
  "${CKPT}/pfv2_gridsfm_GBnetwork_b26_best.pt"
  "${BASE}/results/ckpt/pfv2_graphkit_D_v2/pfv2_graphkit_GBnetwork_b52_best.pt"
  "${BASE}/results/ckpt/pfv2_lumina_D_v2/pfv2_lumina_GBnetwork_b32_best.pt"
)
i=${SLURM_ARRAY_TASK_ID:?}; model=${models[$i]}; checkpoint=${checkpoints[$i]}
mkdir -p "${OUT}"
test -r "${OUT}/g3_validation.json"
export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
cd "${BASE}"
srun "${PY}" -u "${OVERLAY}/output_intervention_metrics.py" \
  --model "${model}" --checkpoint "${checkpoint}" --parquet "${DATA}" \
  --rank 16 --batch 16 --out "${OUT}/${model}_output_interventions_k16.json"
