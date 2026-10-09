#!/bin/bash -l
# Stage 2: test-only fixed-norm directional experiment at the frozen k*=16.
# Stage-1 validation selection is recorded in CONTROLLED_ERROR_GEOMETRY_PROTOCOL.md.
#SBATCH --job-name=cgeom_t16
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/controlled_geometry_20260905/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/controlled_geometry_20260905/logs/%A_%a.err

set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/controlled_geometry_20260905
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
OUT=${BASE}/results/controlled_geometry_20260905
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
CKPT=${BASE}/overlays/residual_split_20260829
models=(g3 gridsfm graphkit lumina)
i=${SLURM_ARRAY_TASK_ID:?}
model=${models[$i]}
mkdir -p "${OUT}/json" "${OUT}/logs"
test -r "${OUT}/json/g3_validation.json"
test -r "${OUT}/json/gridsfm_validation.json"
test -r "${OUT}/json/graphkit_validation.json"
test -r "${OUT}/json/lumina_validation.json"
export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
cd "${BASE}"
srun "${PY}" -u "${OVERLAY}/controlled_error_geometry.py" \
  --phase test --ranks 16 --batch 16 --models "${model}" \
  --parquet "${DATA}" --out "${OUT}/json/${model}_test_k16.json" \
  --pignn "${CKPT}/pignn_global_GBnetwork_g3_s42_40_best_model.ckpt" \
  --gridsfm "${CKPT}/pfv2_gridsfm_GBnetwork_b26_best.pt" \
  --graphkit "${BASE}/results/ckpt/pfv2_graphkit_D_v2/pfv2_graphkit_GBnetwork_b52_best.pt" \
  --lumina "${BASE}/results/ckpt/pfv2_lumina_D_v2/pfv2_lumina_GBnetwork_b32_best.pt"
