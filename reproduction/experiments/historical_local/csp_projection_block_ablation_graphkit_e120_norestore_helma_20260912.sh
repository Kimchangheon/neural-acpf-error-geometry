#!/bin/bash -l
# GraphKit E120 magnitude/angle CSP block ablation under the paper's original
# full-state affine-projection convention (no prescribed-state restoration).
#SBATCH --job-name=gk120_blocks
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/csp_projection_block_ablation_graphkit_e120_norestore_20260912/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/csp_projection_block_ablation_graphkit_e120_norestore_20260912/logs/%A_%a.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=$BASE/overlays/csp_projection_block_ablation_graphkit_e120_norestore_20260912
OUT=$BASE/results/csp_projection_block_ablation_graphkit_e120_norestore_20260912
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
seeds=(41 42 43); stems=(k06_s41_e120 k08_s42_e120 k07_s43_e120)
i=${SLURM_ARRAY_TASK_ID:?}; seed=${seeds[$i]}; stem=${stems[$i]}
checkpoint=$BASE/results/ckpt/gk_e120_20260910/${stem}_best.pt
mkdir -p "$OUT/json" "$OUT/logs"
test -r "$DATA"; test -r "$checkpoint"; test -r "$OVERLAY/csp_projection_block_ablation.py"
export PYTHONPATH="$OVERLAY:$BASE:${PYTHONPATH:-}"
cd "$BASE"
echo "[task] GraphKit E120 seed=$seed checkpoint=$checkpoint"
echo "[protocol] seed-42 split; train-only calibration/bases; k=16; full-state; no restore_known; complex128 PB"
srun "$PY" -u "$OVERLAY/csp_projection_block_ablation.py" --model graphkit --checkpoint "$checkpoint" --parquet "$DATA" --rank 16 --batch 16 --out "$OUT/json/graphkit_s${seed}_e120_blocks_k16_norestore.json"
