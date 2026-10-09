#!/bin/bash -l
# Helma mirror only for the still-pending GridSFM tasks (array ids 3--5).
# PIGNN-GC ids 0--2 remain running on Alex2 and are intentionally excluded.
#SBATCH --job-name=csp_blocks_h
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --array=3-5%3
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/csp_projection_block_ablation_helma_20260910/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/csp_projection_block_ablation_helma_20260910/logs/%A_%a.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OV=$BASE/overlays/csp_projection_block_ablation_helma_20260910
OUT=$BASE/results/csp_projection_block_ablation_helma_20260910
DATA=/home/hpc/b313dc/b313dc11/pb_data/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
i=${SLURM_ARRAY_TASK_ID:?}; seeds=(41 42 43); seed=${seeds[$((i%3))]}
case "$seed" in
  41|43) ckpt=$BASE/results/model_seed_replicates_20260907/ckpt/gridsfm/gridsfm_GBnetwork_s${seed}_40_best.pt ;;
  42) ckpt=$BASE/overlays/residual_split_20260829/pfv2_gridsfm_GBnetwork_b26_best.pt ;;
  *) exit 2 ;;
esac
mkdir -p "$OUT/json" "$OUT/logs"; test -r "$DATA"; test -r "$ckpt"
export PYTHONPATH="$OV:$BASE:${PYTHONPATH:-}"; cd "$BASE"
echo "model=gridsfm seed=$seed checkpoint=$ckpt"
srun "$PY" -u "$OV/csp_projection_block_ablation.py" --model gridsfm --checkpoint "$ckpt" \
  --parquet "$DATA" --rank 16 --batch 16 --out "$OUT/json/gridsfm_s${seed}_blocks_k16.json"
