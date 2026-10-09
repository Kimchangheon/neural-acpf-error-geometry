#!/bin/bash -l
# Four post-hoc output blocks, one model forward pass per split/task.
# Array ordering deliberately matches the current three-seed paper table.
#SBATCH --job-name=csp_blocks
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --array=0-5%2
#SBATCH --output=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/csp_projection_block_ablation_20260910/logs/%A_%a.out
#SBATCH --error=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/csp_projection_block_ablation_20260910/logs/%A_%a.err
set -euo pipefail
BASE=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OV=$BASE/overlays/csp_projection_block_ablation_20260910
OUT=$BASE/results/csp_projection_block_ablation_20260910
DATA=$OUT/data/GBnetwork.parquet
PY=/home/hpc/iwi5/iwi5295h/conda-envs/gridfm-py312/bin/python
i=${SLURM_ARRAY_TASK_ID:?}; families=(g3 gridsfm); seeds=(41 42 43)
family=${families[$((i/3))]}; seed=${seeds[$((i%3))]}
case "$family:$seed" in
g3:41|g3:43) ckpt=$BASE/results/nr1_multiproc_gbnetwork_20260910/checkpoints/pignn_global_GBnetwork_g3_ref_s${seed}_40_best_model.ckpt ;;
g3:42) ckpt=$BASE/overlays/residual_diagnostics_20260828/pignn_global_GBnetwork_g3_s42_40_best_model.ckpt ;;
gridsfm:41|gridsfm:43) ckpt=$BASE/results/nr1_multiproc_gbnetwork_20260910/checkpoints/gridsfm_GBnetwork_s${seed}_40_best.pt ;;
gridsfm:42) ckpt=$BASE/overlays/residual_diagnostics_20260828/pfv2_gridsfm_GBnetwork_b26_best.pt ;;
*) exit 2 ;; esac
mkdir -p "$OUT/json" "$OUT/logs"; test -r "$DATA"; test -r "$ckpt"
export PYTHONPATH="$OV:$BASE:${PYTHONPATH:-}"; cd "$BASE"
echo "model=$family seed=$seed checkpoint=$ckpt"
srun "$PY" -u "$OV/csp_projection_block_ablation.py" --model "$family" --checkpoint "$ckpt" --parquet "$DATA" --rank 16 --batch 16 --out "$OUT/json/${family}_s${seed}_blocks_k16.json"
