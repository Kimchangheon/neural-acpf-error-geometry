#!/bin/bash -l
# Re-score the magnitude/angle-block CSP ablation with no post-output
# prescribed-state restoration.  This isolates the original full-state affine
# projection protocol used by the paper table.
#SBATCH --job-name=csp_blocks_nr
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/csp_projection_block_ablation_norestore_20260910/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/csp_projection_block_ablation_norestore_20260910/logs/%A_%a.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/csp_projection_block_ablation_norestore_20260910
PAPER_OVERLAY=${BASE}/overlays/model_seed_replicates_20260907
OUT=${BASE}/results/csp_projection_block_ablation_norestore_20260910
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
i=${SLURM_ARRAY_TASK_ID:?}; families=(g3 gridsfm); seeds=(41 42 43)
family=${families[$((i / 3))]}; seed=${seeds[$((i % 3))]}
case "${family}:${seed}" in
  g3:41) checkpoint=${BASE}/results/g3_ref_s41_ablation_20260829/ckpt/pignn_global_GBnetwork_g3_ref_s41_40_best_model.ckpt ;;
  g3:42) checkpoint=${BASE}/overlays/residual_split_20260829/pignn_global_GBnetwork_g3_s42_40_best_model.ckpt ;;
  g3:43) checkpoint=${BASE}/results/g3_ref_s43_ablation_20260829/ckpt/pignn_global_GBnetwork_g3_ref_s43_40_best_model.ckpt ;;
  gridsfm:41|gridsfm:43) checkpoint=${BASE}/results/model_seed_replicates_20260907/ckpt/gridsfm/gridsfm_GBnetwork_s${seed}_40_best.pt ;;
  gridsfm:42) checkpoint=${BASE}/overlays/residual_split_20260829/pfv2_gridsfm_GBnetwork_b26_best.pt ;;
  *) exit 2 ;;
esac
mkdir -p "$OUT/json" "$OUT/logs"
test -r "$DATA"; test -r "$checkpoint"; test -r "$OVERLAY/csp_projection_block_ablation.py"
# Use the exact paper-era split helper/scorer dependency stack.  In
# particular, it retains the original random_split evaluation ordering.
export PYTHONPATH="$OVERLAY:$PAPER_OVERLAY:$BASE:${PYTHONPATH:-}"
cd "$BASE"
echo "[task] model=$family seed=$seed checkpoint=$checkpoint"
echo "[protocol] seed-42 split; train-only calibration/bases; k=16; no restore_known; complex128 PB"
srun "$PY" -u "$OVERLAY/csp_projection_block_ablation.py" --model "$family" --checkpoint "$checkpoint" \
  --parquet "$DATA" --rank 16 --batch 16 --out "$OUT/json/${family}_s${seed}_blocks_k16_norestore.json"
