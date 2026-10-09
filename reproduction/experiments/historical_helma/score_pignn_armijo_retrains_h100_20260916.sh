#!/bin/bash -l
#SBATCH --job-name=score_armijo
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/pignn_armijo_retrain_scoring_20260916/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/pignn_armijo_retrain_scoring_20260916/logs/%A_%a.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/pignn_armijo_retrain_scoring_20260916
OUT=${BASE}/results/pignn_armijo_retrain_scoring_20260916
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
SRC=/home/hpc/b313dc/b313dc11/pb_data/GBnetwork.parquet
DATA=${TMPDIR:?}/GBnetwork.parquet
cp "$SRC" "$DATA"
mkdir -p "$OUT/logs" "$OUT/json"

i=${SLURM_ARRAY_TASK_ID:?}
configs=(async_s41 multi_rhs_s41 multi_rhs_s42 multi_rhs_s43)
splits=(validation test)
config=${configs[$((i / 2))]}; split=${splits[$((i % 2))]}
case "$config" in
  async_s41)
    ckpt=${BASE}/results/pignn_gc_chunked_async_e120_lr1e5_20260914/ckpt/pignn_gc_chunked_async_GBnetwork_s41_e120_lr1e5_120_best_model.ckpt
    extra=(--pignn-batched-armijo --pignn-global-context-packed-fastpath)
    ;;
  multi_rhs_s41|multi_rhs_s42|multi_rhs_s43)
    seed=${config##*s}
    ckpt=${BASE}/results/pignn_gc_chunked_multi_rhs_e120_lr1e5_20260914/ckpt/pignn_gc_chunked_multi_rhs_GBnetwork_s${seed}_e120_lr1e5_120_best_model.ckpt
    extra=(--pignn-batched-armijo --pignn-multi-rhs-armijo --pignn-global-context-packed-fastpath)
    ;;
esac
test -r "$ckpt"; test -r "$OVERLAY/output_intervention_metrics.py"
export PYTHONPATH="$OVERLAY:$BASE:${PYTHONPATH:-}"
cd "$BASE"
echo "[score] config=$config split=$split checkpoint=$ckpt"
srun "$PY" -u "$OVERLAY/output_intervention_metrics.py" \
  --model g3 --checkpoint "$ckpt" --parquet "$DATA" --rank 16 --batch 16 \
  --evaluation-split "$split" --projection-variants full restore_known "${extra[@]}" \
  --out "$OUT/json/${config}_${split}_k16.json"
