#!/bin/bash -l
# Compare the pure full-state affine CSP with the operational restore_known policy.
# No training or data generation; each model is scored once with both policies.
#SBATCH --job-name=csp_full_vs_restore
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/csp_fullstate_vs_restore_gbnetwork_20260910/logs/%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/csp_fullstate_vs_restore_gbnetwork_20260910/logs/%j.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=$BASE/overlays/csp_posthoc_diagnostics_gbnetwork_20260910
OUT=$BASE/results/csp_fullstate_vs_restore_gbnetwork_20260910
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
G3=$BASE/overlays/residual_split_20260829/pignn_global_GBnetwork_g3_s42_40_best_model.ckpt
GRIDSFM=$BASE/overlays/residual_split_20260829/pfv2_gridsfm_GBnetwork_b26_best.pt
mkdir -p "$OUT/logs" "$OUT/json"
test -r "$DATA"; test -r "$G3"; test -r "$GRIDSFM"
LOCAL_DATA=${TMPDIR:?}/GBnetwork.parquet
cp "$DATA" "$LOCAL_DATA"
test "$(stat -c %s "$DATA")" = "$(stat -c %s "$LOCAL_DATA")"
export PYTHONPATH=$OVERLAY:$BASE:${PYTHONPATH:-}
export OMP_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$OVERLAY"
for SPEC in "g3:$G3:PIGNN-GC" "gridsfm:$GRIDSFM:GridSFM"; do
  IFS=: read -r KIND CKPT LABEL <<< "$SPEC"
  srun "$PY" -u "$OVERLAY/output_intervention_metrics.py" \
    --model "$KIND" --checkpoint "$CKPT" --parquet "$LOCAL_DATA" --rank 16 --batch 16 \
    --projection-variants full restore_known \
    --out "$OUT/json/${LABEL}_s42_fullstate_vs_restore_k16.json"
done
