#!/bin/bash -l
# Frozen reproduction of the existing 128 x 32 random-JVP protocol plus
# calibrated-error JVP diagnostics.  No training or dataset generation.
#SBATCH --job-name=jac_repro_audit
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/jacobian_reproducibility_gbnetwork_20260910/logs/%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/jacobian_reproducibility_gbnetwork_20260910/logs/%j.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=$BASE/overlays/jacobian_reproducibility_gbnetwork_20260910
OUT=$BASE/results/jacobian_reproducibility_gbnetwork_20260910
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
export OMP_NUM_THREADS=24 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$OVERLAY"
srun "$PY" -u "$OVERLAY/jacobian_subspace_reproducibility.py" \
 --parquet "$LOCAL_DATA" --g3 "$G3" --gridsfm "$GRIDSFM" \
 --rank 16 --scenarios 128 --directions 32 --seed 20260909 --basis-batch 16 \
 --out-json "$OUT/json/jacobian_reproducibility_k16.json" \
 --out-csv "$OUT/json/jacobian_reproducibility_k16.csv" \
 --out-npz "$OUT/json/jacobian_random_gains_k16.npz"
