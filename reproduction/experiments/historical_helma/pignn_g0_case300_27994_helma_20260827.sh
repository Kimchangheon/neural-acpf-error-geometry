#!/bin/bash -l
# Matched G0 signed-log baseline for the 27,994-row case300 corpus.
# This is intentionally separate from the historical 37,930-row v2 G0 result.
# It does not modify or depend on the active G1--G5/RANGE jobs.
#SBATCH --job-name=pignn_g0_27994
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/g0_case300_27994_20260827/job_out/%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/g0_case300_27994_20260827/job_out/%j.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00

set -euo pipefail
trap 'rc=$?; echo "[failure] line=${LINENO} command=${BASH_COMMAND} rc=${rc}" >&2; exit "${rc}"' ERR

BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/global_context_ablation_20260827
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
SOURCE=/home/hpc/b313dc/b313dc11/data_staging/global_context_ablation_20260827/case300_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet
CAMPAIGN=${BASE}/results/g0_case300_27994_20260827
LOG_DIR=${CAMPAIGN}/logs
CKPT_DIR=${CAMPAIGN}/ckpt
local_parquet=${TMPDIR:?TMPDIR is required}/case300.parquet

echo "[preflight] matched case300 G0, corpus=27,994 rows, tmpdir=${TMPDIR}"
test -r "${SOURCE}" || { echo "unreadable source: ${SOURCE}" >&2; exit 10; }
test -x "${PY}" || { echo "unusable python: ${PY}" >&2; exit 11; }
test -r "${OVERLAY}/train_valid_test.py" || { echo "missing overlay driver" >&2; exit 12; }
test -r "${OVERLAY}/GNSMsg_SelfAttention_armijo.py" || { echo "missing overlay model" >&2; exit 13; }
mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${BASE}/results/plots"

echo "[stage] copying $(stat -c %s "${SOURCE}") bytes to ${local_parquet}"
cp "${SOURCE}" "${local_parquet}"
test "$(stat -c %s "${local_parquet}")" = "$(stat -c %s "${SOURCE}")"

export PYTHONPATH=${OVERLAY}:${BASE}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${BASE}"

echo "[campaign] pignn_g0_case300_27994_s42"
echo "[model] EdgeSelfAttn K=40 d=4 d_hi=24 heads=8 local_layers=8"
echo "[solver] direct geometric_safe Armijo vlimit=on preserve_zero_heads=on"
echo "[objective] mse_weight=5 physics_weight=1 logcosh graph-normalized"
echo "[representation] residual=signed_log edge=none global_context=none"
echo "[data] source=${SOURCE} bytes=$(stat -c %s "${SOURCE}")"
echo "[code] $(sha256sum "${OVERLAY}/train_valid_test.py" "${OVERLAY}/GNSMsg_SelfAttention_armijo.py" | tr '\n' ' ')"
echo "[gpu] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

srun "${PY}" -u "${OVERLAY}/train_valid_test.py" \
  --PARQUET "${local_parquet}" \
  --run_name pignn_g0_case300_27994_s42 \
  --log_to_file --log_dir "${LOG_DIR}" --ckpt_dir "${CKPT_DIR}" \
  --mode train_valid_test \
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 \
  --preload_ram --preload_test \
  --BATCH 32 --EPOCHS 40 --LR 1e-5 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 \
  --seed_value 42 --split_seed 42 \
  --BLOCK_DIAG --PINN --model GNSMsg_EdgeSelfAttn \
  --d 4 --d_hi 24 --n_heads 8 --num_attn_layers 8 --K 40 \
  --solver_update_mode direct --use_armijo --armijo_mode geometric_safe \
  --vlimit --preserve_zero_heads \
  --mse_weight 5.0 --physics_weight 1.0 \
  --physics_loss_form logcosh --physics_residual_norm graph \
  --lr_scheduler CosineAnnealingLR \
  --residual_feature_norm signed_log --edge_feature_norm none \
  --global_context_mode none \
  --exact_physics_weight 0
