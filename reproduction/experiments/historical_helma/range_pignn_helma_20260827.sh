#!/bin/bash -l
# RANGE-PIGNN multiple-master ablation on GBnetwork, seed 42.
# The model/driver are loaded from an immutable overlay; G1--G5 jobs are not
# modified, restarted, or used as dependencies.
#SBATCH --job-name=pignn_range
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/range_pignn_20260827/job_out/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/range_pignn_20260827/job_out/%A_%a.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3

set -euo pipefail
trap 'rc=$?; echo "[failure] line=${LINENO} command=${BASH_COMMAND} rc=${rc}" >&2; exit "${rc}"' ERR

BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/range_pignn_20260827
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
SOURCE=/home/hpc/b313dc/b313dc11/data_staging/grl_oegr_gate_20260825/GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet
CAMPAIGN=${BASE}/results/range_pignn_20260827
LOG_DIR=${CAMPAIGN}/logs
CKPT_DIR=${CAMPAIGN}/ckpt
GPU_LOG_DIR=${CAMPAIGN}/gpu_memory

masters=(1 4 8)
task=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
if (( task < 0 || task >= 3 )); then
  echo "invalid task=${task}" >&2
  exit 2
fi
num_masters=${masters[$task]}
run_name=pignn_range_GBnetwork_m${num_masters}_s42
local_parquet=${TMPDIR:?TMPDIR is required}/GBnetwork.parquet
gpu_log=${GPU_LOG_DIR}/${SLURM_ARRAY_JOB_ID}_${task}.csv

echo "[preflight] task=${task} masters=${num_masters} tmpdir=${TMPDIR}"
test -r "${SOURCE}" || { echo "unreadable source: ${SOURCE}" >&2; exit 10; }
test -x "${PY}" || { echo "unusable python: ${PY}" >&2; exit 11; }
test -r "${OVERLAY}/train_valid_test.py" || { echo "missing overlay driver" >&2; exit 12; }
test -r "${OVERLAY}/GNSMsg_SelfAttention_armijo.py" || { echo "missing overlay model" >&2; exit 13; }
test ! -w "${OVERLAY}/train_valid_test.py" || { echo "overlay driver is mutable" >&2; exit 14; }
test ! -w "${OVERLAY}/GNSMsg_SelfAttention_armijo.py" || { echo "overlay model is mutable" >&2; exit 15; }
mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${GPU_LOG_DIR}"

echo "[stage] copying $(stat -c %s "${SOURCE}") bytes to ${local_parquet}"
cp "${SOURCE}" "${local_parquet}"
test "$(stat -c %s "${local_parquet}")" = "$(stat -c %s "${SOURCE}")"

export PYTHONPATH=${OVERLAY}:${BASE}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The executed script path keeps the immutable overlay first on sys.path, but
# plots use historical relative paths and therefore need a writable cwd.
cd "${BASE}"

echo "[campaign] RANGE-PIGNN GBnetwork M=${num_masters} model_seed=42 split_seed=42"
echo "[model] EdgeSelfAttn K=40 d=4 d_hi=24 heads=8 local_layers=8 range_post"
echo "[range] masters=${num_masters} master_dim=24 heads=8 PE=hop_slack_rbf:10 persistent_over_K=true"
echo "[solver] direct geometric_safe Armijo vlimit=on preserve_zero_heads=on"
echo "[objective] mse_weight=5 physics_weight=1 logcosh graph-normalized exact_weight=0"
echo "[representation] residual=signed_log edge=none outer_global_gate=none"
echo "[data] source=${SOURCE} bytes=$(stat -c %s "${SOURCE}")"
echo "[code] $(sha256sum "${OVERLAY}/train_valid_test.py" "${OVERLAY}/GNSMsg_SelfAttention_armijo.py" | tr '\n' ' ')"
echo "[gpu] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

echo "timestamp,pid,used_gpu_memory_mib" > "${gpu_log}"
(
  while true; do
    timestamp=$(date -Ins)
    nvidia-smi --query-compute-apps=pid,used_gpu_memory \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk -v ts="${timestamp}" -F',' '{gsub(/ /,"",$0); print ts "," $0}' \
      || true
    sleep 15
  done
) >> "${gpu_log}" &
monitor_pid=$!
start_seconds=${SECONDS}

if srun "${PY}" -u "${OVERLAY}/train_valid_test.py" \
  --PARQUET "${local_parquet}" \
  --run_name "${run_name}" \
  --log_to_file --log_dir "${LOG_DIR}" --ckpt_dir "${CKPT_DIR}" \
  --mode train_valid_test \
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 \
  --preload_ram --preload_test \
  --BATCH 23 --EPOCHS 40 --LR 1e-5 --VAL_EVERY 1 \
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
  --global_context_mode range_post \
  --range_num_masters "${num_masters}" \
  --range_master_dim 24 --range_num_heads 8 \
  --range_positional_encoding hop_slack_rbf --range_pe_dim 10 \
  --range_step_diagnostic_batches 4 \
  --exact_physics_weight 0; then
  rc=0
else
  rc=$?
fi

kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
elapsed=$((SECONDS - start_seconds))
peak_gpu_mib=$(awk -F',' 'NR>1 && $3+0>m {m=$3+0} END {print m+0}' "${gpu_log}")
echo "[resources] elapsed_seconds=${elapsed} peak_gpu_memory_mib=${peak_gpu_mib} gpu_log=${gpu_log}"
exit "${rc}"
