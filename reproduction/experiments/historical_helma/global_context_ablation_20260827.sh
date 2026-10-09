#!/bin/bash -l
# Global-context PIGNN ablation: G1--G3 only, seed 42.
# G0 signed-log is an existing reference and is deliberately not submitted.
# The batch script is copied into an overlay so the user's working tree is not
# required to be synchronized on the compute node.
#SBATCH --job-name=pignn_global
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/global_context_ablation_20260827/job_out/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/global_context_ablation_20260827/job_out/%A_%a.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-5%3

set -euo pipefail
trap 'rc=$?; echo "[failure] line=${LINENO} command=${BASH_COMMAND} rc=${rc}" >&2; exit "${rc}"' ERR

BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/global_context_ablation_20260827
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
CAMPAIGN=${BASE}/results/global_context_ablation_20260827
LOG_DIR=${CAMPAIGN}/logs
CKPT_DIR=${CAMPAIGN}/ckpt

grids=(case300 GBnetwork)
files=(
  /home/hpc/b313dc/b313dc11/data_staging/global_context_ablation_20260827/case300_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet
  /home/hpc/b313dc/b313dc11/data_staging/grl_oegr_gate_20260825/GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet
)
batches=(32 23)
modes=(meanmax_pre attn_pre attn_post)

task=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
grid_idx=$((task / 3))
mode_idx=$((task % 3))
if (( grid_idx < 0 || grid_idx >= 2 || mode_idx < 0 || mode_idx >= 3 )); then
  echo "invalid task=${task}" >&2
  exit 2
fi

grid=${grids[$grid_idx]}
source_file=${files[$grid_idx]}
batch=${batches[$grid_idx]}
mode=${modes[$mode_idx]}
run_name=pignn_global_${grid}_g$((mode_idx + 1))_s42
local_parquet=${TMPDIR:?TMPDIR is required}/${grid}.parquet

echo "[preflight] task=${task} grid=${grid} mode=${mode} tmpdir=${TMPDIR}"
test -r "${source_file}" || { echo "unreadable source: ${source_file}" >&2; exit 10; }
test -x "${PY}" || { echo "unusable python: ${PY}" >&2; exit 11; }
test -f "${OVERLAY}/train_valid_test.py" || { echo "missing overlay driver" >&2; exit 12; }
test -f "${OVERLAY}/GNSMsg_SelfAttention_armijo.py" || { echo "missing overlay model" >&2; exit 13; }
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

echo "[stage] copying $(stat -c %s "${source_file}") bytes to ${local_parquet}"
cp "${source_file}" "${local_parquet}"
test "$(stat -c %s "${local_parquet}")" = "$(stat -c %s "${source_file}")"

export PYTHONPATH=${OVERLAY}:${BASE}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${BASE}"

echo "[campaign] task=${task} grid=${grid} mode=${mode} model_seed=42 split_seed=42"
echo "[model] PIGNN EdgeSelfAttn K=40 d=4 d_hi=24 heads=8 local_layers=8"
echo "[solver] direct geometric_safe Armijo vlimit=on preserve_zero_heads=on"
echo "[objective] mse_weight=5 physics_weight=1 logcosh graph-normalized exact_weight=0"
echo "[representation] residual=signed_log edge=none global_context=${mode}"
echo "[data] source=${source_file} bytes=$(stat -c %s "${source_file}")"
echo "[code] $(sha256sum "${OVERLAY}/train_valid_test.py" "${OVERLAY}/GNSMsg_SelfAttention_armijo.py" | tr '\n' ' ')"
echo "[gpu] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

srun "${PY}" -u "${OVERLAY}/train_valid_test.py" \
  --PARQUET "${local_parquet}" \
  --run_name "${run_name}" \
  --log_to_file --log_dir "${LOG_DIR}" --ckpt_dir "${CKPT_DIR}" \
  --mode train_valid_test \
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 \
  --preload_ram --preload_test \
  --BATCH "${batch}" --EPOCHS 40 --LR 1e-5 --VAL_EVERY 1 \
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
  --global_context_mode "${mode}" \
  --exact_physics_weight 0
