#!/bin/bash -l
#SBATCH --job-name=lvn_canon_curric
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/lvn_final_plan_20260821/Job_out/curric_%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/lvn_final_plan_20260821/Job_out/curric_%A_%a.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-2

set -euo pipefail

BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
CAMPAIGN=${BASE}/sbatch/lvn_final_plan_20260821
OVERLAY=${CAMPAIGN}/overlay
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
SOURCE_PARQUET=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/pf_ppnr_v2/LVN_heo1_ppNR_37000.parquet
LOCAL_PARQUET=${TMPDIR}/LVN_heo1_ppNR_37000.parquet
RESULT_ROOT=${CAMPAIGN}/results
LOG_DIR=${RESULT_ROOT}/logs
VAULT_CKPT=${RESULT_ROOT}/ckpt
LOCAL_CKPT=${TMPDIR}/lvn_final_plan_ckpt

SEEDS=(42 123 2026)
SEED=${SEEDS[${SLURM_ARRAY_TASK_ID}]}

mkdir -p "${LOG_DIR}" "${VAULT_CKPT}" "${LOCAL_CKPT}"
trap 'status=$?; cp -a "${LOCAL_CKPT}/." "${VAULT_CKPT}/" 2>/dev/null || true; exit ${status}' EXIT

export PYTHONPATH=${OVERLAY}:${BASE}:${PYTHONPATH:-}
cp "${SOURCE_PARQUET}" "${LOCAL_PARQUET}"

common=(
  --PARQUET "${LOCAL_PARQUET}"
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus
  --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128
  --train_ratio 0.3333 --valid_ratio 0.3333 --split_seed 42 --seed_value "${SEED}"
  --max_train_samples 256 --max_valid_samples 64 --max_test_samples 64
  --preload_ram --preload_test
  --log_to_file --log_dir "${LOG_DIR}" --ckpt_dir "${LOCAL_CKPT}"
  --mode train_valid_test --BLOCK_DIAG
  --BATCH 8 --LR 1e-5 --VAL_EVERY 10
  --model GNSMsg_EdgeSelfAttn --d 4 --d_hi 24 --n_heads 8
  --num_attn_layers 8 --K 40 --solver_update_mode direct
  --vlimit --preserve_zero_heads
  --residual_feature_norm ybus --edge_feature_norm signed_log
  --correction_target_norm none
  --mse_weight 1.0 --physics_weight 0
  --physics_loss_form mse --physics_residual_norm graph
  --exact_physics_residual_norm local_ybus
)

run_stage() {
  local name=$1
  local epochs=$2
  local exact_weight=$3
  local init_checkpoint=${4:-}
  local args=(
    "${common[@]}"
    --run_name "${name}" --EPOCHS "${epochs}"
    --exact_physics_weight "${exact_weight}"
  )
  if [[ -n "${init_checkpoint}" ]]; then
    args+=(--init_checkpoint "${init_checkpoint}")
  fi
  srun "${PY}" -u "${OVERLAY}/train_valid_test.py" "${args[@]}"
}

SUP_NAME=lvncanon_s${SEED}_sup
run_stage "${SUP_NAME}" 200 0
SUP_CKPT=${LOCAL_CKPT}/${SUP_NAME}_200_best_model.ckpt
test -s "${SUP_CKPT}"

P1_NAME=lvncanon_s${SEED}_p1e-5
run_stage "${P1_NAME}" 50 1e-5 "${SUP_CKPT}"
P1_CKPT=${LOCAL_CKPT}/${P1_NAME}_50_best_model.ckpt
test -s "${P1_CKPT}"

P2_NAME=lvncanon_s${SEED}_p1e-4
run_stage "${P2_NAME}" 50 1e-4 "${P1_CKPT}"
P2_CKPT=${LOCAL_CKPT}/${P2_NAME}_50_best_model.ckpt
test -s "${P2_CKPT}"

P3_NAME=lvncanon_s${SEED}_p1e-3
run_stage "${P3_NAME}" 50 1e-3 "${P2_CKPT}"
test -s "${LOCAL_CKPT}/${P3_NAME}_50_best_model.ckpt"
