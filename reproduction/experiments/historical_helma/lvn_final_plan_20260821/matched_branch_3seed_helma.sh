#!/bin/bash -l
#SBATCH --job-name=lvn_match_branch
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/lvn_final_plan_20260821/Job_out/matched_%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/lvn_final_plan_20260821/Job_out/matched_%A_%a.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-11

set -euo pipefail

BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
CAMPAIGN=${BASE}/sbatch/lvn_final_plan_20260821
OVERLAY=${CAMPAIGN}/overlay
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
SOURCE_PARQUET=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/pf_ppnr_v2/LVN_heo1_ppNR_37000.parquet
LOCAL_PARQUET=${TMPDIR}/LVN_heo1_ppNR_37000.parquet
RESULT_ROOT=${CAMPAIGN}/matched_results
LOG_DIR=${RESULT_ROOT}/logs
VAULT_CKPT=${RESULT_ROOT}/ckpt
SOURCE_CKPT_DIR=${CAMPAIGN}/results/ckpt
LOCAL_CKPT=${TMPDIR}/lvn_matched_ckpt

SEEDS=(42 123 2026)
WEIGHTS=(0 1e-5 1e-4 1e-3)
WEIGHT_TAGS=(p0 p1e-5 p1e-4 p1e-3)
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
WEIGHT_INDEX=$((SLURM_ARRAY_TASK_ID % 4))
SEED=${SEEDS[${SEED_INDEX}]}
EXACT_WEIGHT=${WEIGHTS[${WEIGHT_INDEX}]}
WEIGHT_TAG=${WEIGHT_TAGS[${WEIGHT_INDEX}]}

mkdir -p "${LOG_DIR}" "${VAULT_CKPT}" "${LOCAL_CKPT}"
trap 'status=$?; cp -a "${LOCAL_CKPT}/." "${VAULT_CKPT}/" 2>/dev/null || true; exit ${status}' EXIT

export PYTHONPATH=${OVERLAY}:${BASE}:${PYTHONPATH:-}
cp "${SOURCE_PARQUET}" "${LOCAL_PARQUET}"

SOURCE_CKPT=${SOURCE_CKPT_DIR}/lvncanon_s${SEED}_sup_200_best_model.ckpt
test -s "${SOURCE_CKPT}"

RUN_NAME=lvnmatched_s${SEED}_${WEIGHT_TAG}

srun "${PY}" -u "${OVERLAY}/train_valid_test.py" \
  --PARQUET "${LOCAL_PARQUET}" \
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --split_seed 42 --seed_value "${SEED}" \
  --max_train_samples 256 --max_valid_samples 64 --max_test_samples 64 \
  --preload_ram --preload_test \
  --log_to_file --log_dir "${LOG_DIR}" --ckpt_dir "${LOCAL_CKPT}" \
  --mode train_valid_test --BLOCK_DIAG \
  --BATCH 8 --LR 1e-5 --VAL_EVERY 10 --EPOCHS 150 \
  --model GNSMsg_EdgeSelfAttn --d 4 --d_hi 24 --n_heads 8 \
  --num_attn_layers 8 --K 40 --solver_update_mode direct \
  --vlimit --preserve_zero_heads \
  --residual_feature_norm ybus --edge_feature_norm signed_log \
  --correction_target_norm none \
  --mse_weight 1.0 --physics_weight 0 \
  --physics_loss_form mse --physics_residual_norm graph \
  --exact_physics_residual_norm local_ybus \
  --exact_physics_weight "${EXACT_WEIGHT}" \
  --init_checkpoint "${SOURCE_CKPT}" \
  --run_name "${RUN_NAME}"

test -s "${LOCAL_CKPT}/${RUN_NAME}_150_best_model.ckpt"
