#!/bin/bash -l
#SBATCH --job-name=lvn_diag12
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/lvn_diagnostics_20260822/Job_out/diag_%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/lvn_diagnostics_20260822/Job_out/diag_%A_%a.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-11

set -euo pipefail

BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
CAMPAIGN=${BASE}/sbatch/lvn_diagnostics_20260822
OVERLAY=${CAMPAIGN}/overlay
PRIOR_OVERLAY=${BASE}/sbatch/lvn_final_plan_20260821/overlay
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
SOURCE_PARQUET=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/pf_ppnr_v2/LVN_heo1_ppNR_37000.parquet
LOCAL_PARQUET=${TMPDIR}/LVN_heo1_ppNR_37000.parquet
SOURCE_CKPT_DIR=${BASE}/sbatch/lvn_final_plan_20260821/matched_results/ckpt
LOG_DIR=${CAMPAIGN}/results/logs
LOCAL_CKPT=${TMPDIR}/lvn_diag_ckpt
NR_IMPL=${OVERLAY}

SEEDS=(42 123 2026)
WEIGHTS=(0 1e-5 1e-4 1e-3)
WEIGHT_TAGS=(p0 p1e-5 p1e-4 p1e-3)
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
WEIGHT_INDEX=$((SLURM_ARRAY_TASK_ID % 4))
SEED=${SEEDS[${SEED_INDEX}]}
EXACT_WEIGHT=${WEIGHTS[${WEIGHT_INDEX}]}
WEIGHT_TAG=${WEIGHT_TAGS[${WEIGHT_INDEX}]}
GRAD_BATCHES=${GRAD_BATCHES:-8}
NR_CASES=${NR_CASES:-64}
RUN_SUFFIX=${RUN_SUFFIX:-}

mkdir -p "${LOG_DIR}" "${LOCAL_CKPT}"
export PYTHONPATH=${OVERLAY}:${PRIOR_OVERLAY}:${BASE}:${PYTHONPATH:-}

test -s "${SOURCE_PARQUET}"
test -f "${NR_IMPL}/newton_raphson_improved.py"
SOURCE_CKPT=${SOURCE_CKPT_DIR}/lvnmatched_s${SEED}_${WEIGHT_TAG}_150_best_model.ckpt
test -s "${SOURCE_CKPT}"
cp "${SOURCE_PARQUET}" "${LOCAL_PARQUET}"

RUN_NAME=lvndiag_s${SEED}_${WEIGHT_TAG}${RUN_SUFFIX}

srun "${PY}" -u "${OVERLAY}/train_valid_test.py" \
  --PARQUET "${LOCAL_PARQUET}" \
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --split_seed 42 --seed_value "${SEED}" \
  --max_train_samples 256 --max_valid_samples 64 --max_test_samples 64 \
  --preload_ram --preload_test \
  --log_to_file --log_dir "${LOG_DIR}" --ckpt_dir "${LOCAL_CKPT}" \
  --mode test --BLOCK_DIAG --skip_initial_eval \
  --BATCH 8 --LR 1e-5 --EPOCHS 0 \
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
  --report_gradient_alignment_batches "${GRAD_BATCHES}" --gradient_alignment_split train \
  --report_nr_polish --nr_polish_solver own \
  --nr_impl_path "${NR_IMPL}" --nr_polish_tol 1e-8 \
  --nr_polish_max_iter 30 --nr_polish_max_cases "${NR_CASES}" \
  --residual_tol_pu 1e-8 --convergence_tol_pu 1e-8 \
  --run_name "${RUN_NAME}"
