#!/bin/bash -l
# Residual-input versus iterative residual-feedback test on full GBnetwork.
# Submit with: sbatch lumina_gb_corrector_helma_20260912.sh
# Array mapping:
#   0: residual input only, K=1, sd002_zerohead
#   1: feedback K=2, alpha=0.5, sd002
#   2: feedback K=4, alpha=0.5, sd002
#   3: feedback K=4, alpha=0.5 with wider per-step clips, sd002
#
# All arms use the stable GB learning rate (3e-4), dropout-0 LUMINA, and the
# full 37,022-row GBnetwork parquet.  The only intentional initialization
# asymmetry is arm 0's zero head, requested to test residual input in isolation.
#
#SBATCH --job-name=gbc
#SBATCH --array=0-3
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/gbc_%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/gbc_%A_%a.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

set -euo pipefail

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    RUN_NAME="gbc01_residual_K1"
    INIT_RECIPE="sd002_zerohead"
    CORRECTOR_K=1
    CORRECTOR_ALPHA=0.5
    EXTRA_CORRECTOR_ARGS=""
    ;;
  1)
    RUN_NAME="gbc02_corrector_K2"
    INIT_RECIPE="sd002"
    CORRECTOR_K=2
    CORRECTOR_ALPHA=0.5
    EXTRA_CORRECTOR_ARGS=""
    ;;
  2)
    RUN_NAME="gbc03_corrector_K4"
    INIT_RECIPE="sd002"
    CORRECTOR_K=4
    CORRECTOR_ALPHA=0.5
    EXTRA_CORRECTOR_ARGS=""
    ;;
  3)
    RUN_NAME="gbc04_corrector_K4wide"
    INIT_RECIPE="sd002"
    CORRECTOR_K=4
    CORRECTOR_ALPHA=0.5
    EXTRA_CORRECTOR_ARGS="--corrector_dtheta_max 1.0 --corrector_dvm_frac 0.3"
    ;;
  *)
    echo "Unknown array task: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

PROJECT_DIR="/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC"
SOURCE_PARQUET="/home/hpc/b313dc/b313dc11/pb_data/GBnetwork.parquet"
LOCAL_PARQUET="${TMPDIR:?}/GBnetwork.parquet"
LOG_DIR="${PROJECT_DIR}/results/logs/gb_corrector_20260912"
CKPT_DIR="${PROJECT_DIR}/results/ckpt/gb_corrector_20260912"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"
cp "${SOURCE_PARQUET}" "${LOCAL_PARQUET}"
test "$(stat -c %s "${LOCAL_PARQUET}")" = "$(stat -c %s "${SOURCE_PARQUET}")"

srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test_lumina.py \
  --PARQUET "${LOCAL_PARQUET}" --run_name "${RUN_NAME}" \
  --log_to_file --log_dir "${LOG_DIR}" --ckpt_dir "${CKPT_DIR}" \
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 \
  --BATCH 32 --EPOCHS 120 --LR 3e-4 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 --split_seed 42 \
  --task pf --init_mode scratch --physics_loss_form logcosh \
  --model_config "${PROJECT_DIR}/lumina_ckpt/lumina_config_do0.json" \
  --mask_known_v --mse_weight 1.0 --physics_weight 1e-2 \
  --bus_physics_features --bus_residual_features --theta_anchor start \
  --init_recipe "${INIT_RECIPE}" --corrector_K "${CORRECTOR_K}" \
  --corrector_alpha "${CORRECTOR_ALPHA}" ${EXTRA_CORRECTOR_ARGS}
