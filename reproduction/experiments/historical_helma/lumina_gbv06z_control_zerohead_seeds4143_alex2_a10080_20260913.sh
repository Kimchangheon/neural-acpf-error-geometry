#!/bin/bash -l
# LUMINA gbv06z_control_zerohead model-seed replicas.
# Same data split and hyperparameters as the seed-42 control; separate output
# directories prevent any existing checkpoint from being overwritten.
#SBATCH --job-name=lum_gbv06z_s
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/lumina_gbv06z_seedrep_20260913/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/lumina_gbv06z_seedrep_20260913/logs/%A_%a.err

set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OUT=${BASE}/results/lumina_gbv06z_seedrep_20260913
DATA=/home/hpc/b313dc/b313dc11/pb_data/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
CONFIG=${BASE}/lumina_ckpt/lumina_config_do0.json
OVERLAY=${BASE}/overlays/lumina_gbv06z_seedrep_20260913
seeds=(41 43)
seed=${seeds[${SLURM_ARRAY_TASK_ID:?}]}
RUN=gbv06z_control_zerohead_s${seed}
CKPT_DIR=${OUT}/ckpt/s${seed}
LOG_DIR=${OUT}/training_logs/s${seed}
mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${OUT}/logs"
test -r "${DATA}"; test -r "${CONFIG}"; test -r "${OVERLAY}/train_valid_test_lumina.py"
export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${BASE}"
echo "[protocol] gbv06z_control_zerohead; model_seed=${seed}; split_seed=42; epochs=120"
echo "[outputs] run=${RUN} ckpt_dir=${CKPT_DIR} log_dir=${LOG_DIR}"
srun "${PY}" -u "${OVERLAY}/train_valid_test_lumina.py" \
  --PARQUET "${DATA}" --run_name "${RUN}" --log_to_file \
  --log_dir "${LOG_DIR}" --ckpt_dir "${CKPT_DIR}" \
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 \
  --BATCH 32 --EPOCHS 120 --LR 1e-3 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value "${seed}" --split_seed 42 \
  --task pf --init_mode scratch --physics_loss_form logcosh \
  --model_config "${CONFIG}" --mask_known_v --mse_weight 1.0 --physics_weight 1e-2 \
  --bus_physics_features --theta_anchor start --init_recipe sd002_zerohead
