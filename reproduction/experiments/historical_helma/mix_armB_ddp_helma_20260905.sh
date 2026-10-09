#!/bin/bash -l
# Does training on N-1 alongside the intact grid help on N-2?
#
# Two arms of equal size, both scored afterwards on the same held-out N-2 corpus:
#   armA  control + a second control block   (no outages in training)
#   armB  control + N-1                      (single outages in training)
# Equal row counts matter: the G3 reference was trained on 37k rows and these are
# ~7.6k, so a comparison against it would confound mixing with data volume.  A
# against B does not.
#
# share_grid/share_ybus are OFF in BOTH arms.  Arm B needs it off because Y
# differs row to row; arm A could cache, but then the two arms would differ in
# the data path as well as in the data, so it is disabled there too.  This costs
# a lot of speed, which is why this runs on 4 GPUs.
#SBATCH --job-name=mix_armB
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/mix_n1_20260905/armB/slurm/%x_%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/mix_n1_20260905/armB/slurm/%x_%j.err
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

set -euo pipefail
trap 'rc=$?; echo "[failure] line=${LINENO} command=${BASH_COMMAND} rc=${rc}" >&2; exit "${rc}"' ERR

BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/overlays/mix_n1_20260905
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
DATA=/home/hpc/b313dc/b313dc11/data_staging/mix_n1_20260905/armB_control_plus_n1.parquet
CAMP=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/mix_n1_20260905/armB
RUN=pignn_g3_mix_armB_s42

test -r "${DATA}"; test -x "${PY}"; test -r "${OVERLAY}/train_valid_test.py"
mkdir -p "${CAMP}/logs" "${CAMP}/ckpt" "${CAMP}/slurm"

export PYTHONPATH=${OVERLAY}:${BASE}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
cd "${BASE}"
echo "[campaign] ${RUN}  arm=armB  data=$(basename ${DATA})"
echo "[rows] $(${PY} -c "import pyarrow.parquet as pq;print(pq.ParquetFile('${DATA}').metadata.num_rows)")"
echo "[note] share_grid OFF in both arms; topology varies within arm B"
echo "[gpu] $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x ${SLURM_NTASKS_PER_NODE}"

srun --kill-on-bad-exit=1 "${PY}" -u "${OVERLAY}/train_valid_test.py" \
  --DDP --ddp_timeout_hours 3 \
  --PARQUET "${DATA}" \
  --run_name "${RUN}" --log_to_file --log_dir "${CAMP}/logs" --ckpt_dir "${CAMP}/ckpt" \
  --mode train_valid_test --PER_UNIT --target_S_base 1e8 \
  --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 \
  --BATCH 6 --EPOCHS 40 --LR 1e-5 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 --split_seed 42 \
  --BLOCK_DIAG --PINN --model GNSMsg_EdgeSelfAttn --d 4 --d_hi 24 \
  --n_heads 8 --num_attn_layers 8 --K 40 --solver_update_mode direct \
  --use_armijo --armijo_mode geometric_safe --vlimit --preserve_zero_heads \
  --mse_weight 5.0 --physics_weight 1.0 --physics_loss_form logcosh \
  --physics_residual_norm graph --lr_scheduler CosineAnnealingLR \
  --residual_feature_norm signed_log --edge_feature_norm none \
  --global_context_mode attn_post --exact_physics_weight 0
