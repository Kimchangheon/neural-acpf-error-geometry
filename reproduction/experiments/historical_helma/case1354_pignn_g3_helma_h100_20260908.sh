#!/bin/bash -l
# Fresh 40-epoch PIGNN-G3 run on case1354pegase.
# The data split and core solver/training setup match the ppNR-v2 campaign;
# G3 adds the published global-context sidecar and its electrical-scale
# residual feature conditioning.  It is a new G3 run, not a resumed baseline.
#SBATCH --job-name=c1354_pignn_g3
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/case1354_pignn_g3_20260908/logs/%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/case1354_pignn_g3_20260908/logs/%j.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00

set -euo pipefail
trap 'rc=$?; echo "[failure] line=${LINENO} command=${BASH_COMMAND} rc=${rc}" >&2; exit "${rc}"' ERR

BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/case1354_pignn_g3_20260908
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
# Helma compute nodes do not mount the Alex vault path.  This immutable copy
# is staged and byte-verified under the Helma home filesystem before submit.
DATA=/home/hpc/b313dc/b313dc11/data_staging/case1354pegase_ppnr_v2_37038.parquet
OUT=${BASE}/results/case1354_pignn_g3_20260908
RUN=pignn_global_case1354pegase_g3_s42
LOCAL_PARQUET=${TMPDIR:?TMPDIR is required}/case1354pegase.parquet

test -r "${DATA}"; test -x "${PY}"
test -r "${OVERLAY}/train_valid_test.py"
test -r "${OVERLAY}/GNSMsg_SelfAttention_armijo.py"
mkdir -p "${OUT}/logs" "${OUT}/ckpt"

cp "${DATA}" "${LOCAL_PARQUET}"
test "$(stat -c %s "${LOCAL_PARQUET}")" = "$(stat -c %s "${DATA}")"

export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${BASE}"

echo "[campaign] ${RUN}"
echo "[data] case1354pegase ppNR-v2; rows=37038; seed=42; split_seed=42"
echo "[model] PIGNN-G3: direct iterative PF, local attention + attn_post global context"
echo "[core] d=4 d_hi=24 heads=8 local_layers=8 K=40 batch=32 epochs=40"
echo "[objective] mse_weight=5 physics_weight=1 logcosh graph; residual=signed_log"
echo "[code] $(sha256sum "${OVERLAY}/train_valid_test.py" "${OVERLAY}/GNSMsg_SelfAttention_armijo.py" | tr '\n' ' ')"
echo "[gpu] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

srun "${PY}" -u "${OVERLAY}/train_valid_test.py" \
  --PARQUET "${LOCAL_PARQUET}" \
  --run_name "${RUN}" --log_to_file --log_dir "${OUT}/logs" --ckpt_dir "${OUT}/ckpt" \
  --mode train_valid_test --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 \
  --preload_ram --preload_test --BATCH 32 --EPOCHS 40 --LR 1e-5 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 --split_seed 42 \
  --BLOCK_DIAG --PINN --model GNSMsg_EdgeSelfAttn --d 4 --d_hi 24 \
  --n_heads 8 --num_attn_layers 8 --K 40 --solver_update_mode direct \
  --use_armijo --armijo_mode geometric_safe --vlimit --preserve_zero_heads \
  --mse_weight 5.0 --physics_weight 1.0 --physics_loss_form logcosh \
  --physics_residual_norm graph --lr_scheduler CosineAnnealingLR \
  --residual_feature_norm signed_log --edge_feature_norm none \
  --global_context_mode attn_post --global_context_gate_mode scalar_tanh \
  --exact_physics_weight 0
