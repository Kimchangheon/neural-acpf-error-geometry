#!/bin/bash -l
# Controlled G3 (global attention after local attention) with MSE physics loss.
# G3 with the general physics residual normalised per bus
# (--physics_residual_norm local_ybus), everything else identical to G3.
#
# Why: "graph" divides every bus by one per-graph scalar, so on a grid whose
# |Y_ii| spans five decades the normalised residuals are not commensurate.
# With log-cosh -- gradient tanh, saturated at +-1 far below the values reached
# here -- the objective then cannot distinguish 26 pu from 200 pu: every large
# residual contributes the same gradient.  That matches what was measured:
# log-cosh never fixes the tail, and switching to MSE (job 797773) blew it up
# instead, max|dP| 18.6 -> 197.8 pu.  The defect is the normalisation, not the
# loss shape.  local_ybus divides bus i by |Y_ii||V_i|^2, which is what
# dS_i/d|V_i| is proportional to, so a normalised residual of 1 means the same
# voltage-space error at every bus and the saturation point becomes physically
# meaningful.  It is the loss-side counterpart of residual_feature_norm=ybus.
#SBATCH --job-name=pignn_g3_pnlocal
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/g3_pnlocal_ablation_20260829/logs/%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/g3_pnlocal_ablation_20260829/logs/%j.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00

set -euo pipefail
trap 'rc=$?; echo "[failure] line=${LINENO} command=${BASH_COMMAND} rc=${rc}" >&2; exit "${rc}"' ERR

BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/g3_pnlocal_20260829
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
DATA=/home/hpc/b313dc/b313dc11/data_staging/grl_oegr_gate_20260825/GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet
CAMPAIGN=${BASE}/results/g3_pnlocal_ablation_20260829
LOG_DIR=${CAMPAIGN}/logs
CKPT_DIR=${CAMPAIGN}/ckpt
RUN=pignn_global_GBnetwork_g3_pnlocal_s42
LOCAL_PARQUET=${TMPDIR:?TMPDIR is required}/GBnetwork.parquet

test -r "${DATA}"; test -x "${PY}"
test -r "${OVERLAY}/train_valid_test.py"; test -r "${OVERLAY}/GNSMsg_SelfAttention_armijo.py"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"
cp "${DATA}" "${LOCAL_PARQUET}"
test "$(stat -c %s "${LOCAL_PARQUET}")" = "$(stat -c %s "${DATA}")"

export PYTHONPATH=${OVERLAY}:${BASE}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${BASE}"
echo "[campaign] ${RUN}"
echo "[common] GBnetwork rows=37022 batch=23 K=40 d=4 d_hi=24 heads=8 local_layers=8"
echo "[solver] direct geometric_safe Armijo vlimit=on preserve_zero_heads=on"
echo "[objective] mse_weight=5 mse_weight_mode=none physics_weight=1 physics_loss_form=logcosh physics_residual_norm=local_ybus"
echo "[representation] residual=signed_log edge=none global_context=attn_post"
echo "[data] seed=42 split_seed=42 target_S_base=100MVA complex128"
echo "[code] $(sha256sum "${OVERLAY}/train_valid_test.py" "${OVERLAY}/GNSMsg_SelfAttention_armijo.py" | tr '\n' ' ')"
echo "[gpu] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

srun "${PY}" -u "${OVERLAY}/train_valid_test.py" \
  --PARQUET "${LOCAL_PARQUET}" \
  --run_name "${RUN}" --log_to_file --log_dir "${LOG_DIR}" --ckpt_dir "${CKPT_DIR}" \
  --mode train_valid_test --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 \
  --preload_ram --preload_test --BATCH 23 --EPOCHS 40 --LR 1e-5 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 --split_seed 42 \
  --BLOCK_DIAG --PINN --model GNSMsg_EdgeSelfAttn --d 4 --d_hi 24 \
  --n_heads 8 --num_attn_layers 8 --K 40 --solver_update_mode direct \
  --use_armijo --armijo_mode geometric_safe --vlimit --preserve_zero_heads \
  --mse_weight 5.0 --physics_weight 1.0 --physics_loss_form logcosh  \
  --physics_residual_norm local_ybus --lr_scheduler CosineAnnealingLR \
  --residual_feature_norm signed_log --edge_feature_norm none \
  --global_context_mode attn_post --exact_physics_weight 0
