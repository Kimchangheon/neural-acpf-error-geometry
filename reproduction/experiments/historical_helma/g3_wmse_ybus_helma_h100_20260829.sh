#!/bin/bash -l
# Controlled G3 (global attention after local attention) with MSE physics loss.
# G3 with the supervised term weighted by |Y_ii| (--mse_weight_mode ybus),
# everything else identical to the log-cosh G3.
#
# Why: the AC residual at a bus scales roughly with |Y_ii|, so an unweighted
# ||V - V*||^2 values a 1e-3 pu error the same at a bus where it costs 1e-3
# pu of mismatch and at one where it costs 1 pu.  Weak buses outnumber stiff
# ones ~10:1 on GBnetwork, so the supervised term pulls the fit away from the
# buses that carry the residual: measured, |Y_ii|>1e4 is 6% of buses but 20%
# (P) / 33% (Q) of G3's residual mass, against 1.3% / 1.7% for a constant
# predictor that beats it on median residual.  Weighting by |Y_ii| points the
# supervised and physics terms in the same direction.
#
# Normalised to mean 1 per graph, so --mse_weight 5 stays comparable with the
# unweighted G3 run this is paired against.
#SBATCH --job-name=pignn_g3_wmse
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/g3_wmse_ablation_20260829/logs/%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/g3_wmse_ablation_20260829/logs/%j.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00

set -euo pipefail
trap 'rc=$?; echo "[failure] line=${LINENO} command=${BASH_COMMAND} rc=${rc}" >&2; exit "${rc}"' ERR

BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/g3_wmse_20260829
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
DATA=/home/hpc/b313dc/b313dc11/data_staging/grl_oegr_gate_20260825/GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet
CAMPAIGN=${BASE}/results/g3_wmse_ablation_20260829
LOG_DIR=${CAMPAIGN}/logs
CKPT_DIR=${CAMPAIGN}/ckpt
RUN=pignn_global_GBnetwork_g3_wmse_s42
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
echo "[objective] mse_weight=5 mse_weight_mode=ybus physics_weight=1 physics_loss_form=logcosh physics_residual_norm=graph"
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
  --mse_weight 5.0 --physics_weight 1.0 --physics_loss_form logcosh --mse_weight_mode ybus \
  --physics_residual_norm graph --lr_scheduler CosineAnnealingLR \
  --residual_feature_norm signed_log --edge_feature_norm none \
  --global_context_mode attn_post --exact_physics_weight 0
