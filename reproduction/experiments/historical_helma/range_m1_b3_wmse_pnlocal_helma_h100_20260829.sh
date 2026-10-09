#!/bin/bash -l
# RANGE M=1 with the equilibrated objective.
#
# RANGE was closed as a direction on GBnetwork: M=1 landed inside the G1--G5
# band and M=4/8 collapsed to identical masters.  But every RANGE run, like
# every G run, trained against a supervised term that weights all buses equally
# in voltage space and a physics term normalised by S_abs.amax over the whole
# graph -- one scalar per grid, with log-cosh saturated far below the residuals
# reached here.  The measured consequence is that 6% of GBnetwork's buses, the
# ones above |Y_ii|=1e4, carry 20% (P) and 33% (Q) of the model's residual mass
# against 1.3%/1.7% for a predictor that ignores the scenario entirely.
#
# A persistent master token is a mechanism for moving information between
# distant parts of the grid, which is what a stiffness-localised failure would
# need; under an objective that cannot see that failure it had nothing to do.
# This run pairs RANGE M=1 against the g3_wmse_pnlocal cell -- identical except
# for global_context_mode -- so the comparison isolates the master token.
#
# This variant also carries the B3 input equilibration (residual ybus, edge
# diagonal), so all three of representation, supervision and physics use the
# same local scale.  Its control is g3_b3_wmse_pnlocal.
#SBATCH --job-name=pignn_range_m1_b3_wmse_pnlocal
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/range_m1_b3_wmse_pnlocal_ablation_20260829/logs/%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/range_m1_b3_wmse_pnlocal_ablation_20260829/logs/%j.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00

set -euo pipefail
trap 'rc=$?; echo "[failure] line=${LINENO} command=${BASH_COMMAND} rc=${rc}" >&2; exit "${rc}"' ERR

BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/range_m1_b3_wmse_pnlocal_20260829
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
DATA=/home/hpc/b313dc/b313dc11/data_staging/grl_oegr_gate_20260825/GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet
CAMPAIGN=${BASE}/results/range_m1_b3_wmse_pnlocal_ablation_20260829
LOG_DIR=${CAMPAIGN}/logs
CKPT_DIR=${CAMPAIGN}/ckpt
RUN=pignn_range_GBnetwork_m1_b3_wmse_pnlocal_s42
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
echo "[objective] mse_weight=5 mse_weight_mode=ybus physics_weight=1 physics_loss_form=logcosh physics_residual_norm=local_ybus"
echo "[representation] residual=ybus edge=diagonal (OEGR B3) global_context=range_post M=1"
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
  --physics_residual_norm local_ybus --lr_scheduler CosineAnnealingLR \
  --residual_feature_norm ybus --edge_feature_norm diagonal \
  --global_context_mode range_post \
  --range_num_masters 1 --range_master_dim 24 --range_num_heads 8 \
  --range_positional_encoding hop_slack_rbf --range_pe_dim 10 \
  --range_step_diagnostic_batches 4 --exact_physics_weight 0
