#!/bin/bash -l
# Final test-only evaluation of the saved RANGE-PIGNN checkpoints.
#SBATCH --job-name=range_test
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/range_pignn_20260827/test_out/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/range_pignn_20260827/test_out/%A_%a.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3

set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/range_pignn_20260827
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
SOURCE=/home/hpc/b313dc/b313dc11/data_staging/grl_oegr_gate_20260825/GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet
CAMPAIGN=${BASE}/results/range_pignn_20260827
masters=(1 4 8)
task=${SLURM_ARRAY_TASK_ID:?}
num_masters=${masters[$task]}
run_name=pignn_range_GBnetwork_m${num_masters}_s42
local_parquet=${TMPDIR:?}/GBnetwork.parquet

mkdir -p "${CAMPAIGN}/test_out"
cp "${SOURCE}" "${local_parquet}"
export PYTHONPATH=${OVERLAY}:${BASE}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${BASE}"

srun "${PY}" -u "${OVERLAY}/train_valid_test.py" \
  --PARQUET "${local_parquet}" --run_name "${run_name}" \
  --log_dir "${CAMPAIGN}/logs" --ckpt_dir "${CAMPAIGN}/ckpt" \
  --mode test --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 \
  --preload_ram --preload_test --BATCH 23 --EPOCHS 40 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 --split_seed 42 \
  --BLOCK_DIAG --PINN --model GNSMsg_EdgeSelfAttn \
  --d 4 --d_hi 24 --n_heads 8 --num_attn_layers 8 --K 40 \
  --solver_update_mode direct --use_armijo --armijo_mode geometric_safe \
  --vlimit --preserve_zero_heads --mse_weight 5.0 --physics_weight 1.0 \
  --physics_loss_form logcosh --physics_residual_norm graph \
  --residual_feature_norm signed_log --edge_feature_norm none \
  --global_context_mode range_post --range_num_masters "${num_masters}" \
  --range_master_dim 24 --range_num_heads 8 \
  --range_positional_encoding hop_slack_rbf --range_pe_dim 10 \
  --range_step_diagnostic_batches 4 --exact_physics_weight 0
