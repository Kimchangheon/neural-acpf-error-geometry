#!/bin/bash -l
# Same recipe as pignn_e120_lr1e5; only the verified fast-path flags differ.
#SBATCH --job-name=pignn_ctx_async
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/pignn_gc_chunked_async_e120_lr1e5_20260914/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/pignn_gc_chunked_async_e120_lr1e5_20260914/logs/%A_%a.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY="$BASE/overlays/fastpath_retrain_20260913"
OUT="$BASE/results/pignn_gc_chunked_async_e120_lr1e5_20260914"
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
SEEDS=(41 42 43)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
LOCAL_PARQUET="${TMPDIR:?}/GBnetwork.parquet"
DATA=/home/hpc/b313dc/b313dc11/pb_data/GBnetwork.parquet
cp "$DATA" "$LOCAL_PARQUET"
test "$(stat -c %s "$LOCAL_PARQUET")" = "$(stat -c %s "$DATA")"
mkdir -p "$OUT/logs" "$OUT/ckpt"
export PYTHONPATH="$OVERLAY:$BASE:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE"
echo "[fast-path] chunked_global_context=1 async_candidates=1 multi_rhs=0"
echo "[code] $(sha256sum "$OVERLAY/train_valid_test.py" "$OVERLAY/GNSMsg_SelfAttention_armijo.py" | tr '\n' ' ')"
srun "$PY" -u "$OVERLAY/train_valid_test.py" \
  --PARQUET "$LOCAL_PARQUET" \
  --run_name "pignn_gc_chunked_async_GBnetwork_s${SEED}_e120_lr1e5" --log_to_file --log_dir "$OUT/logs" --ckpt_dir "$OUT/ckpt" \
  --mode train_valid_test --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 \
  --preload_ram --preload_test --BATCH 23 --EPOCHS 120 --LR 1e-5 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value "$SEED" --split_seed 42 \
  --BLOCK_DIAG --PINN --model GNSMsg_EdgeSelfAttn --d 4 --d_hi 24 \
  --n_heads 8 --num_attn_layers 8 --K 40 --solver_update_mode direct \
  --use_armijo --armijo_mode geometric_safe --vlimit --preserve_zero_heads \
  --mse_weight 5.0 --physics_weight 1.0 --physics_loss_form logcosh \
  --physics_residual_norm graph --lr_scheduler default \
  --residual_feature_norm signed_log --edge_feature_norm none \
  --global_context_mode attn_post --global_context_packed_fastpath \
  --armijo_batched_candidates --exact_physics_weight 0
