#!/bin/bash -l
#SBATCH --job-name=ctrl_graphkit_lr1e4
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/ctrl_graphkit_lr1e4.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/ctrl_graphkit_lr1e4.err
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
LOCAL_PARQUET="${TMPDIR:?}/GBnetwork.parquet"
# NOTE: the original v2 scripts deleted the source parquet after copying it.
# That is not reproduced here -- the corpus is shared and must survive.
cp "/home/hpc/b313dc/b313dc11/pb_data/GBnetwork.parquet" "${LOCAL_PARQUET}"
test "$(stat -c %s "${LOCAL_PARQUET}")" = "$(stat -c %s "/home/hpc/b313dc/b313dc11/pb_data/GBnetwork.parquet")"

# Controlled repeat of pfv2_graphkit_GBnetwork_b52 with ONE change: LR 5e-4 -> 1e-4.
# The v2 run never descended monotonically -- valid |V| RMSE went 1.719, 0.041,
# 0.059, 0.050, 0.035, 0.036, 0.033, 0.035 across 40 epochs, oscillating to the
# end.  That is the signature of too large a step for a 12-layer model, so the
# step is the variable under test.  Everything else is copied verbatim.
srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test_gridfm.py \
  --PARQUET "${LOCAL_PARQUET}" \
  --run_name ctrl_graphkit_lr1e4 \
  --log_to_file --log_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/ctrl_20260906 --ckpt_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/ctrl_20260906 \
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 \
  --dataset_complex_dtype complex128 \
  --BATCH 52 --EPOCHS 40 --LR 1e-4 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 \
  --task pf --gridfm_impl graphkit --hidden_size 48 --num_layers 12 --n_heads 8 --zero_init_head --vn_feature_mode log --feature_transform signed_log --mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh
