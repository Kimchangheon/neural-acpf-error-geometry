#!/bin/bash -l
#SBATCH --job-name=ctrl_ophg_sage
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/ctrl_ophg_sage.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/ctrl_ophg_sage.err
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
LOCAL_PARQUET="${TMPDIR:?}/GBnetwork.parquet"
DATA=/home/hpc/b313dc/b313dc11/pb_data/GBnetwork.parquet
cp "$DATA" "${LOCAL_PARQUET}"
test "$(stat -c %s "${LOCAL_PARQUET}")" = "$(stat -c %s "$DATA")"

# NOT a LUMINA arm.  LUMINA's released architecture is the SDK's HGT, whose
# message passing is conv(x_dict, edge_index_dict): edge_attr_dict sits in its
# signature and is never read, so the branch admittances our adapter builds
# (ac_line 9 dims, transformer 11 -- exactly the widths the SDK's own
# edge_attr_dims declares) never reach it.  Five single-variable retrainings of
# HGT -- dropout, depth, input LayerNorm, and two base_kv scalings -- moved the
# final |V| RMSE by at most 5.8%, so the ceiling is not regularisation, capacity
# or input scaling.
#
# This pair asks whether it is the missing edge features.  Same SDK file, same
# constructor shape, same output heads and forward signature; only the backend
# differs.  'gat' sets edge_attr_support=True and feeds edge_attr into
# GATConv(edge_dim=...); 'sage' is the same class and wiring with the edge
# features ignored.  gat-minus-sage is therefore the effect of the edge
# features, separated from the effect of changing operator.  Reported as an
# architecture ablation, never as a LUMINA result.
srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test_lumina.py \
  --PARQUET "${LOCAL_PARQUET}" \
  --run_name ctrl_ophg_sage \
  --log_to_file --log_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/ctrl_20260906 --ckpt_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/ctrl_20260906 \
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 \
  --dataset_complex_dtype complex128 \
  --BATCH 32 --EPOCHS 40 --LR 1e-4 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 \
  --task pf --init_mode scratch --lumina_arch opfhetero_sage \
  --mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh \
  --model_config /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/lumina_ckpt/lumina_config.json
