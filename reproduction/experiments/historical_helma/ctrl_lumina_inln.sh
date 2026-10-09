#!/bin/bash -l
#SBATCH --job-name=ctrl_lumina_inln
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/ctrl_lumina_inln.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/ctrl_lumina_inln.err
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

# Controlled repeat of pfv2_lumina_GBnetwork_b32 with ONE change:
# --input_layernorm.  Hyperparameters are back to the v2 originals (dropout
# 0.24398, 6 layers, LR 1e-4) precisely so that normalisation is the only
# variable -- the dropout and depth arms were already run and moved the final
# |V| RMSE by 4.4% and 1.3%, which is why they are not varied again here.
#
# Why normalisation: LUMINA's HGT is Linear->ReLU with no normalisation
# anywhere in the module, while GridSFM -- which trains to 0.0109 on this same
# grid from the same RAW vn_kv -- applies nn.LayerNorm to the raw input dims
# before its input projection.  Measured on a real batch, LUMINA's bus row has
# base_kv spanning 6.6-400 (std 124.8) against every other column at 0-1.5, and
# all seven bus columns carry ZERO scenario-to-scenario variation on this fixed
# grid, so the bus embedding is dominated by a constant that says nothing about
# the scenario.
srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test_lumina.py \
  --PARQUET "${LOCAL_PARQUET}" \
  --run_name ctrl_lumina_inln \
  --log_to_file --log_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/ctrl_20260906 --ckpt_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/ctrl_20260906 \
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 \
  --dataset_complex_dtype complex128 \
  --BATCH 32 --EPOCHS 40 --LR 1e-4 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 \
  --task pf --init_mode scratch --input_layernorm \
  --mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh \
  --model_config /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/lumina_ckpt/lumina_config.json
