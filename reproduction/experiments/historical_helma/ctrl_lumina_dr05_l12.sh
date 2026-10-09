#!/bin/bash -l
#SBATCH --job-name=ctrl_lumina_dr05_l12
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/ctrl_lumina_dr05_l12.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/ctrl_lumina_dr05_l12.err
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

# Controlled repeat of pfv2_lumina_GBnetwork_b32.
#   ctrl_lumina_dr05     : dropout 0.244 -> 0.05           (one variable)
#   ctrl_lumina_dr05_l12 : and num_layers 6 -> 12          (two, read against dr05)
# The v2 run stalled: valid |V| RMSE moved 0.0479 -> 0.0440 over the last 25
# epochs, and the resulting model predicts almost no scenario-to-scenario
# variation (sigma_pred 0.00157 against a reference spread of 0.02341, slope
# 0.007).  Heavy dropout on a regression head pushes exactly that way, and its
# value (0.24398355775520905) is a search artifact carried over from elsewhere.
# The 6-layer depth is the second suspect: angle error stalled at 17.2 deg
# against 4.9 (graphkit, 12 layers) and 5.2 (gridsfm).
srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test_lumina.py \
  --PARQUET "${LOCAL_PARQUET}" \
  --run_name ctrl_lumina_dr05_l12 \
  --log_to_file --log_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/ctrl_20260906 --ckpt_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/ctrl_20260906 \
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 \
  --dataset_complex_dtype complex128 \
  --BATCH 32 --EPOCHS 40 --LR 1e-4 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 \
  --task pf --init_mode scratch --mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh --model_config /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/lumina_ckpt/lumina_config_dr05_l12.json
