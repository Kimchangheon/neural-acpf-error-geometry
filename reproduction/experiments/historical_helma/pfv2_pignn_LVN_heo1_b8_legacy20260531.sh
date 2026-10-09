#!/bin/bash -l
#SBATCH --job-name=pfv2_pignn_LVN_heo1_b8_legacy20260531
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/pfv2_pignn_LVN_heo1_b8_legacy20260531.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/pfv2_pignn_LVN_heo1_b8_legacy20260531.err
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16


set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
LOCAL_PARQUET="${TMPDIR}/LVN_heo1.parquet"
cp "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/pf/LVN_heo1_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37000_NR_branchrows_directSI.parquet" "${LOCAL_PARQUET}"
true

srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test.py \
  --PARQUET "${LOCAL_PARQUET}" \
  --run_name pfv2_pignn_LVN_heo1_b8_legacy20260531 \
  --log_to_file --log_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/pfv2_pignn_D_v2fix --ckpt_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_pignn_D_v2fix \
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --dataset_complex_dtype complex64 \
  --BATCH 8 --EPOCHS 40 --LR 3e-6 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 \
  --BLOCK_DIAG --PINN --d 4 --d_hi 24 --n_heads 8 --num_attn_layers 8 --K 40 --use_armijo --armijo_mode geometric_safe --mse_weight 1.0 --physics_loss_form logcosh --vlimit --d_hi 16 --mse_weight 10.0 --physics_loss_form mse --armijo_mode fixed --lr_scheduler CosineAnnealingLR
