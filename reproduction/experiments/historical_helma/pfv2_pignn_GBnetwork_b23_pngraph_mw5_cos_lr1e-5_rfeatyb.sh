#!/bin/bash -l
#SBATCH --job-name=pfv2_pignn_GBnetwork_b23_pngraph_mw5_cos_lr1e-5_rfeatyb
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/pfv2_pignn_GBnetwork_b23_pngraph_mw5_cos_lr1e-5_rfeatyb.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/pfv2_pignn_GBnetwork_b23_pngraph_mw5_cos_lr1e-5_rfeatyb.err
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16


set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
LOCAL_PARQUET="${TMPDIR}/GBnetwork.parquet"
cp "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/pf/GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet" "${LOCAL_PARQUET}"
true

srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test.py \
  --PARQUET "${LOCAL_PARQUET}" \
  --run_name pfv2_pignn_GBnetwork_b23_pngraph_mw5_cos_lr1e-5_rfeatyb \
  --log_to_file --log_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/pfv2_pignn_D_v2fix --ckpt_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_pignn_D_v2fix \
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 \
  --BATCH 23 --EPOCHS 40 --LR 1e-5 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 \
  --BLOCK_DIAG --PINN --d 4 --d_hi 24 --n_heads 8 --num_attn_layers 8 --K 40 --use_armijo --armijo_mode geometric_safe --mse_weight 1.0 --physics_loss_form logcosh --vlimit --physics_residual_norm graph --mse_weight 5.0 --lr_scheduler CosineAnnealingLR --residual_feature_norm ybus
