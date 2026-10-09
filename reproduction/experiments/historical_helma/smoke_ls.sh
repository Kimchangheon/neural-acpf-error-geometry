#!/bin/bash -l
#SBATCH --job-name=smoke_ls
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/smoke_ls.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/smoke_ls.err
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
cp "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/pf/case118_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet" "${TMPDIR}/p.parquet"
for impl in graphkit graphkit_inc graphkit_ls; do
  echo "########## $impl"
  srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test_gridfm.py --PARQUET "${TMPDIR}/p.parquet" \
    --task pf --gridfm_impl $impl --run_name smoke_$impl \
    --PER_UNIT --target_S_base 1e8 --share_grid --lazy_parquet --row_group_cache_size 4 \
    --dataset_complex_dtype complex128 \
    --BATCH 16 --EPOCHS 1 --LR 5e-4 --VAL_EVERY 1 \
    --max_train_samples 320 --max_valid_samples 160 --max_test_samples 160 \
    --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 \
    --mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh \
    --hidden_size 48 --num_layers 12 --n_heads 8 \
    --zero_init_head --vn_feature_mode log --feature_transform signed_log 2>&1 | tail -18
done
