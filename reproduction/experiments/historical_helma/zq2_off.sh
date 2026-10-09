#!/bin/bash -l
#SBATCH --job-name=zq2_off
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/zq2_off.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/zq2_off.err
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
# helma compute nodes see only /home/hpc, so the parquet is staged there first.
cp "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/pf/LVN_heo1_ppcY_backbone_dc_compile_cNR_ls0.60-1.40_u0clean_siNR_36000_NR_branchrows_directSI.parquet" "${TMPDIR}/p.parquet"
srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test_gridfm.py \
  --PARQUET "${TMPDIR}/p.parquet" --task pf --gridfm_impl mirror \
  --run_name zq_off_LVN_heo1_b64 \
  --log_to_file --log_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/pf_zq_probe2_203940 --ckpt_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pf_zq_probe2_203940 \
  --PER_UNIT --target_S_base 1e8 --share_grid --lazy_parquet --row_group_cache_size 4 \
  --dataset_complex_dtype complex128 \
  --BATCH 64 --EPOCHS 12 --LR 5e-4 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 \
  --mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh \
  --hidden_size 48 --num_layers 12 --n_heads 8 \
  --zero_init_head --vn_feature_mode log --feature_transform signed_log 
