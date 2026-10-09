#!/bin/bash -l
#SBATCH --job-name=pignn2_case14_K30
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/pignn2_case14_K30.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/pignn2_case14_K30.err
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
# h100, not preempt: the first attempt lost both case118 runs to preemption.
srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test_pignn_opf.py \
  --case_name pglib_opf_case14_ieee --opfdata_root /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/opfdata --num_groups 1 \
  --run_name pignn2_case14_K30 \
  --log_to_file --log_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/opfdata_pignn_case14_h100_20260811_195924 --ckpt_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/opfdata_pignn_case14_h100_20260811_195924 \
  --validate_branch_rows \
  --BATCH 4 --EPOCHS 40 --LR 1e-3 --K 30 \
  --d_hi 32 --n_heads 4 --num_attn_layers 1 --armijo_mode geometric_safe \
  --mse_weight 1.0 --physics_weight 1e-2 --pinn_weight 1e-2 \
  --opf_limit_weight 1.0 --opf_band_weight 0.0
