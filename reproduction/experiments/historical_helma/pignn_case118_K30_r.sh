#!/bin/bash -l
#SBATCH --job-name=pignn_case118_K30_r
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/pignn_case118_K30_r.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/pignn_case118_K30_r.err
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
# The original case118 runs were PREEMPTED at 2 h on the preempt partition
# (epoch 24 and 11 of 40). h100 is not preemptible; resume from the best
# checkpoint those runs saved rather than restarting from scratch.
srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test_pignn_opf.py \
  --case_name pglib_opf_case118_ieee --opfdata_root /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/opfdata --num_groups 1 \
  --run_name pignn_case118_K30_r \
  --log_to_file --log_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/opfdata_pignn_case118_preempt_20260811_134016 --ckpt_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/opfdata_pignn_case118_preempt_20260811_134016 \
  --resume_state_dict /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/opfdata_pignn_case118_preempt_20260811_134016/pignn_case118_K30_best.pt \
  --validate_branch_rows \
  --BATCH 4 --EPOCHS 40 --LR 1e-3 --K 30 \
  --d_hi 32 --n_heads 4 --num_attn_layers 1 --armijo_mode geometric_safe \
  --mse_weight 1.0 --physics_weight 1e-2 --pinn_weight 1e-2 \
  --opf_limit_weight 1.0 --opf_band_weight 0.0
