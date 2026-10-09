#!/bin/bash -l
#SBATCH --job-name=gk_case118_gridfm_shared
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/gk_case118_gridfm_shared.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/gk_case118_gridfm_shared.err
#SBATCH --gres=gpu:1
#SBATCH --partition=preempt
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
# Released gridfm_graphkit model, same capacity (48/12/8) as the local mirror.
srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test_opfdata.py \
  --model gridfm --gridfm_impl graphkit --init_mode scratch --loss shared \
  --case_name pglib_opf_case118_ieee --opfdata_root /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/opfdata --num_groups 1 \
  --run_name gk_case118_gridfm_shared \
  --log_to_file --log_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/opfdata_graphkit_case118_preempt_20260810_231137 --ckpt_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/opfdata_graphkit_case118_preempt_20260810_231137 \
  --BATCH 8 --EPOCHS 40 --LR 5e-4 \
  --hidden_size 48 --num_layers 12 --n_heads 8 \
  --opf_limit_weight 1.0 --opf_band_weight 0.0
