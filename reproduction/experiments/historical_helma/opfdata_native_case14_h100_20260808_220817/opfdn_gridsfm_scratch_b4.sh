#!/bin/bash -l
#SBATCH --job-name=opfdn_gridsfm_scratch_b4
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/opfdn_gridsfm_scratch_b4.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/opfdn_gridsfm_scratch_b4.err
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
export HF_HUB_DISABLE_XET=1
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC

srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test_opfdata.py \
  --model gridsfm --init_mode scratch --loss native \
  --case_name pglib_opf_case14_ieee \
  --opfdata_root /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/opfdata --num_groups 1 \
  --run_name opfdn_gridsfm_scratch_b4 \
  --log_to_file --log_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/opfdata_native_case14_h100_20260808_220817 --ckpt_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/opfdata_native_case14_h100_20260808_220817 \
  --BATCH 4 --EPOCHS 40 --LR 1e-4 \
  --opf_limit_weight 1.0 --opf_band_weight 0.0 \
  
