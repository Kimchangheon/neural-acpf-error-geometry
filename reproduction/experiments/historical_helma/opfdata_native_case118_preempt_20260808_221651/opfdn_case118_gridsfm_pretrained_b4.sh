#!/bin/bash -l
#SBATCH --job-name=opfdn_case118_gridsfm_pretrained_b4
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/opfdn_case118_gridsfm_pretrained_b4.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/opfdn_case118_gridsfm_pretrained_b4.err
#SBATCH --gres=gpu:1
#SBATCH --partition=preempt
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
export HF_HUB_DISABLE_XET=1
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC

srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test_opfdata.py \
  --model gridsfm --init_mode pretrained --loss native \
  --case_name pglib_opf_case118_ieee \
  --opfdata_root /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/opfdata --num_groups 1 \
  --run_name opfdn_case118_gridsfm_pretrained_b4 \
  --log_to_file --log_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/opfdata_native_case118_preempt_20260808_221651 --ckpt_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/opfdata_native_case118_preempt_20260808_221651 \
  --BATCH 4 --EPOCHS 40 --LR 1e-4 \
  --opf_limit_weight 1.0 --opf_band_weight 0.0 \
  --pretrained_checkpoint /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/GridSFM/checkpoints/gridsfm_open_v1.1.pt
