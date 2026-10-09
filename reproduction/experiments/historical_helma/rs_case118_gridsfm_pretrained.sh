#!/bin/bash -l
#SBATCH --job-name=rs_case118_gridsfm_pretrained
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/rs_case118_gridsfm_pretrained.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/rs_case118_gridsfm_pretrained.err
#SBATCH --gres=gpu:1
#SBATCH --partition=preempt
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
# Eval-only re-scoring of an already-trained checkpoint, to add the pooled
# P/Q residual statistics that the original runs did not report.
srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test_opfdata.py \
  --model gridsfm --init_mode scratch --loss native \
  --resume_state_dict /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/opfdata_native_case118_preempt_20260808_221651/opfdn_case118_gridsfm_pretrained_b4_best.pt \
  --case_name pglib_opf_case118_ieee --opfdata_root /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/opfdata --num_groups 1 \
  --run_name rs_case118_gridsfm_pretrained --log_to_file --log_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/residual_stats --ckpt_dir ${TMPDIR} \
  --BATCH 4 --EPOCHS 0 --LR 1e-4 \
  --opf_limit_weight 1.0 --opf_band_weight 0.0 
