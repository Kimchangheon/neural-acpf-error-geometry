#!/bin/bash -l
#SBATCH --job-name=smoke_pignn
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/smoke_pignn.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/smoke_pignn.err
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=16
set -uo pipefail
B=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
export PYTHONPATH=$B:${PYTHONPATH:-}
cd $B
$PY -u train_valid_test_pignn_opf.py \
  --case_name pglib_opf_case14_ieee \
  --opfdata_root /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/opfdata --num_groups 1 \
  --run_name smoke_pignn --log_dir /tmp/pg --ckpt_dir /tmp/pg \
  --validate_branch_rows \
  --BATCH 4 --EPOCHS 1 --LR 1e-3 --K 10 \
  --max_train_samples 32 --max_valid_samples 16 --max_test_samples 16 2>&1
echo "exit=$?"
