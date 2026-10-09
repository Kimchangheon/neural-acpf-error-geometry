#!/bin/bash -l
#SBATCH --job-name=smoke_gk
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/smoke_gk.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/smoke_gk.err
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=16
set -uo pipefail
B=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
export PYTHONPATH=$B:${PYTHONPATH:-}
cd $B
for LOSS in shared native; do
  echo "########## graphkit GridFM, loss=$LOSS"
  $PY -u train_valid_test_opfdata.py --model gridfm --gridfm_impl graphkit \
    --init_mode scratch --loss $LOSS \
    --case_name pglib_opf_case14_ieee \
    --opfdata_root /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/opfdata --num_groups 1 \
    --run_name smoke_gk_$LOSS --log_dir /tmp/gk --ckpt_dir /tmp/gk \
    --BATCH 8 --EPOCHS 1 --LR 5e-4 \
    --max_train_samples 64 --max_valid_samples 32 --max_test_samples 32 \
    --hidden_size 48 --num_layers 12 --n_heads 8 2>&1 | tail -8
  echo "   exit=$?"
done
