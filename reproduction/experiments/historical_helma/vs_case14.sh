#!/bin/bash -l
#SBATCH --job-name=vs_case14
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/vs_case14.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/vs_case14.err
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
# Score the SAME checkpoint through the streamed store. It must reproduce the
# PyG-path result (RMSE 9.6199e-03); the store is an optimisation, so any
# change in a reported digit would be a bug.
srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test_pignn_opf.py \
  --source stream --case_name pglib_opf_case14_ieee --opfdata_root /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/opfdata \
  --resume_state_dict /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/opfdata_pignn_case14_h100_20260811_195924/pignn2_case14_K10_best.pt \
  --run_name vs_case14_stream --log_to_file \
  --log_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/stream_equiv --ckpt_dir ${TMPDIR} \
  --BATCH 4 --EPOCHS 0 --LR 1e-3 --K 10 \
  --d_hi 32 --n_heads 4 --num_attn_layers 1 --armijo_mode geometric_safe \
  --mse_weight 1.0 --physics_weight 1e-2 --pinn_weight 1e-2 \
  --opf_limit_weight 1.0 --opf_band_weight 0.0
