#!/bin/bash -l
#SBATCH --job-name=smoke_h100_opfd_gridsfm
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/smoke_h100_opfd_gridsfm.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/smoke_h100_opfd_gridsfm.err
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
export HF_HUB_DISABLE_XET=1
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC

echo "[gpu] $(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | head -1)"
/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -c "import torch; print('[torch]', torch.__version__, 'cuda', torch.version.cuda, 'device', torch.cuda.get_device_name(0), 'cc', torch.cuda.get_device_capability(0))"

srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test_opfdata.py \
  --model gridsfm --init_mode pretrained \
  --case_name pglib_opf_case14_ieee \
  --opfdata_root /home/vault/b313dc/b313dc11/opfdata --num_groups 1 \
  --run_name smoke_h100_opfd_gridsfm \
  --log_to_file --log_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/helma_smoke_20260807_232058 --ckpt_dir /home/vault/b313dc/b313dc11/results/ckpt/helma_smoke_20260807_232058 \
  --BATCH 4 --EPOCHS 1 --LR 1e-4 \
  --max_train_samples 128 --max_valid_samples 64 --max_test_samples 64 \
  --hidden_size 24 --num_layers 3 --n_heads 4 --zero_init_head \
  --pretrained_checkpoint /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/GridSFM/checkpoints/gridsfm_open_v1.1.pt
