#!/bin/bash -l
#SBATCH --job-name=smoke_h200_pq_lumina
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/smoke_h200_pq_lumina.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/smoke_h200_pq_lumina.err
#SBATCH --gres=gpu:1
#SBATCH --partition=h200
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
export HF_HUB_DISABLE_XET=1
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC

echo "[gpu] $(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | head -1)"
/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -c "import torch; print('[torch]', torch.__version__, 'cuda', torch.version.cuda, 'device', torch.cuda.get_device_name(0), 'cc', torch.cuda.get_device_capability(0))"

srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test_lumina.py \
  --PARQUET /home/vault/b313dc/b313dc11/data/case14_opf_task_ready.parquet --task opf \
  --run_name smoke_h200_pq_lumina \
  --log_to_file --log_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/helma_smoke_20260807_232347 --ckpt_dir /home/vault/b313dc/b313dc11/results/ckpt/helma_smoke_20260807_232347 \
  --PER_UNIT --target_S_base 1e8 --share_grid --lazy_parquet --row_group_cache_size 2 \
  --dataset_complex_dtype complex128 \
  --BATCH 4 --EPOCHS 1 --LR 1e-4 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 \
  --max_train_samples 128 --max_valid_samples 64 --max_test_samples 64 \
  --mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh \
  --model_config /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/LUMINA/checkpoints/lumina_config.json --init_mode scratch --treat_voltage_mismatch_as_transformer
