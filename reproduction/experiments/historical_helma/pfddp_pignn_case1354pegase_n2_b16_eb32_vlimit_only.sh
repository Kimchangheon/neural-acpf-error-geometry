#!/bin/bash -l
#SBATCH --job-name=pfddp_pignn_case1354pegase_n2_b16_eb32_vlimit_only
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/pfddp_pignn_case1354pegase_n2_b16_eb32_vlimit_only.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/pfddp_pignn_case1354pegase_n2_b16_eb32_vlimit_only.err
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC

# Rendezvous for init_process_group(init_method="env://"). The port is derived
# from the job id so two DDP jobs on one node cannot collide.
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export NCCL_TIMEOUT=3600
export OMP_NUM_THREADS=16
echo "[job] ${SLURM_NNODES} nodes x 2 gpus = ${SLURM_NTASKS} ranks"
echo "[job] master ${MASTER_ADDR}:${MASTER_PORT} nodes ${SLURM_JOB_NODELIST}"

srun --kill-on-bad-exit=1 /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test.py \
  --DDP --ddp_timeout_hours 3 \
  --PARQUET "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/pf/case1354pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37038_NR_branchrows_directSI.parquet" \
  --run_name pfddp_pignn_case1354pegase_n2_b16_eb32_vlimit_only \
  --log_to_file --log_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/pfddp_pignn_ablation_20260824_case1354_vlimit_only --ckpt_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfddp_pignn_ablation_20260824_case1354_vlimit_only \
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 \
  --dataset_complex_dtype complex128 \
  --BATCH 16 --target_effective_batch 32 \
  --EPOCHS 40 --LR 1e-4 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 \
  --BLOCK_DIAG --PINN --d 4 --d_hi 24 --n_heads 8 --num_attn_layers 8 --K 40 --use_armijo --armijo_mode geometric_safe --physics_loss_form logcosh --vlimit --mse_weight 1.0 --physics_residual_norm none 
