#!/bin/bash -l
#SBATCH --job-name=pfddp_gridsfm_case9241pegase_n8_b6_eb48
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/pfddp_gridsfm_case9241pegase_n8_b6_eb48.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/pfddp_gridsfm_case9241pegase_n8_b6_eb48.err
#SBATCH --partition=h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:4
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
echo "[job] ${SLURM_NNODES} nodes x 4 gpus = ${SLURM_NTASKS} ranks"
echo "[job] master ${MASTER_ADDR}:${MASTER_PORT} nodes ${SLURM_JOB_NODELIST}"

srun --kill-on-bad-exit=1 /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test_gridsfm.py \
  --DDP --ddp_timeout_hours 3 \
  --PARQUET "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/pf/case9241pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_38508_NR_branchrows_directSI.parquet" \
  --run_name pfddp_gridsfm_case9241pegase_n8_b6_eb48 \
  --log_to_file --log_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/pfddp_gridsfm_big3 --ckpt_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfddp_gridsfm_big3 \
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 \
  --dataset_complex_dtype complex128 \
  --BATCH 6 --target_effective_batch 48 \
  --EPOCHS 300 --LR 1e-4 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 \
  --task pf --init_mode scratch --mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh --ddp_static_graph
