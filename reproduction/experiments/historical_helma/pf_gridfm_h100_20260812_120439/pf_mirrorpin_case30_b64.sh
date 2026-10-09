#!/bin/bash -l
#SBATCH --job-name=pf_mirrorpin_case30_b64
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/pf_mirrorpin_case30_b64.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/pf_mirrorpin_case30_b64.err
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC

echo "[gpu] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "[grid] case30 (30 buses), batch 64"

# Training reads randomly; NFS random reads are slow, node-local NVMe is not.
LOCAL_PARQUET="${TMPDIR}/case30.parquet"
cp "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/pf/case30_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet" "${LOCAL_PARQUET}"
# The $HOME copy has done its job now that the data is on local disk. Dropping
# it here is what lets the dispatcher start the next grid: $HOME holds only
# ~42 GB against an 86 GB corpus. The master copy stays in vault, so this is
# recoverable -- but a requeued job would not find the file, which is why these
# run on h100 rather than the preemptible partition.
rm -f "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/pf/case30_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet"

srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test_gridfm.py \
  --PARQUET "${LOCAL_PARQUET}" --task pf --gridfm_impl mirror --pin_known \
  --run_name pf_mirrorpin_case30_b64 \
  --log_to_file --log_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/pf_gridfm_h100_20260812_120439 --ckpt_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pf_gridfm_h100_20260812_120439 \
  --PER_UNIT --target_S_base 1e8 --share_grid --lazy_parquet --row_group_cache_size 4 \
  --dataset_complex_dtype complex128 \
  --BATCH 64 --EPOCHS 40 --LR 5e-4 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 \
  --mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh \
  --hidden_size 48 --num_layers 12 --n_heads 8 \
  --zero_init_head --vn_feature_mode log --feature_transform signed_log
