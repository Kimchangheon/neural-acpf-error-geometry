#!/bin/bash -l
#SBATCH --job-name=pf_lumina_case6495rte_b32
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/pf_lumina_case6495rte_b32.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/pf_lumina_case6495rte_b32.err
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
LOCAL_PARQUET="${TMPDIR}/case6495rte.parquet"
cp "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/pf/case6495rte_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_23332_NR_branchrows_directSI_rg20.parquet" "${LOCAL_PARQUET}"
rm -f "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/pf/case6495rte_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_23332_NR_branchrows_directSI_rg20.parquet"

srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test_lumina.py \
  --PARQUET "${LOCAL_PARQUET}" --task pf \
  --run_name pf_lumina_case6495rte_b32 \
  --log_to_file --log_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/pf_lumina_h100_fix_large --ckpt_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pf_lumina_h100_fix_large \
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 \
  --dataset_complex_dtype complex128 \
  --init_mode scratch \
  --BATCH 32 --EPOCHS 40 --LR 1e-4 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 \
  --mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh \
  --model_config /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/lumina_ckpt/lumina_config.json
