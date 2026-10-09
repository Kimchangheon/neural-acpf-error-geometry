#!/bin/bash -l
#SBATCH --job-name=r2_smoke
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/r2_smoke_20260828/%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/r2_smoke_20260828/%j.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
DATA=/home/hpc/b313dc/b313dc11/data_staging/global_context_ablation_20260827/case300_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet
OUT=${BASE}/results/r2_smoke_20260828
mkdir -p "${OUT}"
cp "${DATA}" "${TMPDIR:?}/case300.parquet"
export PYTHONPATH=${BASE}:${PYTHONPATH:-}
cd "${BASE}"
# Two epochs on a 3000-row slice: enough to print the R2 block under both
# baselines and confirm the per-bus path is the one selected.
srun "${PY}" -u "${BASE}/train_valid_test.py" \
  --PARQUET "${TMPDIR}/case300.parquet" --run_name r2_smoke_case300 \
  --log_dir "${OUT}" --ckpt_dir "${OUT}" \
  --mode train_valid_test --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 \
  --BATCH 16 --EPOCHS 1 --LR 1e-5 \
  --max_train_samples 400 --max_valid_samples 400 --max_test_samples 400 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 --split_seed 42 \
  --BLOCK_DIAG --PINN --model GNSMsg_EdgeSelfAttn --d 4 --d_hi 24 \
  --n_heads 8 --num_attn_layers 4 --K 10 --solver_update_mode direct \
  --use_armijo --armijo_mode geometric_safe --vlimit --preserve_zero_heads \
  --mse_weight 5.0 --physics_weight 1.0 --physics_loss_form logcosh \
  --physics_residual_norm graph --residual_feature_norm signed_log
