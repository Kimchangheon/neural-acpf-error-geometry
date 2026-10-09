#!/bin/bash -l
#SBATCH --job-name=pignn_g0diag
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/grl_oegr_gate_20260825/job_out/%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/grl_oegr_gate_20260825/job_out/%j.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/grl_oegr_gate_20260825
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
SRC=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/pf/case1354pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37038_NR_branchrows_directSI.parquet
OUT=${BASE}/results/grl_oegr_gate_20260825
mkdir -p "${OUT}/logs" "${OUT}/ckpt"
test -r "${SRC}"; test -x "${PY}"; test -f "${OVERLAY}/train_valid_test.py"
LOCAL=${TMPDIR:?TMPDIR required}/case1354pegase.parquet
cp "${SRC}" "${LOCAL}"
test "$(stat -c %s "${LOCAL}")" = "$(stat -c %s "${SRC}")"
export PYTHONPATH=${OVERLAY}:${BASE}:${PYTHONPATH:-}
cd "${BASE}"
echo "[ablation] G0 residual=signed_log, edge=diagonal only, seed=42, split_seed=42"
echo "[common] K=40 d=4 d_hi=24 heads=8 layers=8 LR=1e-5 cosine vlimit direct geometric_safe"
srun "${PY}" -u "${OVERLAY}/train_valid_test.py" \
  --PARQUET "${LOCAL}" --run_name pignn_oegr_full_case1354pegase_G0diagedge_s42 \
  --log_to_file --log_dir "${OUT}/logs" --ckpt_dir "${OUT}/ckpt" --mode train_valid_test \
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus --lazy_parquet --row_group_cache_size 4 \
  --dataset_complex_dtype complex128 --preload_ram --preload_test --BATCH 32 --EPOCHS 40 --LR 1e-5 \
  --VAL_EVERY 1 --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 --split_seed 42 \
  --BLOCK_DIAG --PINN --model GNSMsg_EdgeSelfAttn --d 4 --d_hi 24 --n_heads 8 --num_attn_layers 8 --K 40 \
  --solver_update_mode direct --use_armijo --armijo_mode geometric_safe --vlimit --mse_weight 5 --physics_weight 1 \
  --physics_loss_form logcosh --physics_residual_norm graph --lr_scheduler CosineAnnealingLR \
  --residual_feature_norm signed_log --edge_feature_norm diagonal --exact_physics_weight 0
