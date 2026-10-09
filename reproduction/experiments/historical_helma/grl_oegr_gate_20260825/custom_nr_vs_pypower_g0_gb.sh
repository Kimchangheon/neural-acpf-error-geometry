#!/bin/bash -l
#SBATCH --job-name=nrcompare
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/grl_oegr_gate_20260825/job_out/%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/grl_oegr_gate_20260825/job_out/%j.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC; OVERLAY=${BASE}/overlays/grl_oegr_gate_20260825
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python; SRC=/home/hpc/b313dc/b313dc11/data_staging/grl_oegr_gate_20260825/GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet
OUT=${BASE}/results/grl_oegr_gate_20260825; LOCAL=${TMPDIR:?TMPDIR required}/GBnetwork.parquet; cp "${SRC}" "${LOCAL}"
CKPT=${OUT}/ckpt/pignn_oegr_full_GBnetwork_signedlog_s42_40_best_model.ckpt; test -s "${CKPT}"; export PYTHONPATH=${OVERLAY}:${BASE}:${PYTHONPATH:-}; cd "${BASE}"
for solver in own pypower; do
  srun --exclusive "${PY}" -u "${OVERLAY}/train_valid_test.py" --PARQUET "${LOCAL}" --run_name nrcompare_g0_GBnetwork_${solver}_s42 --mode test --init_checkpoint "${CKPT}" --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 --preload_ram --preload_test --BATCH 23 --train_ratio .3333 --valid_ratio .3333 --seed_value 42 --split_seed 42 --BLOCK_DIAG --PINN --model GNSMsg_EdgeSelfAttn --d 4 --d_hi 24 --n_heads 8 --num_attn_layers 8 --K 40 --solver_update_mode direct --use_armijo --armijo_mode geometric_safe --vlimit --mse_weight 5 --physics_weight 1 --physics_loss_form logcosh --physics_residual_norm graph --residual_feature_norm signed_log --edge_feature_norm none --exact_physics_weight 0 --report_nr_polish --nr_polish_solver "${solver}" --nr_polish_tol 1e-8 --nr_polish_max_iter 30 --nr_polish_max_cases 200
done
