#!/bin/bash -l
#SBATCH --job-name=heo1_micro40
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/heo1_micro40_%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/heo1_micro40_%A_%a.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-2%3
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
DATA=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/nrtraj/LVN_heo1_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_nrtrajv1_siNR_37000_NR_branchrows_directSI.parquet
SEEDS=(42 123 2026); SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
GROUP=lvn_heo1_nrtraj_micro40
mkdir -p "$BASE/results/logs/$GROUP" "$BASE/results/ckpt/$GROUP"
test -s "$DATA"; scratch=${TMPDIR}/${SLURM_JOB_ID}; mkdir -p "$scratch"
local=$scratch/$(basename "$DATA"); cp "$DATA" "$local"
test "$(stat -c %s "$DATA")" = "$(stat -c %s "$local")"
export PYTHONPATH=$BASE:${PYTHONPATH:-}; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$BASE"
name=heo1_micro40_pilot256_s${SEED}
echo "[design] 4 custom-NR macro transitions x 10 polar micro-targets = 40 supervised steps"
echo "[labels] custom-NR intermediates; pandapower V_newton terminal authority"
srun "$PY" -u train_pignn_newton_trajectory.py \
 --PARQUET "$local" --output_dir "$BASE/results/ckpt/$GROUP" --run_name "$name" \
 --epochs 200 --batch_size 4 --lr 1e-4 --weight_decay 1e-3 --seed "$SEED" --split_seed 42 \
 --train_ratio 0.70 --valid_ratio 0.15 --max_train_samples 256 --max_valid_samples 64 --max_test_samples 64 \
 --num_workers 0 --dataset_complex_dtype complex128 --target_S_base 1e8 --lazy_parquet --share_grid --row_group_cache_size 4 \
 --d 4 --d_hi 24 --n_heads 8 --num_attn_layers 8 --residual_feature_norm ybus --edge_feature_norm signed_log \
 --dtheta_max 0.30 --dvm_frac 0.10 --step_target_norm per_graph --micro_steps_per_transition 10 \
 --rollout_steps 40 --rollout_loss_weight 0.25 --terminal_pp_loss_weight 0.25 --grad_clip 1.0 --val_every 1 \
 2>&1 | tee "$BASE/results/logs/$GROUP/${name}.log"
