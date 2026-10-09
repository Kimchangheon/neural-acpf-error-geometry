#!/bin/bash -l
#SBATCH --job-name=pignn_hetero
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/pignn_hetero_%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/pignn_hetero_%A_%a.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-2%3
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
DATA_DIR=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/pignn_hetero
GRID=${GRID:?set GRID=GBnetwork or LVN_heo1}
GROUP=${GROUP:?set GROUP}
EPOCHS_N=${EPOCHS_N:-40}
TRAIN_N=${TRAIN_N:-256}
VALID_N=${VALID_N:-64}
TEST_N=${TEST_N:-64}
case "$GRID" in
 GBnetwork) BUSES=2224; FILE=GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet ;;
 LVN_heo1) BUSES=722; FILE=LVN_heo1_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37000_NR_branchrows_directSI.parquet ;;
 *) echo "unsupported GRID=$GRID" >&2; exit 2 ;;
esac
MODE=(baseline hetero hetero_exact_local)
M=${MODE[$SLURM_ARRAY_TASK_ID]}
DATA=$DATA_DIR/$FILE
mkdir -p "$BASE/results/logs/$GROUP" "$BASE/results/ckpt/$GROUP"
test -s "$DATA"
batch=$((2700 / BUSES)); ((batch < 1)) && batch=1; ((batch > 8)) && batch=8
scratch=${TMPDIR}/${SLURM_JOB_ID}; mkdir -p "$scratch"; local=$scratch/$FILE; cp "$DATA" "$local"
test "$(stat -c %s "$DATA")" = "$(stat -c %s "$local")"
export PYTHONPATH=$BASE:${PYTHONPATH:-}; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$BASE"
name=${GRID}_${M}_s42
common=(--PARQUET "$local" --run_name "$name" --log_to_file --log_dir "$BASE/results/logs/$GROUP" --ckpt_dir "$BASE/results/ckpt/$GROUP" --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 --BLOCK_DIAG --BATCH "$batch" --EPOCHS "$EPOCHS_N" --LR 1e-5 --lr_scheduler CosineAnnealingLR --VAL_EVERY 1 --train_ratio 0.3333 --valid_ratio 0.3333 --max_train_samples "$TRAIN_N" --max_valid_samples "$VALID_N" --max_test_samples "$TEST_N" --d 4 --d_hi 24 --n_heads 8 --num_attn_layers 8 --K 40 --PINN --mse_weight 5.0 --physics_weight 1.0 --physics_loss_form logcosh --physics_residual_norm graph --residual_feature_norm signed_log --vlimit --preserve_zero_heads --seed_value 42)
if [[ "$M" != baseline ]]; then common+=(--heterogeneous_injection_features --bus_type_features); fi
if [[ "$M" == hetero_exact_local ]]; then common+=(--mse_weight 1.0 --physics_weight 0.0 --exact_physics_weight 1e-3 --exact_physics_residual_norm local_ybus); fi
echo "[grid] $GRID buses=$BUSES mode=$M batch=$batch"
echo "[control] documented GB/large-grid best: graph + mw5 + cosine lr1e-5 + residual signed-log"
echo "[transfer] only typed generator/load projections (+bus role one-hot) are new; PIGNN K=40 and masks unchanged"
srun "$PY" -u train_valid_test.py "${common[@]}" 2>&1 | tee "$BASE/results/logs/$GROUP/${name}.log"
