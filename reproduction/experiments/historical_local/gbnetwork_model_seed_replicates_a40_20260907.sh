#!/bin/bash -l
# Three-seed GBnetwork baseline protocol: fixed split seed 42, model seeds 41/43.
# Seed-42 checkpoints are the frozen 40-epoch baselines used in the paper.
# This array trains only the missing seed-41 and seed-43 replicas for GridSFM,
# GridFM-GraphKit, and LUMINA, with their original batch/configuration settings.
# PIGNN-GC replicas already exist from the same fixed-split G3 reference campaign.
#SBATCH --job-name=gb_seedrep
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-5%4
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/model_seed_replicates_20260907/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/model_seed_replicates_20260907/logs/%A_%a.err

set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/model_seed_replicates_20260907
OUT=${BASE}/results/model_seed_replicates_20260907
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
CONFIG=${BASE}/lumina_ckpt/lumina_config.json

i=${SLURM_ARRAY_TASK_ID:?}
families=(gridsfm gridfm_graphkit lumina)
seeds=(41 43)
family=${families[$((i / 2))]}
seed=${seeds[$((i % 2))]}
run="${family}_GBnetwork_s${seed}_40"
mkdir -p "${OUT}/logs" "${OUT}/ckpt/${family}"
test -r "${DATA}"; test -x "${PY}"
export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${BASE}"

common=(--PARQUET "${DATA}" --task pf --run_name "${run}" --log_to_file
  --log_dir "${OUT}/logs" --ckpt_dir "${OUT}/ckpt/${family}"
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus
  --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128
  --VAL_EVERY 1 --train_ratio 0.3333 --valid_ratio 0.3333
  --seed_value "${seed}" --split_seed 42 --mse_weight 1.0 --physics_weight 1e-2
  --physics_loss_form logcosh)

echo "[protocol] family=${family} model_seed=${seed} split_seed=42 epochs=40"
case "${family}" in
  gridsfm)
    "${PY}" -u "${OVERLAY}/train_valid_test_gridsfm.py" "${common[@]}" \
      --BATCH 26 --EPOCHS 40 --LR 1e-4 --init_mode scratch
    ;;
  gridfm_graphkit)
    "${PY}" -u "${OVERLAY}/train_valid_test_gridfm.py" "${common[@]}" \
      --BATCH 52 --EPOCHS 40 --LR 5e-4 --gridfm_impl graphkit \
      --hidden_size 48 --num_layers 12 --n_heads 8 --zero_init_head \
      --vn_feature_mode log --feature_transform signed_log
    ;;
  lumina)
    test -r "${CONFIG}"
    "${PY}" -u "${OVERLAY}/train_valid_test_lumina.py" "${common[@]}" \
      --BATCH 32 --EPOCHS 40 --LR 1e-4 --init_mode scratch \
      --model_config "${CONFIG}"
    ;;
esac
