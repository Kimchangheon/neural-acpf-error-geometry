#!/bin/bash -l
# IEEE14 GraphKit control: unnormalised supervised MSE only.
#SBATCH --job-name=c14_gk_rawmse
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/case14_graphkit_raw_mse_only_20260908/logs/%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/case14_graphkit_raw_mse_only_20260908/logs/%j.err

set -euo pipefail
trap 'rc=$?; echo "[failure] line=${LINENO} command=${BASH_COMMAND} rc=${rc}" >&2; exit "${rc}"' ERR

BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/case14_normalized_mse_only_20260908
OUT=${BASE}/results/case14_graphkit_raw_mse_only_20260908
DATA=/home/hpc/b313dc/b313dc11/data_staging/case14_normalized_mse_only_20260908/case14.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
RUN=graphkit_case14_raw_mseonly_s42_e40
LOCAL_DATA=${TMPDIR:?TMPDIR is required}/case14.parquet

test -r "${DATA}"; test -x "${PY}"
mkdir -p "${OUT}/logs" "${OUT}/ckpt"
cp "${DATA}" "${LOCAL_DATA}"
test "$(stat -c %s "${DATA}")" = "$(stat -c %s "${LOCAL_DATA}")"

export PYTHONPATH=${OVERLAY}:${BASE}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${BASE}"

echo "[protocol] GraphKit IEEE14 unnormalised supervised MSE only; seed=42 epochs=40"
echo "[paired-with] graphkit_case14_normbus_mseonly_s42_e40 (job 823360_1)"
srun "${PY}" -u "${OVERLAY}/train_valid_test_gridfm.py" \
  --PARQUET "${LOCAL_DATA}" --task pf --gridfm_impl graphkit \
  --run_name "${RUN}" --log_to_file --log_dir "${OUT}/logs" \
  --ckpt_dir "${OUT}/ckpt" --PER_UNIT --target_S_base 1e8 \
  --share_grid --share_ybus --lazy_parquet --row_group_cache_size 4 \
  --dataset_complex_dtype complex128 --preload_ram --preload_test \
  --BATCH 64 --EPOCHS 40 --LR 5e-4 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 --split_seed 42 \
  --mse_weight 1.0 --physics_weight 0.0 --correction_target_norm none \
  --hidden_size 48 --num_layers 12 --n_heads 8 --zero_init_head \
  --vn_feature_mode log --feature_transform signed_log
