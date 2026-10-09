#!/bin/bash -l
# Matched IEEE case14 ablation: PIGNN-style target whitening, supervised MSE only.
#SBATCH --job-name=c14_normmse
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/case14_normalized_mse_only_20260908/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/case14_normalized_mse_only_20260908/logs/%A_%a.err

set -euo pipefail
trap 'rc=$?; echo "[failure] line=${LINENO} command=${BASH_COMMAND} rc=${rc}" >&2; exit "${rc}"' ERR

BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/case14_normalized_mse_only_20260908
OUT=${BASE}/results/case14_normalized_mse_only_20260908
DATA=/home/hpc/b313dc/b313dc11/data_staging/case14_normalized_mse_only_20260908/case14.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
CONFIG=${BASE}/lumina_ckpt/lumina_config.json

families=(gridsfm graphkit lumina)
family=${families[${SLURM_ARRAY_TASK_ID:?}]}
run=${family}_case14_normbus_mseonly_s42_e40
local_data=${TMPDIR:?TMPDIR is required}/case14.parquet

test -r "${DATA}"; test -x "${PY}"
test -r "${OVERLAY}/supervised_voltage_loss.py"
mkdir -p "${OUT}/logs" "${OUT}/ckpt/${family}"
cp "${DATA}" "${local_data}"
test "$(stat -c %s "${DATA}")" = "$(stat -c %s "${local_data}")"

export PYTHONPATH=${OVERLAY}:${BASE}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${BASE}"

common=(--PARQUET "${local_data}" --task pf --run_name "${run}" --log_to_file
  --log_dir "${OUT}/logs" --ckpt_dir "${OUT}/ckpt/${family}"
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus
  --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128
  --preload_ram --preload_test --BATCH 64 --EPOCHS 40 --VAL_EVERY 1
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 --split_seed 42
  --mse_weight 1.0 --physics_weight 0.0
  --correction_target_norm bus --correction_target_norm_eps 1e-4)

echo "[protocol] family=${family} case=IEEE14 seed=42 epochs=40"
echo "[objective] PIGNN-style per-bus/channel target whitening; supervised MSE only"
echo "[code] $(sha256sum "${OVERLAY}/supervised_voltage_loss.py" "${OVERLAY}/train_valid_test_${family/graphkit/gridfm}.py" | tr '\n' ' ')"
echo "[gpu] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

case "${family}" in
  gridsfm)
    srun "${PY}" -u "${OVERLAY}/train_valid_test_gridsfm.py" "${common[@]}" \
      --LR 1e-4 --init_mode scratch
    ;;
  graphkit)
    srun "${PY}" -u "${OVERLAY}/train_valid_test_gridfm.py" "${common[@]}" \
      --LR 5e-4 --gridfm_impl graphkit --hidden_size 48 --num_layers 12 \
      --n_heads 8 --zero_init_head --vn_feature_mode log \
      --feature_transform signed_log
    ;;
  lumina)
    test -r "${CONFIG}"
    srun "${PY}" -u "${OVERLAY}/train_valid_test_lumina.py" "${common[@]}" \
      --LR 1e-4 --init_mode scratch --model_config "${CONFIG}"
    ;;
esac
