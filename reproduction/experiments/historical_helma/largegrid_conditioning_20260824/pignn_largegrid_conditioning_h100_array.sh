#!/bin/bash -l
#SBATCH --job-name=pignn_lgcond
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/largegrid_conditioning_20260824/job_out/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/largegrid_conditioning_20260824/job_out/%A_%a.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-29%9

set -euo pipefail
trap 'rc=$?; echo "[failure] line=${LINENO} command=${BASH_COMMAND} rc=${rc}" >&2; exit "${rc}"' ERR

BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/largegrid_conditioning_20260824
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
SOURCE_ROOT=/home/hpc/b313dc/b313dc11/data_staging/largegrid_conditioning_20260824
CAMPAIGN=${BASE}/results/largegrid_conditioning_20260824
LOG_DIR=${CAMPAIGN}/logs
CKPT_DIR=${CAMPAIGN}/ckpt

grids=(case300 case1354pegase case1888rte)
files=(
  case300_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37930_NR_branchrows_directSI.parquet
  case1354pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37038_NR_branchrows_directSI.parquet
  case1888rte_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37000_NR_branchrows_directSI.parquet
)
batches=(32 32 28)
variants=(
  anchor
  rfeat_signedlog
  rfeat_ybus
  efeat_signedlog
  rfeat_signedlog_efeat_signedlog
  rfeat_ybus_efeat_signedlog
  exact_raw_w001
  exact_local_w001
  exact_local_w0001
  full_ybus_edge_exactlocal_w001
)

task=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
grid_idx=$((task / 10))
variant_idx=$((task % 10))
if (( grid_idx < 0 || grid_idx >= ${#grids[@]} || variant_idx < 0 || variant_idx >= ${#variants[@]} )); then
  echo "invalid task=${task}" >&2
  exit 2
fi

grid=${grids[$grid_idx]}
source_file=${SOURCE_ROOT}/${files[$grid_idx]}
batch=${batches[$grid_idx]}
variant=${variants[$variant_idx]}

rfeat=none
efeat=none
exact_weight=0
exact_norm=local_ybus
case "${variant}" in
  anchor) ;;
  rfeat_signedlog) rfeat=signed_log ;;
  rfeat_ybus) rfeat=ybus ;;
  efeat_signedlog) efeat=signed_log ;;
  rfeat_signedlog_efeat_signedlog) rfeat=signed_log; efeat=signed_log ;;
  rfeat_ybus_efeat_signedlog) rfeat=ybus; efeat=signed_log ;;
  exact_raw_w001) exact_weight=0.01; exact_norm=none ;;
  exact_local_w001) exact_weight=0.01; exact_norm=local_ybus ;;
  exact_local_w0001) exact_weight=0.001; exact_norm=local_ybus ;;
  full_ybus_edge_exactlocal_w001)
    rfeat=ybus
    efeat=signed_log
    exact_weight=0.01
    exact_norm=local_ybus
    ;;
  *) echo "unknown variant=${variant}" >&2; exit 3 ;;
esac

epochs=40
sample_args=()
prefix=screen
if [[ ${SMOKE:-0} == 1 ]]; then
  epochs=1
  sample_args=(--max_train_samples 64 --max_valid_samples 16 --max_test_samples 16)
  prefix=smoke
fi

run_name=pignn_lgcond_${prefix}_${grid}_${variant}_s42
local_parquet=${TMPDIR:?TMPDIR is required}/${grid}.parquet

echo "[preflight] task=${task} grid=${grid} variant=${variant} tmpdir=${TMPDIR:-unset}"
test -r "${source_file}" || { echo "unreadable source: ${source_file}" >&2; exit 10; }
test -x "${PY}" || { echo "unusable python: ${PY}" >&2; exit 11; }
test -f "${OVERLAY}/train_valid_test.py" || { echo "missing overlay: ${OVERLAY}/train_valid_test.py" >&2; exit 12; }
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"
echo "[stage] copying $(stat -c %s "${source_file}") bytes to ${local_parquet}"
cp "${source_file}" "${local_parquet}"
test "$(stat -c %s "${local_parquet}")" = "$(stat -c %s "${source_file}")"

export PYTHONPATH=${OVERLAY}:${BASE}:${PYTHONPATH:-}
cd "${BASE}"

echo "[campaign] task=${task} grid=${grid} variant=${variant} smoke=${SMOKE:-0}"
echo "[controlled] seed=42 epochs=${epochs} batch=${batch} lr=1e-5 vlimit=on pnorm=graph mse_weight=5 physics_weight=1"
echo "[ablation] residual_feature_norm=${rfeat} edge_feature_norm=${efeat} exact_weight=${exact_weight} exact_norm=${exact_norm}"
echo "[data] source=${source_file} bytes=$(stat -c %s "${source_file}")"
echo "[code] $(sha256sum "${OVERLAY}/train_valid_test.py" "${OVERLAY}/GNSMsg_SelfAttention_armijo.py" "${OVERLAY}/known_operator_pf.py" | tr '\n' ' ')"
echo "[gpu] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

srun "${PY}" -u "${OVERLAY}/train_valid_test.py" \
  --PARQUET "${local_parquet}" \
  --run_name "${run_name}" \
  --log_to_file --log_dir "${LOG_DIR}" --ckpt_dir "${CKPT_DIR}" \
  --mode train_valid_test \
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 \
  --preload_ram --preload_test \
  --BATCH "${batch}" --EPOCHS "${epochs}" --LR 1e-5 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 --split_seed 42 \
  "${sample_args[@]}" \
  --BLOCK_DIAG --PINN --model GNSMsg_EdgeSelfAttn \
  --d 4 --d_hi 24 --n_heads 8 --num_attn_layers 8 --K 40 \
  --solver_update_mode direct --use_armijo --armijo_mode geometric_safe \
  --vlimit --mse_weight 5.0 --physics_weight 1.0 \
  --physics_loss_form logcosh --physics_residual_norm graph \
  --lr_scheduler CosineAnnealingLR \
  --residual_feature_norm "${rfeat}" --edge_feature_norm "${efeat}" \
  --exact_physics_weight "${exact_weight}" \
  --exact_physics_residual_norm "${exact_norm}"
