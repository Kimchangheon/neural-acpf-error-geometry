#!/bin/bash -l
#SBATCH --job-name=pignn_oegr_gate
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/grl_oegr_gate_20260825/job_out/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/grl_oegr_gate_20260825/job_out/%A_%a.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-17%8

set -euo pipefail
trap 'rc=$?; echo "[failure] line=${LINENO} command=${BASH_COMMAND} rc=${rc}" >&2; exit "${rc}"' ERR

BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/grl_oegr_gate_20260825
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
CAMPAIGN=${BASE}/results/grl_oegr_gate_20260825
LOG_DIR=${CAMPAIGN}/logs
CKPT_DIR=${CAMPAIGN}/ckpt

grids=(case1354pegase GBnetwork)
files=(
  /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/pf/case1354pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37038_NR_branchrows_directSI.parquet
  /home/hpc/b313dc/b313dc11/data_staging/grl_oegr_gate_20260825/GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet
)
batches=(32 23)
variants=(signedlog B3_equilibrated B4_equilibrated_stiffness)
seeds=(41 42 43)

task=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
grid_idx=$((task / 9))
within_grid=$((task % 9))
variant_idx=$((within_grid / 3))
seed_idx=$((within_grid % 3))
if (( grid_idx < 0 || grid_idx >= 2 || variant_idx < 0 || variant_idx >= 3 )); then
  echo "invalid task=${task}" >&2
  exit 2
fi

grid=${grids[$grid_idx]}
source_file=${files[$grid_idx]}
batch=${batches[$grid_idx]}
variant=${variants[$variant_idx]}
seed=${seeds[$seed_idx]}

rfeat=signed_log
efeat=none
stiffness_args=()
case "${variant}" in
  signedlog)
    ;;
  B3_equilibrated)
    rfeat=ybus
    efeat=diagonal
    ;;
  B4_equilibrated_stiffness)
    rfeat=ybus
    efeat=diagonal
    stiffness_args=(--relative_stiffness_feature)
    ;;
  *) echo "unknown variant=${variant}" >&2; exit 3 ;;
esac

epochs=40
sample_args=()
prefix=full
if [[ ${SMOKE:-0} == 1 ]]; then
  epochs=1
  sample_args=(--max_train_samples 64 --max_valid_samples 16 --max_test_samples 16)
  prefix=smoke
fi

run_name=pignn_oegr_${prefix}_${grid}_${variant}_s${seed}
local_parquet=${TMPDIR:?TMPDIR is required}/${grid}.parquet

echo "[preflight] task=${task} grid=${grid} variant=${variant} seed=${seed} tmpdir=${TMPDIR}"
test -r "${source_file}" || { echo "unreadable source: ${source_file}" >&2; exit 10; }
test -x "${PY}" || { echo "unusable python: ${PY}" >&2; exit 11; }
test -f "${OVERLAY}/train_valid_test.py" || { echo "missing overlay: ${OVERLAY}/train_valid_test.py" >&2; exit 12; }
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"
echo "[stage] copying $(stat -c %s "${source_file}") bytes to ${local_parquet}"
cp "${source_file}" "${local_parquet}"
test "$(stat -c %s "${local_parquet}")" = "$(stat -c %s "${source_file}")"

export PYTHONPATH=${OVERLAY}:${BASE}:${PYTHONPATH:-}
cd "${BASE}"

echo "[campaign] task=${task} grid=${grid} variant=${variant} model_seed=${seed} split_seed=42 smoke=${SMOKE:-0}"
echo "[controlled] H100 K=40 d=4 d_hi=24 heads=8 layers=8 epochs=${epochs} batch=${batch} lr=1e-5 cosine"
echo "[objective] PINN mse_weight=5 physics_weight=1 form=logcosh residual_norm=graph exact_physics_weight=0"
echo "[solver] direct geometric_safe Armijo vlimit=on"
echo "[representation] residual=${rfeat} edge=${efeat} relative_stiffness=$([[ ${#stiffness_args[@]} -gt 0 ]] && echo on || echo off)"
echo "[data] source=${source_file} bytes=$(stat -c %s "${source_file}")"
echo "[code] $(sha256sum "${OVERLAY}/train_valid_test.py" "${OVERLAY}/GNSMsg_SelfAttention_armijo.py" | tr '\n' ' ')"
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
  --train_ratio 0.3333 --valid_ratio 0.3333 \
  --seed_value "${seed}" --split_seed 42 \
  "${sample_args[@]}" \
  --BLOCK_DIAG --PINN --model GNSMsg_EdgeSelfAttn \
  --d 4 --d_hi 24 --n_heads 8 --num_attn_layers 8 --K 40 \
  --solver_update_mode direct --use_armijo --armijo_mode geometric_safe \
  --vlimit --mse_weight 5.0 --physics_weight 1.0 \
  --physics_loss_form logcosh --physics_residual_norm graph \
  --lr_scheduler CosineAnnealingLR \
  --residual_feature_norm "${rfeat}" --edge_feature_norm "${efeat}" \
  "${stiffness_args[@]}" \
  --exact_physics_weight 0
