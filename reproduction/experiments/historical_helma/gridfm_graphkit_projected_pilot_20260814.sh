#!/bin/bash -l
#SBATCH --job-name=gkproj_pf
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/gridfm_graphkit_projected_pilot_20260814_%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/gridfm_graphkit_projected_pilot_20260814_%A_%a.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-5

set -euo pipefail

BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
DATA=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/pf
GROUP=gridfm_graphkit_projected_pilot_20260814

# Two physics weights per grid reproduce the earlier projected-PIGNN ablation.
# Dataset caps and seed are identical to that pilot, while 48x12x8 and LR=5e-4
# are the released GridFM-GraphKit configuration.
GRIDS=(case24_ieee_rts case24_ieee_rts case300 case300 case1354pegase case1354pegase)
BUSES=(24 24 300 300 1354 1354)
BATCHES=(16 16 4 4 1 1)
MAX_TRAIN=(1024 1024 1024 1024 256 256)
MAX_VALID=(256 256 256 256 64 64)
MAX_TEST=(256 256 256 256 64 64)
PHYSICS_WEIGHTS=(1.0 0.01 1.0 0.01 1.0 0.01)
WEIGHT_TAGS=(w1 w001 w1 w001 w1 w001)
FILES=(
  case24_ieee_rts_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet
  case24_ieee_rts_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet
  case300_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet
  case300_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet
  case1354pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_32627_NR_branchrows_directSI_rg20.parquet
  case1354pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_32627_NR_branchrows_directSI_rg20.parquet
)

i=${SLURM_ARRAY_TASK_ID}
grid=${GRIDS[$i]}
batch=${BATCHES[$i]}
physics_weight=${PHYSICS_WEIGHTS[$i]}
source_parquet=${DATA}/${FILES[$i]}
task_scratch=${TMPDIR}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
local_parquet=${task_scratch}/${FILES[$i]}
run_name=${GROUP}_${grid}_${WEIGHT_TAGS[$i]}_b${batch}_s42

mkdir -p "${task_scratch}" "${BASE}/results/logs/${GROUP}" "${BASE}/results/ckpt/${GROUP}"
test -s "${source_parquet}"
cp "${source_parquet}" "${local_parquet}"

export PYTHONPATH=${BASE}:${PYTHONPATH:-}
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${BASE}"

echo "[gpu] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "[grid] ${grid}, buses=${BUSES[$i]}, batch=${batch}, physics_weight=${physics_weight}"
echo "[samples] ${MAX_TRAIN[$i]}/${MAX_VALID[$i]}/${MAX_TEST[$i]}, epochs=20, seed=42"
echo "[model] released GraphKit 48x12x8 with fused projected DPF operator"

srun "${PY}" -u train_valid_test_gridfm.py \
  --PARQUET "${local_parquet}" --task pf --gridfm_impl graphkit_projected \
  --run_name "${run_name}" --log_to_file \
  --log_dir "${BASE}/results/logs/${GROUP}" \
  --ckpt_dir "${BASE}/results/ckpt/${GROUP}" \
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 \
  --BATCH "${batch}" --EPOCHS 20 --LR 5e-4 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 \
  --max_train_samples "${MAX_TRAIN[$i]}" \
  --max_valid_samples "${MAX_VALID[$i]}" \
  --max_test_samples "${MAX_TEST[$i]}" \
  --hidden_size 48 --num_layers 12 --n_heads 8 --dropout 0.0 \
  --feature_transform signed_log --vn_feature_mode log \
  --mse_weight 1.0 --physics_weight "${physics_weight}" \
  --physics_loss_form logcosh --convergence_tol_pu 1e-3 \
  --armijo_max_backtracks 8 \
  --projected_dpf_lr 0.003377 \
  --projected_dpf_beta1 0.979681 --projected_dpf_beta2 0.963442 \
  --projected_dpf_mu 1e-4 \
  --projected_dpf_max_vm_fraction 0.10 \
  --projected_dpf_max_angle_step 0.30 \
  --seed_value 42
