#!/bin/bash -l
#SBATCH --job-name=heo1_nrtraj
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/heo1_nrtraj_%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/heo1_nrtraj_%j.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

set -euo pipefail

BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
STAGED=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/nrtraj/LVN_heo1_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_nrtrajv1_siNR_37000_NR_branchrows_directSI.parquet
GROUP=lvn_heo1_nrtraj_v1
MODE=${MODE:-pilot}

case "${MODE}" in
  smoke)
    epochs=1; train_n=8; valid_n=4; test_n=4; time_tag=smoke
    ;;
  pilot)
    epochs=200; train_n=256; valid_n=64; test_n=64; time_tag=pilot256
    ;;
  *)
    echo "unknown MODE=${MODE}" >&2; exit 2
    ;;
esac

test -s "${STAGED}"
task_scratch=${TMPDIR}/${SLURM_JOB_ID}
local_parquet=${task_scratch}/$(basename "${STAGED}")
mkdir -p "${task_scratch}" "${BASE}/results/logs/${GROUP}" "${BASE}/results/ckpt/${GROUP}"
cp "${STAGED}" "${local_parquet}"
test "$(stat -c %s "${STAGED}")" = "$(stat -c %s "${local_parquet}")"

export PYTHONPATH=${BASE}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${BASE}"

run_name=heo1_nrtraj_${time_tag}_s42
echo "[gpu] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "[data] custom-NR intermediate states; pandapower V_newton terminal label"
echo "[split] seed=42 train/valid/test=${train_n}/${valid_n}/${test_n}"
echo "[objective] teacher=1 rollout=0.25 terminal_pp=0.25"
echo "[model] K=1 shared direct operator; Armijo off; vlimit on; residual=ybus; edge=signed_log"

srun "${PY}" -u train_pignn_newton_trajectory.py \
  --PARQUET "${local_parquet}" \
  --output_dir "${BASE}/results/ckpt/${GROUP}" \
  --run_name "${run_name}" \
  --epochs "${epochs}" --batch_size 4 --lr 1e-4 --weight_decay 1e-3 \
  --seed 42 --split_seed 42 --train_ratio 0.70 --valid_ratio 0.15 \
  --max_train_samples "${train_n}" \
  --max_valid_samples "${valid_n}" \
  --max_test_samples "${test_n}" \
  --num_workers 0 --dataset_complex_dtype complex128 --target_S_base 1e8 \
  --lazy_parquet --share_grid --row_group_cache_size 4 \
  --d 4 --d_hi 24 --n_heads 8 --num_attn_layers 8 \
  --residual_feature_norm ybus --edge_feature_norm signed_log \
  --dtheta_max 0.30 --dvm_frac 0.10 \
  --step_target_norm per_graph \
  --min_mag_step_scale 1e-4 --min_angle_step_scale 1e-4 \
  --rollout_loss_weight 0.25 --terminal_pp_loss_weight 0.25 \
  --grad_clip 1.0 --val_every 1 \
  2>&1 | tee "${BASE}/results/logs/${GROUP}/${run_name}.log"

