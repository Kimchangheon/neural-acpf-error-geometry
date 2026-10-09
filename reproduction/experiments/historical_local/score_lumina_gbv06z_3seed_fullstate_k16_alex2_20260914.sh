#!/bin/bash -l
# Score the matched three-seed gbv06z control family; no training.
#SBATCH --job-name=lum_gbv06z_full16
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/lumina_gbv06z_3seed_fullstate_csp_k16_20260914/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/lumina_gbv06z_3seed_fullstate_csp_k16_20260914/logs/%A_%a.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/lumina_gbv06z_3seed_fullstate_csp_k16_20260914
OUT=${BASE}/results/lumina_gbv06z_3seed_fullstate_csp_k16_20260914
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
seeds=(41 42 43)
seed=${seeds[${SLURM_ARRAY_TASK_ID:?}]}
case "$seed" in
  41|43) CKPT=${BASE}/results/lumina_gbv06z_seedrep_20260913/ckpt/s${seed}/gbv06z_control_zerohead_s${seed}_best.pt ;;
  42) CKPT=${BASE}/results/ckpt/gb_vsweep2_20260911/gbv06z_control_zerohead_best.pt ;;
esac
LARGS='--model_config /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/lumina_ckpt/lumina_config_do0.json --mask_known_v --bus_physics_features --theta_anchor start --init_recipe sd002_zerohead'
mkdir -p "${OUT}/json" "${OUT}/logs"
test -r "${DATA}"; test -r "${CKPT}"; test -r "${OVERLAY}/output_intervention_metrics.py"
export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${BASE}"
echo "[checkpoint] seed=${seed} ${CKPT}"
echo "[protocol] fixed split seed=42; train-only calibration/SVD basis; full-state P16/CSP16"
srun "${PY}" -u "${OVERLAY}/output_intervention_metrics.py" \
 --model lumina --checkpoint "${CKPT}" --parquet "${DATA}" --rank 16 --batch 16 \
 --projection-variants full --lumina-args "${LARGS}" \
 --out "${OUT}/json/lumina_gbv06z_s${seed}_fullstate_csp_k16.json"
