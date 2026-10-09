#!/bin/bash -l
# One representative checkpoint per family; no training and no PF scoring.
#SBATCH --job-name=gb_inf_batch_probe
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/gbnetwork_inference_walltime_h100_20260912/logs/probe_%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/gbnetwork_inference_walltime_h100_20260912/logs/probe_%A_%a.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/gbnetwork_inference_walltime_h100_20260912
OUT=${BASE}/results/gbnetwork_inference_walltime_h100_20260912
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
mkdir -p "${OUT}/probe" "${OUT}/logs"
case ${SLURM_ARRAY_TASK_ID:?} in
  0) FAMILY=pignn_gc; MODEL=g3; START=8; CAP=12288
     CKPT=${BASE}/overlays/residual_split_20260829/pignn_global_GBnetwork_g3_s42_40_best_model.ckpt; LARGS='' ;;
  1) FAMILY=gridsfm; MODEL=gridsfm; START=16; CAP=2048
     CKPT=${BASE}/overlays/residual_split_20260829/pfv2_gridsfm_GBnetwork_b26_best.pt; LARGS='' ;;
  2) FAMILY=graphkit; MODEL=graphkit; START=16; CAP=2048
     CKPT=${BASE}/results/ckpt/gk_e120_20260910/k08_s42_e120_best.pt; LARGS='' ;;
  3) FAMILY=lumina; MODEL=lumina; START=16; CAP=4096
     CKPT=${BASE}/results/ckpt/gb_vsweep2_20260911/gbv06z_control_zerohead_best.pt
     LARGS="--model_config ${BASE}/lumina_ckpt/lumina_config_do0.json --mask_known_v --bus_physics_features --theta_anchor start --init_recipe sd002_zerohead" ;;
  *) exit 2 ;;
esac
test -r "${DATA}"; test -r "${CKPT}"
export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${BASE}"
srun "${PY}" -u "${OVERLAY}/benchmark_inference_walltime.py" \
  --stage probe --model "${MODEL}" --checkpoint "${CKPT}" --parquet "${DATA}" \
  --lumina-args "${LARGS}" --probe-start "${START}" --probe-cap "${CAP}" \
  --out "${OUT}/probe/${FAMILY}.json"
