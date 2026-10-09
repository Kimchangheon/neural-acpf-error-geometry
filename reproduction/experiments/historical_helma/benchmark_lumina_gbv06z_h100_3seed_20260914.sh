#!/bin/bash -l
# GPU-synchronized Raw/C/P16/CSP16 timing for exactly the H100-rescored seeds.
#SBATCH --job-name=lum_gbv06z_h100_t
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/lumina_gbv06z_h100_3seed_inference_20260914/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/lumina_gbv06z_h100_3seed_inference_20260914/logs/%A_%a.err

set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/gbnetwork_inference_walltime_h100_20260912
OUT=${BASE}/results/lumina_gbv06z_h100_3seed_inference_20260914
DATA=${BASE}/data_staging/controlled_geometry_20260905/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
seeds=(41 42 43)
seed=${seeds[${SLURM_ARRAY_TASK_ID:?}]}
case "$seed" in
  41|43) CKPT=${BASE}/results/lumina_gbv06z_seedrep_helma_h100_20260914/ckpt/s${seed}/gbv06z_control_zerohead_h100_s${seed}_best.pt ;;
  42) CKPT=${BASE}/results/ckpt/gb_vsweep2_20260911/gbv06z_control_zerohead_best.pt ;;
esac
LARGS="--model_config ${BASE}/lumina_ckpt/lumina_config_do0.json --mask_known_v --bus_physics_features --theta_anchor start --init_recipe sd002_zerohead"
mkdir -p "${OUT}/timing" "${OUT}/logs"
test -r "${DATA}"; test -r "${CKPT}"
export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${BASE}"
srun "${PY}" -u "${OVERLAY}/benchmark_inference_walltime.py" \
  --stage benchmark --model lumina --checkpoint "${CKPT}" --parquet "${DATA}" \
  --lumina-args "${LARGS}" \
  --batch-json "${BASE}/results/gbnetwork_inference_walltime_h100_20260912/probe/lumina.json" \
  --fit-batch 16 --repeats 3 --out "${OUT}/timing/lumina_gbv06z_h100_s${seed}.json"
