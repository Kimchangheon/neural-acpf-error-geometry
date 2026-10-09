#!/usr/bin/env bash
# Matched 120-epoch, chord-prior-free LUMINA replicas: model seeds 41/42/43.
# The existing H100 maximum-batch probe is reused because architecture and
# inference configuration are identical; this job performs no model training.
#SBATCH --job-name=lum_gbv06z_time
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/lumina_gbv06z_3seed_inference_h100_20260914/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/lumina_gbv06z_3seed_inference_h100_20260914/logs/%A_%a.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY="$BASE/overlays/gbnetwork_inference_walltime_h100_20260912"
OUT="$BASE/results/lumina_gbv06z_3seed_inference_h100_20260914"
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
SEEDS=(41 42 43)
CKPTS=(
  "$BASE/results/lumina_gbv06z_seedrep_20260913/ckpt/s41/gbv06z_control_zerohead_s41_best.pt"
  "$BASE/results/ckpt/gb_vsweep2_20260911/gbv06z_control_zerohead_best.pt"
  "$BASE/results/lumina_gbv06z_seedrep_20260913/ckpt/s43/gbv06z_control_zerohead_s43_best.pt"
)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}; CKPT=${CKPTS[$SLURM_ARRAY_TASK_ID]}
LARGS="--model_config $BASE/lumina_ckpt/lumina_config_do0.json --mask_known_v --bus_physics_features --theta_anchor start --init_recipe sd002_zerohead"
mkdir -p "$OUT/timing" "$OUT/logs"
test -r "$DATA"; test -r "$CKPT"
export PYTHONPATH="$OVERLAY:$BASE:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE"
srun --ntasks=1 --cpus-per-task=16 "$PY" -u "$OVERLAY/benchmark_inference_walltime.py" \
  --stage benchmark --model lumina --checkpoint "$CKPT" --parquet "$DATA" \
  --lumina-args "$LARGS" \
  --batch-json "$BASE/results/gbnetwork_inference_walltime_h100_20260912/probe/lumina.json" \
  --fit-batch 16 --repeats 3 \
  --out "$OUT/timing/lumina_gbv06z_s${SEED}.json"
