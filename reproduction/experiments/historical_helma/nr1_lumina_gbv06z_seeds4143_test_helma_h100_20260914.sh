#!/bin/bash -l
# LUMINA seeds 41/43: frozen eta=1 test inference/cache and NR1 scoring.
# The damping value was selected on the existing seed-42 validation run.
#SBATCH --job-name=lum_nr1_s4143_t
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr1_lumina_gbv06z_3seed_20260914/logs/test_%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr1_lumina_gbv06z_3seed_20260914/logs/test_%A_%a.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=$BASE/overlays/nr1_lumina_gbv06z_3seed_20260914
RESULT=$BASE/results/nr1_lumina_gbv06z_3seed_20260914
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
seeds=(41 43)
checkpoints=(
  "$BASE/results/lumina_gbv06z_seedrep_20260913/ckpt/s41/gbv06z_control_zerohead_s41_best.pt"
  "$BASE/results/lumina_gbv06z_seedrep_20260913/ckpt/s43/gbv06z_control_zerohead_s43_best.pt"
)
i=${SLURM_ARRAY_TASK_ID:?}
seed=${seeds[$i]}
checkpoint=${checkpoints[$i]}
lumina_args="--model_config $BASE/lumina_ckpt/lumina_config_do0.json --mask_known_v --bus_physics_features --theta_anchor start --init_recipe sd002_zerohead"
mkdir -p "$RESULT"/{logs,json,cache}
test -r "$checkpoint"
test -r "$DATA"
export PYTHONPATH="$OVERLAY:$BASE:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "[frozen protocol] seed=$seed eta=1.0 selected by the existing LUMINA seed-42 validation sweep"
srun "$PY" -u "$OVERLAY/nr1_multiprocess_baseline.py" \
  --stage test --model lumina --checkpoint "$checkpoint" --parquet "$DATA" \
  --cache-dir "$RESULT/cache/lumina_s${seed}_test" \
  --out "$RESULT/json/lumina_s${seed}_test.json" \
  --etas 1.0 --workers 16 --batch 16 --lumina-args "$lumina_args"
