#!/bin/bash -l
# One GPU inference/cache job per fixed checkpoint.  The output is later used
# by CPU-only NR jobs, so neither NR solver repeats neural inference.
#SBATCH --job-name=nr_cache_gb
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --array=0-10%4
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr_warmstart_gbnetwork_20260912/logs/cache_%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr_warmstart_gbnetwork_20260912/logs/cache_%A_%a.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=$BASE/overlays/gbnetwork_inference_walltime_h100_20260912
OUT=$BASE/results/nr_warmstart_gbnetwork_20260912
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
SPEC=$BASE/nr_warmstart_specs_20260912.tsv
mkdir -p "$OUT/logs" "$OUT/cache"
mapfile -t ROWS < <(grep -v '^#' "$SPEC")
IFS=$'\t' read -r LABEL KIND RELCKPT BATCH <<< "${ROWS[$SLURM_ARRAY_TASK_ID]}"
CKPT=$BASE/$RELCKPT
test -r "$DATA"; test -r "$CKPT"
LUM_ARGS=()
if [[ "$KIND" == "lumina" ]]; then
  LUM_ARGS=(--lumina-args '--model_config /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/lumina_ckpt/lumina_config_do0.json --mask_known_v --bus_physics_features --theta_anchor start --init_recipe sd002_zerohead')
fi
export PYTHONPATH=$OVERLAY:$BASE:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE"
srun "$PY" -u cache_nr_warmstarts.py --model "$KIND" --checkpoint "$CKPT" \
  --parquet "$DATA" --out "$OUT/cache/$LABEL" --batch "$BATCH" --rank 16 "${LUM_ARGS[@]}"
