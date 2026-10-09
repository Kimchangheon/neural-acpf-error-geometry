#!/usr/bin/env bash
# Same scope as the earlier async benchmark: 12,344 held-out GBnetwork
# scenarios, three checkpoint seeds, three synchronized repeats, batch 6144.
# This is a speed-only run; no voltage or PB metrics are scored here.
#SBATCH --job-name=pignn_ctx_async
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/gbnetwork_inference_chunked_context_async_h100_20260913/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/gbnetwork_inference_chunked_context_async_h100_20260913/logs/%A_%a.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY="$BASE/overlays/fastpath_retrain_20260913"
OUT="$BASE/results/gbnetwork_inference_chunked_context_async_h100_20260913"
PARQUET=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
SEEDS=(41 42 43)
CKPTS=(
  "$BASE/results/g3_ref_s41_ablation_20260829/ckpt/pignn_global_GBnetwork_g3_ref_s41_40_best_model.ckpt"
  "$BASE/overlays/residual_split_20260829/pignn_global_GBnetwork_g3_s42_40_best_model.ckpt"
  "$BASE/results/g3_ref_s43_ablation_20260829/ckpt/pignn_global_GBnetwork_g3_ref_s43_40_best_model.ckpt"
)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}; CKPT=${CKPTS[$SLURM_ARRAY_TASK_ID]}
mkdir -p "$OUT/logs" "$OUT/timing"
export PYTHONPATH="$OVERLAY:$BASE:${PYTHONPATH:-}"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32
cd "$OVERLAY"
exec "$PY" "$OVERLAY/benchmark_inference_walltime.py" \
  --stage benchmark --model g3 --checkpoint "$CKPT" --parquet "$PARQUET" \
  --fit-batch 16 --repeats 3 --timing-batch-fraction 1.0 \
  --pignn-batched-armijo --pignn-global-context-packed-fastpath \
  --batch-json "$BASE/results/gbnetwork_inference_async_armijo_h100_20260913/probe_async_safe_6144.json" \
  --out "$OUT/timing/pignn_gc_s${SEED}_chunked_context_async.json"
