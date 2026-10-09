#!/bin/bash -l
# Full-state (legacy) P16/CSP16 evaluation for the three independently
# trained 120-epoch GraphKit GBnetwork checkpoints.  The resumed-from-40
# checkpoint is intentionally excluded.  No known-variable restoration is
# applied: this reproduces the original table's full-state projection policy.
#SBATCH --job-name=gk_e120_full16
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/graphkit_e120_fullstate_csp_k16_20260911/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/graphkit_e120_fullstate_csp_k16_20260911/logs/%A_%a.err

set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/graphkit_e120_fullstate_csp_k16_20260911
OUT=${BASE}/results/graphkit_e120_fullstate_csp_k16_20260911
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python

seeds=(41 42 43)
stems=(k06_s41_e120 k08_s42_e120 k07_s43_e120)
i=${SLURM_ARRAY_TASK_ID:?}
seed=${seeds[$i]}
stem=${stems[$i]}
CKPT=${BASE}/results/ckpt/gk_e120_20260910/${stem}_best.pt

mkdir -p "${OUT}/json" "${OUT}/logs"
test -r "${DATA}"; test -r "${CKPT}"; test -r "${OVERLAY}/output_intervention_metrics.py"
export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
cd "${BASE}"
echo "[checkpoint] ${CKPT}"
echo "[protocol] seed-42 split; train-only calibration/bases; k=16; complex128 PB"
echo "[projection] full state only; no restore_known; no unknown-only restriction"
srun "${PY}" -u "${OVERLAY}/output_intervention_metrics.py" \
  --model graphkit --checkpoint "${CKPT}" --parquet "${DATA}" \
  --rank 16 --batch 16 --projection-variants full \
  --out "${OUT}/json/graphkit_s${seed}_e120_fullstate_csp_k16.json"
