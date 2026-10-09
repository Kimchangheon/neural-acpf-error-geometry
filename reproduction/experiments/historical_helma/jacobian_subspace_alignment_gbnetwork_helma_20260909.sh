#!/bin/bash -l
# Direct reduced-AC-Jacobian alignment evidence for the train solution basis.
#SBATCH --job-name=jac_sub_align
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/jacobian_subspace_alignment_20260909/logs/%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/jacobian_subspace_alignment_20260909/logs/%j.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/jacobian_subspace_alignment_20260909
OUT=${BASE}/results/jacobian_subspace_alignment_20260909
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
mkdir -p "${OUT}/json" "${OUT}/logs"
test -r "${DATA}"; test -r "${OVERLAY}/jacobian_subspace_alignment.py"
export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
cd "${BASE}"
srun "${PY}" -u "${OVERLAY}/jacobian_subspace_alignment.py" \
 --parquet "${DATA}" --rank 16 --basis-batch 16 \
 --scenarios 128 --directions 32 --seed 20260909 \
 --out "${OUT}/json/jacobian_subspace_alignment_k16.json"
