#!/bin/bash -l
# Alex2/A40 mirror of the direct reduced-AC-Jacobian alignment experiment.
#SBATCH --job-name=jac_sub_align
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/jacobian_subspace_alignment_20260909_alex2/logs/%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/jacobian_subspace_alignment_20260909_alex2/logs/%j.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/jacobian_subspace_alignment_20260909
OUT=${BASE}/results/jacobian_subspace_alignment_20260909_alex2
SOURCE=/home/vault/b313dc/b313dc11/PIGNN-Attn-LS/data/pignn_hetero/GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet
DATA=${TMPDIR:?TMPDIR is required}/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
mkdir -p "${OUT}/json" "${OUT}/logs"
test -r "${SOURCE}"; test -r "${OVERLAY}/jacobian_subspace_alignment.py"
cp "${SOURCE}" "${DATA}"
test "$(stat -c %s "${SOURCE}")" = "$(stat -c %s "${DATA}")"
trap 'rm -f "${DATA}"' EXIT
export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
cd "${BASE}"
srun "${PY}" -u "${OVERLAY}/jacobian_subspace_alignment.py" \
 --parquet "${DATA}" --rank 16 --basis-batch 16 \
 --scenarios 128 --directions 32 --seed 20260909 \
 --out "${OUT}/json/jacobian_subspace_alignment_k16.json"
