#!/bin/bash -l
# Alex2/A40 validation-only scalar shrinkage selection; no test scoring.
#SBATCH --job-name=g3_shrink_val
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/pignn_g3_shrinkage_20260909_alex2/logs/%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/pignn_g3_shrinkage_20260909_alex2/logs/%j.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/pignn_g3_shrinkage_20260909
OUT=${BASE}/results/pignn_g3_shrinkage_20260909_alex2
SOURCE=/home/vault/b313dc/b313dc11/PIGNN-Attn-LS/data/pignn_hetero/GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet
DATA=${TMPDIR:?TMPDIR is required}/GBnetwork.parquet
CKPT=${BASE}/overlays/residual_split_20260829/pignn_global_GBnetwork_g3_s42_40_best_model.ckpt
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
mkdir -p "${OUT}/json" "${OUT}/logs"
test -r "${SOURCE}"; test -r "${CKPT}"; test -r "${OVERLAY}/evaluate_scalar_shrinkage.py"
cp "${SOURCE}" "${DATA}"
test "$(stat -c %s "${SOURCE}")" = "$(stat -c %s "${DATA}")"
trap 'rm -f "${DATA}"' EXIT
export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
cd "${BASE}"
srun "${PY}" -u "${OVERLAY}/evaluate_scalar_shrinkage.py" --stage validation \
 --checkpoint "${CKPT}" --parquet "${DATA}" --batch 16 \
 --out "${OUT}/json/validation.json"
