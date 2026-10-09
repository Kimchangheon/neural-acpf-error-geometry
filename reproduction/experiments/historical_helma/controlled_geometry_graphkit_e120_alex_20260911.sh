#!/bin/bash -l
# Frozen-rank (k=16) fixed-norm directional diagnostic for the three
# independently trained GraphKit 120-epoch GBnetwork checkpoints.  This is a
# test-only reproduction of the recorded controlled-geometry Stage-2 protocol;
# it deliberately does not perform rank selection.
#SBATCH --job-name=gk120_cgeom
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/controlled_geometry_graphkit_e120_alex_20260911/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/controlled_geometry_graphkit_e120_alex_20260911/logs/%A_%a.err

set -euo pipefail

BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/graphkit_e120_fullstate_csp_k16_alex_20260911
OUT=${BASE}/results/controlled_geometry_graphkit_e120_alex_20260911
SOURCE=/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet
DATA=${TMPDIR:?TMPDIR is required}/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python

seeds=(41 42 43)
stems=(k06_s41_e120 k08_s42_e120 k07_s43_e120)
i=${SLURM_ARRAY_TASK_ID:?}
seed=${seeds[$i]}
stem=${stems[$i]}
CKPT=${BASE}/results/graphkit_e120_fullstate_csp_k16_alex_20260911/ckpt/${stem}_best.pt

mkdir -p "${OUT}/json" "${OUT}/logs"
test -r "${SOURCE}"
test -r "${CKPT}"
test -r "${OVERLAY}/controlled_error_geometry.py"
cp "${SOURCE}" "${DATA}"
test "$(stat -c %s "${SOURCE}")" = "$(stat -c %s "${DATA}")"

export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
cd "${BASE}"
echo "[checkpoint] ${CKPT}"
echo "[protocol] frozen Stage-2: seed-42 split; train-only calibration/basis; k=16; complex128 GENCO PB"
echo "[diagnostic] fixed-norm off-/in-subspace directional counterfactuals; calibrated angles fixed; no clipping"
srun "${PY}" -u "${OVERLAY}/controlled_error_geometry.py" \
  --phase test --ranks 16 --batch 16 --models graphkit \
  --parquet "${DATA}" --out "${OUT}/json/graphkit_s${seed}_e120_test_k16.json" \
  --pignn /dev/null --gridsfm /dev/null --graphkit "${CKPT}" --lumina /dev/null
