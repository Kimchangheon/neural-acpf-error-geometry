#!/bin/bash -l
# Independent Alex2 A100 reproduction of Helma job 836267.  It evaluates the
# same three independently trained 120-epoch GraphKit checkpoints after their
# byte-preserving transfer from Helma.  The Alex parquet differs in on-disk
# size but has been row-group fingerprinted against the Helma reference.
#SBATCH --job-name=gk_e120_a100
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/graphkit_e120_fullstate_csp_k16_alex_20260911/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/graphkit_e120_fullstate_csp_k16_alex_20260911/logs/%A_%a.err

set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/graphkit_e120_fullstate_csp_k16_alex_20260911
OUT=${BASE}/results/graphkit_e120_fullstate_csp_k16_alex_20260911
SOURCE=/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet
DATA=${TMPDIR:?TMPDIR is required}/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python

seeds=(41 42 43)
stems=(k06_s41_e120 k08_s42_e120 k07_s43_e120)
i=${SLURM_ARRAY_TASK_ID:?}
seed=${seeds[$i]}
stem=${stems[$i]}
CKPT=${OUT}/ckpt/${stem}_best.pt

mkdir -p "${OUT}/json" "${OUT}/logs"
test -r "${SOURCE}"; test -r "${CKPT}"; test -r "${OVERLAY}/output_intervention_metrics.py"
cp "${SOURCE}" "${DATA}"
test "$(stat -c %s "${SOURCE}")" = "$(stat -c %s "${DATA}")"
export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
cd "${BASE}"
echo "[checkpoint] ${CKPT}"
echo "[protocol] Helma-836267 reproduction; seed-42 split; train-only calibration/bases; k=16; complex128 PB"
echo "[projection] full state only; no restore_known; no unknown-only restriction"
srun "${PY}" -u "${OVERLAY}/output_intervention_metrics.py" \
  --model graphkit --checkpoint "${CKPT}" --parquet "${DATA}" \
  --rank 16 --batch 16 --projection-variants full \
  --out "${OUT}/json/graphkit_s${seed}_e120_fullstate_csp_k16.json"
