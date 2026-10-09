#!/bin/bash -l
# Re-score the existing case1354 ppNR-v2 PIGNN baseline after reconstructing
# its original 8-channel (residual_feature_norm=none) input contract.
#SBATCH --job-name=c1354_pignn_rescore
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/case1354_output_interventions_20260907/logs/%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/case1354_output_interventions_20260907/logs/%j.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00

set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/case1354_pignn_rescore_20260908
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
# Helma compute nodes do not mount the Alex vault path.  This immutable copy
# is staged and byte-verified under the Helma home filesystem before submit.
DATA=/home/hpc/b313dc/b313dc11/data_staging/case1354pegase_ppnr_v2_37038.parquet
CKPT=${BASE}/results/ckpt/pfv2_pignn_D_v2fix/pfv2_pignn_case1354pegase_b32_pngraph_mw5_cos_lr1e-5_40_best_model.ckpt
OUT=${BASE}/results/case1354_output_interventions_20260907/json/pignn_base_output_interventions_k16.json

test -r "${DATA}"; test -r "${CKPT}"; test -x "${PY}"
test -r "${OVERLAY}/output_intervention_metrics.py"
mkdir -p "$(dirname "${OUT}")"
export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
cd "${BASE}"
srun "${PY}" -u "${OVERLAY}/output_intervention_metrics.py" \
  --model pignn_base --checkpoint "${CKPT}" --parquet "${DATA}" \
  --rank 16 --batch 16 --out "${OUT}"
