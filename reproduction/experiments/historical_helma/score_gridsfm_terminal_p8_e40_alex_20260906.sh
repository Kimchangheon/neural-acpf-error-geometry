#!/bin/bash -l
# Re-score the completed Helma terminal-P8 checkpoint on alex2.  Both hosts
# share the project filesystem; only the available GPU partition differs.
# The four evaluations use the same staged GBnetwork file and diagnostic code.
#SBATCH --job-name=gb40_p8_score
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/gridsfm_terminal_p8_20260905/slurm/%x_%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/gridsfm_terminal_p8_20260905/slurm/%x_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00

set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/gridsfm_terminal_p8_20260905
OUT=${BASE}/results/gridsfm_terminal_p8_20260905/pb_scores
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
REF=${BASE}/results/ckpt/pfv2_gridsfm_D_v2/pfv2_gridsfm_GBnetwork_b26_best.pt
TRT=${BASE}/results/gridsfm_terminal_p8_20260905/ckpt/gbnetwork_gridsfm_terminal_p8_e40_best.pt
mkdir -p "${OUT}"
export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"

score () {
  local tag=$1 ckpt=$2 debias=$3
  local args=(--model gridsfm --checkpoint "${ckpt}" --parquet "${DATA}" --batch 8 --project_k 8 --output "${OUT}/${tag}.json")
  if [[ "${debias}" == yes ]]; then args+=(--debias); fi
  srun "${PY}" -u "${OVERLAY}/diagnose_residual_distributions.py" "${args[@]}"
}

score reference40_p8 "${REF}" no
score reference40_csp8 "${REF}" yes
score terminalp8_e40 "${TRT}" no
score terminalp8_e40_csp8 "${TRT}" yes
