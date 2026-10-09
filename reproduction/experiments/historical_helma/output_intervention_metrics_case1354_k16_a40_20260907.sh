#!/bin/bash -l
# case1354pegase 40-epoch baseline output-space table, fixed P16 transfer.
#SBATCH --job-name=c1354_out16
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/case1354_output_interventions_20260907/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/case1354_output_interventions_20260907/logs/%A_%a.err

set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/controlled_geometry_20260905
DATA=/home/vault/b313dc/b313dc11/PIGNN-Attn-LS/data/pf/case1354pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37038_NR_branchrows_directSI.parquet
OUT=${BASE}/results/case1354_output_interventions_20260907
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
models=(pignn_base gridsfm graphkit lumina)
checkpoints=(
 "${BASE}/results/ckpt/pfv2_pignn_D_v2fix/pfv2_pignn_case1354pegase_b32_pngraph_mw5_cos_lr1e-5_40_best_model.ckpt"
 "${BASE}/results/ckpt/pfv2_gridsfm_D_v2c/pfv2_gridsfm_case1354pegase_b32_best.pt"
 "${BASE}/results/ckpt/pfv2_graphkit_D_v2c/pfv2_graphkit_case1354pegase_b64_best.pt"
 "${BASE}/results/ckpt/pfv2_lumina_D_v2c/pfv2_lumina_case1354pegase_b32_best.pt"
)
i=${SLURM_ARRAY_TASK_ID:?}; model=${models[$i]}; checkpoint=${checkpoints[$i]}
mkdir -p "${OUT}/json" "${OUT}/logs"
test -r "${DATA}"; test -r "${checkpoint}"
export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
cd "${BASE}"
srun "${PY}" -u "${OVERLAY}/output_intervention_metrics.py" \
 --model "${model}" --checkpoint "${checkpoint}" --parquet "${DATA}" \
 --rank 16 --batch 16 --out "${OUT}/json/${model}_output_interventions_k16.json"
