#!/bin/bash -l
# Rescore two completed GBnetwork LUMINA checkpoints.  No training.
# Both rows use the manuscript's full-state P16/CSP16 convention.
#SBATCH --job-name=lumina_2ckpt_full16
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/lumina_two_ckpt_fullstate_csp_k16_20260912/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/lumina_two_ckpt_fullstate_csp_k16_20260912/logs/%A_%a.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/lumina_two_ckpt_fullstate_csp_k16_20260912
OUT=${BASE}/results/lumina_two_ckpt_fullstate_csp_k16_20260912
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
mkdir -p "${OUT}/json" "${OUT}/logs"

case ${SLURM_ARRAY_TASK_ID:?} in
  0)
    LABEL=gbv05_anchor_prior_z_ft
    CKPT=${BASE}/results/ckpt/gb_vsweep_20260911/gbv05_anchor_prior_z_ft_best.pt
    # Exact non-default output/feature parametrisation from its sbatch source.
    LARGS='--model_config /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/lumina_ckpt/lumina_config_do0.json --mask_known_v --bus_physics_features --theta_anchor start --v_anchor prior --bus_vmag_prior chord --vmag_prior_angle start --init_recipe sd002_zerohead'
    ;;
  1)
    LABEL=gbv06z_control_zerohead
    CKPT=${BASE}/results/ckpt/gb_vsweep2_20260911/gbv06z_control_zerohead_best.pt
    LARGS='--model_config /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/lumina_ckpt/lumina_config_do0.json --mask_known_v --bus_physics_features --theta_anchor start --init_recipe sd002_zerohead'
    ;;
  *) exit 2 ;;
esac
test -r "${DATA}"; test -r "${CKPT}"; test -r "${OVERLAY}/output_intervention_metrics.py"
export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${BASE}"
echo "[checkpoint] ${LABEL}: ${CKPT}"
echo "[lumina flags] ${LARGS}"
echo "[protocol] fixed seed-42 split; train-only calibration/bases; k=16; full-state projection"
srun "${PY}" -u "${OVERLAY}/output_intervention_metrics.py" \
  --model lumina --checkpoint "${CKPT}" --parquet "${DATA}" \
  --rank 16 --batch 16 --projection-variants full --lumina-args "${LARGS}" \
  --out "${OUT}/json/${LABEL}_fullstate_csp_k16.json"
