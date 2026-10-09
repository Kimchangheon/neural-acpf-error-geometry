#!/bin/bash -l
# Full-state P16/CSP16 scoring for the two independently trained 120-epoch
# GridSFM families: standard output and training/inference --mask_known_v.
# Full-state projection is deliberate, matching the existing manuscript table:
# no restore_known and no unknown-only coordinate restriction.
#SBATCH --job-name=gsfm_e120_full16
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --array=0-5%4
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/gridsfm_e120_fullstate_csp_k16_20260912/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/gridsfm_e120_fullstate_csp_k16_20260912/logs/%A_%a.err

set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/gridsfm_e120_fullstate_csp_k16_20260912
OUT=${BASE}/results/gridsfm_e120_fullstate_csp_k16_20260912
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python

i=${SLURM_ARRAY_TASK_ID:?}
families=(mask mask mask nomask nomask nomask)
seeds=(41 42 43 41 42 43)
family=${families[$i]}; seed=${seeds[$i]}
case "${family}" in
  mask)
    model=gridsfm_mask
    ckpt=${BASE}/results/ckpt/gsfm_mask_e120_20260910/gsfm_gb_mask_s${seed}_e120_best.pt ;;
  nomask)
    model=gridsfm
    ckpt=${BASE}/results/ckpt/gsfm_nomask_e120_20260910/gsfm_gb_nomask_s${seed}_e120_best.pt ;;
  *) echo "Invalid family ${family}" >&2; exit 2 ;;
esac

mkdir -p "${OUT}/json" "${OUT}/logs"
test -r "${DATA}"; test -r "${ckpt}"; test -r "${OVERLAY}/output_intervention_metrics.py"
export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
cd "${BASE}"
echo "[checkpoint] family=${family} seed=${seed} model=${model} path=${ckpt}"
echo "[protocol] seed-42 split; train-only calibration/bases; k=16; complex128 PB"
echo "[projection] full state only; no restore_known; no unknown-only restriction"
echo "[mask] ${family}; mask model applies training-time --mask_known_v at inference"
srun "${PY}" -u "${OVERLAY}/output_intervention_metrics.py" \
  --model "${model}" --checkpoint "${ckpt}" --parquet "${DATA}" \
  --rank 16 --batch 16 --projection-variants full \
  --out "${OUT}/json/gridsfm_${family}_s${seed}_e120_fullstate_csp_k16.json"
