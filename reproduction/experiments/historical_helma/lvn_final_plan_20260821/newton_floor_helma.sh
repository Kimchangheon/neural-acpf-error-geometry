#!/bin/bash -l
#SBATCH --job-name=lvn_newton_floor
#SBATCH --output=/home/vault/b313dc/b313dc11/results/lvn_final_plan_20260821/slurm/newton_floor_%j.out
#SBATCH --error=/home/vault/b313dc/b313dc11/results/lvn_final_plan_20260821/slurm/newton_floor_%j.err
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48

set -euo pipefail

BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
CAMPAIGN=${BASE}/sbatch/lvn_final_plan_20260821
OVERLAY=${CAMPAIGN}/overlay
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
SOURCE_PARQUET=/home/vault/b313dc/b313dc11/data/pf_ppnr_v2/LVN_heo1_ppNR_37000.parquet
LOCAL_PARQUET=${TMPDIR}/LVN_heo1_ppNR_37000.parquet

export PYTHONPATH=${OVERLAY}:${BASE}:${PYTHONPATH:-}
cp "${SOURCE_PARQUET}" "${LOCAL_PARQUET}"

"${PY}" -u "${OVERLAY}/validate_newton_physics_floor.py" \
  --parquet "${LOCAL_PARQUET}" \
  --samples 8 \
  --target-s-base 1e8 \
  --max-raw-residual 1e-5 \
  --max-local-grad-norm 1e-5
