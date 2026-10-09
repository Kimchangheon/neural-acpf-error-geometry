#!/bin/bash
set -euo pipefail

CAMPAIGN=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/lvn_final_plan_20260821
RESULT_ROOT=/home/vault/b313dc/b313dc11/results/lvn_final_plan_20260821
mkdir -p "${RESULT_ROOT}/slurm" "${RESULT_ROOT}/logs" "${RESULT_ROOT}/ckpt"

regression_job=$(sbatch --parsable "${CAMPAIGN}/newton_floor_helma.sh")
curriculum_job=$(sbatch --parsable --dependency="afterok:${regression_job}" "${CAMPAIGN}/curriculum_3seed_helma.sh")

printf 'regression_job=%s\n' "${regression_job}"
printf 'curriculum_job=%s dependency=afterok:%s\n' "${curriculum_job}" "${regression_job}"
