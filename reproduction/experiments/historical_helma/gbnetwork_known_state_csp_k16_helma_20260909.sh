#!/bin/bash -l
# Re-score only P16/CSP16 variants that preserve AC-PF prescribed voltage
# variables.  Both policies are evaluated from the same per-model forward
# passes: (1) full-state projection followed by setpoint restoration, and
# (2) basis/projection restricted to unknown PF coordinates.
#SBATCH --job-name=gb_known_csp16
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --array=0-11%4
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/gbnetwork_known_state_csp_k16_20260909/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/gbnetwork_known_state_csp_k16_20260909/logs/%A_%a.err

set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/gbnetwork_known_state_csp_k16_20260909
OUT=${BASE}/results/gbnetwork_known_state_csp_k16_20260909
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python

i=${SLURM_ARRAY_TASK_ID:?}
families=(g3 gridsfm graphkit lumina)
seeds=(41 42 43)
family=${families[$((i / 3))]}
seed=${seeds[$((i % 3))]}

case "${family}:${seed}" in
  g3:41) checkpoint=${BASE}/results/g3_ref_s41_ablation_20260829/ckpt/pignn_global_GBnetwork_g3_ref_s41_40_best_model.ckpt ;;
  g3:42) checkpoint=${BASE}/overlays/residual_split_20260829/pignn_global_GBnetwork_g3_s42_40_best_model.ckpt ;;
  g3:43) checkpoint=${BASE}/results/g3_ref_s43_ablation_20260829/ckpt/pignn_global_GBnetwork_g3_ref_s43_40_best_model.ckpt ;;
  gridsfm:41|gridsfm:43) checkpoint=${BASE}/results/model_seed_replicates_20260907/ckpt/gridsfm/gridsfm_GBnetwork_s${seed}_40_best.pt ;;
  gridsfm:42) checkpoint=${BASE}/overlays/residual_split_20260829/pfv2_gridsfm_GBnetwork_b26_best.pt ;;
  graphkit:41|graphkit:43) checkpoint=${BASE}/results/model_seed_replicates_20260907/ckpt/gridfm_graphkit/gridfm_graphkit_GBnetwork_s${seed}_40_best.pt ;;
  graphkit:42) checkpoint=${BASE}/results/ckpt/pfv2_graphkit_D_v2/pfv2_graphkit_GBnetwork_b52_best.pt ;;
  lumina:41|lumina:43) checkpoint=${BASE}/results/model_seed_replicates_20260907/ckpt/lumina/lumina_GBnetwork_s${seed}_40_best.pt ;;
  lumina:42) checkpoint=${BASE}/results/ckpt/pfv2_lumina_D_v2/pfv2_lumina_GBnetwork_b32_best.pt ;;
  *) echo "Unknown task mapping ${family}:${seed}" >&2; exit 2 ;;
esac

mkdir -p "${OUT}/json" "${OUT}/logs"
test -r "${DATA}"; test -r "${checkpoint}"; test -r "${OVERLAY}/output_intervention_metrics.py"
export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
cd "${BASE}"
echo "[task] family=${family} seed=${seed} checkpoint=${checkpoint}"
echo "[protocol] seed-42 split; train-only calibration and bases; k=16; complex128 PB"
echo "[variants] restore_known (full P16 then PV/slack |V| + slack theta setpoints); unknown_only (P16 only on PQ |V| and non-slack theta)"
srun "${PY}" -u "${OVERLAY}/output_intervention_metrics.py" \
  --model "${family}" --checkpoint "${checkpoint}" --parquet "${DATA}" \
  --rank 16 --batch 16 --projection-variants restore_known unknown_only \
  --out "${OUT}/json/${family}_s${seed}_known_state_csp_k16.json"
