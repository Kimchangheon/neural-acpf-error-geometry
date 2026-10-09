#!/bin/bash -l
# Scores the two new model-seed replicas after the training array succeeds.
# The evaluator always reconstructs the fixed seed-42 split.
#SBATCH --job-name=gb_seedmet
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/model_seed_replicates_20260907/logs/metrics_%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/model_seed_replicates_20260907/logs/metrics_%A_%a.err

set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/model_seed_replicates_20260907
OUT=${BASE}/results/model_seed_replicates_20260907
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
i=${SLURM_ARRAY_TASK_ID:?}
families=(g3 gridsfm graphkit lumina)
seeds=(41 43)
family=${families[$((i / 2))]}; seed=${seeds[$((i % 2))]}
case "${family}:${seed}" in
  g3:41) checkpoint=${BASE}/results/g3_ref_s41_ablation_20260829/ckpt/pignn_global_GBnetwork_g3_ref_s41_40_best_model.ckpt ;;
  g3:43) checkpoint=${BASE}/results/g3_ref_s43_ablation_20260829/ckpt/pignn_global_GBnetwork_g3_ref_s43_40_best_model.ckpt ;;
  gridsfm:*) checkpoint=${OUT}/ckpt/gridsfm/gridsfm_GBnetwork_s${seed}_40_best.pt ;;
  graphkit:*) checkpoint=${OUT}/ckpt/gridfm_graphkit/gridfm_graphkit_GBnetwork_s${seed}_40_best.pt ;;
  lumina:*) checkpoint=${OUT}/ckpt/lumina/lumina_GBnetwork_s${seed}_40_best.pt ;;
esac
mkdir -p "${OUT}/metrics"
test -r "${checkpoint}"; test -r "${DATA}"
export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
cd "${BASE}"
srun "${PY}" -u "${OVERLAY}/output_intervention_metrics.py" \
  --model "${family}" --checkpoint "${checkpoint}" --parquet "${DATA}" --rank 16 --batch 16 \
  --out "${OUT}/metrics/${family}_s${seed}_output_interventions_k16.json"
