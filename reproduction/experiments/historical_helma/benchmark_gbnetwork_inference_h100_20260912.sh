#!/bin/bash -l
# Three checkpoints per family; Raw/C/P16/CSP16 timing only.
#SBATCH --job-name=gb_inf_walltime
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
# Three seeds for PIGNN-GC/GridSFM/GraphKit, plus the single available
# chord-free 120-epoch LUMINA control checkpoint.
#SBATCH --array=0-9%4
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/gbnetwork_inference_walltime_h100_20260912/logs/time_%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/gbnetwork_inference_walltime_h100_20260912/logs/time_%A_%a.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/gbnetwork_inference_walltime_h100_20260912
OUT=${BASE}/results/gbnetwork_inference_walltime_h100_20260912
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
i=${SLURM_ARRAY_TASK_ID:?}; family_index=$((i / 3)); seed_index=$((i % 3))
seeds=(41 42 43); seed=${seeds[$seed_index]}
LARGS=''
case ${family_index} in
  0) FAMILY=pignn_gc; MODEL=g3
     case ${seed} in
       41) CKPT=${BASE}/results/g3_ref_s41_ablation_20260829/ckpt/pignn_global_GBnetwork_g3_ref_s41_40_best_model.ckpt ;;
       42) CKPT=${BASE}/overlays/residual_split_20260829/pignn_global_GBnetwork_g3_s42_40_best_model.ckpt ;;
       43) CKPT=${BASE}/results/g3_ref_s43_ablation_20260829/ckpt/pignn_global_GBnetwork_g3_ref_s43_40_best_model.ckpt ;;
     esac ;;
  1) FAMILY=gridsfm; MODEL=gridsfm
     case ${seed} in
       41) CKPT=${BASE}/results/model_seed_replicates_20260907/ckpt/gridsfm/gridsfm_GBnetwork_s41_40_best.pt ;;
       42) CKPT=${BASE}/overlays/residual_split_20260829/pfv2_gridsfm_GBnetwork_b26_best.pt ;;
       43) CKPT=${BASE}/results/model_seed_replicates_20260907/ckpt/gridsfm/gridsfm_GBnetwork_s43_40_best.pt ;;
     esac ;;
  2) FAMILY=graphkit; MODEL=graphkit
     case ${seed} in
       41) CKPT=${BASE}/results/ckpt/gk_e120_20260910/k06_s41_e120_best.pt ;;
       42) CKPT=${BASE}/results/ckpt/gk_e120_20260910/k08_s42_e120_best.pt ;;
       43) CKPT=${BASE}/results/ckpt/gk_e120_20260910/k07_s43_e120_best.pt ;;
     esac ;;
  3) FAMILY=lumina; MODEL=lumina; seed=42
     LARGS="--model_config ${BASE}/lumina_ckpt/lumina_config_do0.json --mask_known_v --bus_physics_features --theta_anchor start --init_recipe sd002_zerohead"
     CKPT=${BASE}/results/ckpt/gb_vsweep2_20260911/gbv06z_control_zerohead_best.pt ;;
  *) exit 2 ;;
esac
mkdir -p "${OUT}/timing" "${OUT}/logs"
test -r "${DATA}"; test -r "${CKPT}"; test -r "${OUT}/probe/${FAMILY}.json"
export PYTHONPATH="${OVERLAY}:${BASE}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${BASE}"
echo "[benchmark] family=${FAMILY} seed=${seed} checkpoint=${CKPT}"
srun "${PY}" -u "${OVERLAY}/benchmark_inference_walltime.py" \
  --stage benchmark --model "${MODEL}" --checkpoint "${CKPT}" --parquet "${DATA}" \
  --lumina-args "${LARGS}" --batch-json "${OUT}/probe/${FAMILY}.json" \
  --fit-batch 16 --repeats 3 --out "${OUT}/timing/${FAMILY}_s${seed}.json"
