#!/bin/bash -l
#SBATCH --job-name=offgeom_s42
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/off_subspace_conditional_association_20260914/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/off_subspace_conditional_association_20260914/logs/%A_%a.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=${BASE}/overlays/off_subspace_conditional_association_20260914
OUT=${BASE}/results/off_subspace_conditional_association_20260914
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python

case "${SLURM_ARRAY_TASK_ID:?}" in
  0)
    kind=g3; name=PIGNN-GC
    checkpoint=${BASE}/overlays/residual_split_20260829/pignn_global_GBnetwork_g3_s42_40_best_model.ckpt
    lumina_args=""
    ;;
  1)
    kind=gridsfm; name=GridSFM
    checkpoint=${BASE}/overlays/residual_split_20260829/pfv2_gridsfm_GBnetwork_b26_best.pt
    lumina_args=""
    ;;
  2)
    kind=graphkit; name=GridFM-GraphKit
    checkpoint=${BASE}/results/ckpt/gk_e120_20260910/k08_s42_e120_best.pt
    lumina_args=""
    ;;
  3)
    kind=lumina; name=LUMINA
    checkpoint=${BASE}/results/ckpt/gb_vsweep2_20260911/gbv06z_control_zerohead_best.pt
    lumina_args="--model_config ${BASE}/lumina_ckpt/lumina_config_do0.json --mask_known_v --bus_physics_features --theta_anchor start --init_recipe sd002_zerohead"
    ;;
esac

mkdir -p "$OUT/json" "$OUT/raw" "$OUT/logs"
test -r "$DATA"; test -r "$checkpoint"; test -r "$OVERLAY/off_subspace_conditional_association.py"
export PYTHONPATH="$OVERLAY:$BASE:${PYTHONPATH:-}"
cd "$BASE"
srun "$PY" -u "$OVERLAY/off_subspace_conditional_association.py" \
  --parquet "$DATA" --kind "$kind" --name "$name" --checkpoint "$checkpoint" \
  --lumina-args "$lumina_args" --rank 16 --batch 16 --bootstrap 2000 \
  --out-json "$OUT/json/${kind}_s42.json" --out-npz "$OUT/raw/${kind}_s42.npz"
