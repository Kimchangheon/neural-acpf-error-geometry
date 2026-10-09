#!/bin/bash -l
# Independent (non-array) replacement for the throttled timing array.
# Submit with: sbatch --export=FAMILY=...,SEED=... this_script
#SBATCH --job-name=gb_inf_walltime
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/gbnetwork_inference_walltime_h100_20260912/logs/time_ind_%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/gbnetwork_inference_walltime_h100_20260912/logs/time_ind_%j.err
set -euo pipefail
: "${FAMILY:?set FAMILY=pignn_gc|gridsfm|graphkit|lumina}"
: "${SEED:?set SEED=41|42|43 (lumina uses 42)}"
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
OVERLAY=$BASE/overlays/gbnetwork_inference_walltime_h100_20260912
OUT=$BASE/results/gbnetwork_inference_walltime_h100_20260912
DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
case "$FAMILY:$SEED" in
 pignn_gc:41) MODEL=g3; CKPT=$BASE/results/g3_ref_s41_ablation_20260829/ckpt/pignn_global_GBnetwork_g3_ref_s41_40_best_model.ckpt;;
 pignn_gc:42) MODEL=g3; CKPT=$BASE/overlays/residual_split_20260829/pignn_global_GBnetwork_g3_s42_40_best_model.ckpt;;
 pignn_gc:43) MODEL=g3; CKPT=$BASE/results/g3_ref_s43_ablation_20260829/ckpt/pignn_global_GBnetwork_g3_ref_s43_40_best_model.ckpt;;
 gridsfm:41) MODEL=gridsfm; CKPT=$BASE/results/model_seed_replicates_20260907/ckpt/gridsfm/gridsfm_GBnetwork_s41_40_best.pt;;
 gridsfm:42) MODEL=gridsfm; CKPT=$BASE/overlays/residual_split_20260829/pfv2_gridsfm_GBnetwork_b26_best.pt;;
 gridsfm:43) MODEL=gridsfm; CKPT=$BASE/results/model_seed_replicates_20260907/ckpt/gridsfm/gridsfm_GBnetwork_s43_40_best.pt;;
 graphkit:41) MODEL=graphkit; CKPT=$BASE/results/ckpt/gk_e120_20260910/k06_s41_e120_best.pt;;
 graphkit:42) MODEL=graphkit; CKPT=$BASE/results/ckpt/gk_e120_20260910/k08_s42_e120_best.pt;;
 graphkit:43) MODEL=graphkit; CKPT=$BASE/results/ckpt/gk_e120_20260910/k07_s43_e120_best.pt;;
 lumina:42) MODEL=lumina; CKPT=$BASE/results/ckpt/gb_vsweep2_20260911/gbv06z_control_zerohead_best.pt; LARGS="--model_config $BASE/lumina_ckpt/lumina_config_do0.json --mask_known_v --bus_physics_features --theta_anchor start --init_recipe sd002_zerohead";;
 *) echo "Unsupported FAMILY/SEED: $FAMILY/$SEED" >&2; exit 2;;
esac
LARGS=${LARGS:-}
mkdir -p "$OUT/timing" "$OUT/logs"
test -r "$DATA"; test -r "$CKPT"; test -r "$OUT/probe/$FAMILY.json"
export PYTHONPATH="$OVERLAY:$BASE:${PYTHONPATH:-}" OMP_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE"
srun "$PY" -u "$OVERLAY/benchmark_inference_walltime.py" --stage benchmark --model "$MODEL" --checkpoint "$CKPT" --parquet "$DATA" --lumina-args "$LARGS" --batch-json "$OUT/probe/$FAMILY.json" --fit-batch 16 --repeats 3 --out "$OUT/timing/${FAMILY}_s${SEED}_ind_${SLURM_JOB_ID}.json"
