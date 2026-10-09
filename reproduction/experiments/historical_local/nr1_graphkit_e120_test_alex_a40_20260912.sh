#!/bin/bash -l
# Frozen-rank test scoring; eta comes only from validation selection.
#SBATCH --job-name=gk120_nr1_test_a40
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr1_graphkit_e120_gbnetwork_a40_20260912/logs/test_%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr1_graphkit_e120_gbnetwork_a40_20260912/logs/test_%A_%a.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC; O=$BASE/overlays/nr1_graphkit_e120_gbnetwork_a40_20260912; R=$BASE/results/nr1_graphkit_e120_gbnetwork_a40_20260912; DATA=/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet; PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
seeds=(41 42 43); stems=(k06_s41_e120 k08_s42_e120 k07_s43_e120); i=${SLURM_ARRAY_TASK_ID:?}; s=${seeds[$i]}; stem=${stems[$i]}; c=$BASE/results/ckpt/gk_e120_20260910/${stem}_best.pt
ETA=$("$PY" -c "import json; print(json.load(open('$R/json/validation_eta_selection.json'))['selected_eta'])")
mkdir -p "$R"/{logs,json,cache}; export PYTHONPATH="$O:$BASE:${PYTHONPATH:-}" SCENARIO_SYNTHESIS_ROOT="$BASE/../ScenarioSynthesis_PPC" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
srun "$PY" -u "$O/nr1_multiprocess_baseline.py" --stage test --model graphkit --checkpoint "$c" --parquet "$DATA" --cache-dir "$R/cache/graphkit_s${s}_test" --out "$R/json/graphkit_s${s}_test.json" --etas "$ETA" --workers 16 --batch 16
