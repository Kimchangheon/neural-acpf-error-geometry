#!/bin/bash -l
# Runs only after the six validation scorers have produced their JSON files.
# It freezes the global damping factor before any test-set cache is made.
#SBATCH --job-name=nr1_select_eta
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr1_multiproc_gbnetwork_20260910/logs/select_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr1_multiproc_gbnetwork_20260910/logs/select_%j.err
set -euo pipefail
BASE=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
O=$BASE/overlays/nr1_multiproc_gbnetwork_20260910
R=$BASE/results/nr1_multiproc_gbnetwork_20260910
PY=/home/hpc/iwi5/iwi5295h/conda-envs/gridfm-py312/bin/python
"$PY" "$O/select_nr1_eta.py" --glob "$R/json/*_validation.json" --out "$R/json/validation_eta_selection.json"
cat "$R/json/validation_eta_selection.json"
