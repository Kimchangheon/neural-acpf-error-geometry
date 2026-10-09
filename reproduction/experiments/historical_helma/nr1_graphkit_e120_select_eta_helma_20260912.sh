#!/bin/bash -l
# Freeze the damping parameter using only the preceding validation array.
#SBATCH --job-name=gk120_nr1_eta
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr1_graphkit_e120_gbnetwork_20260912/logs/select_%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr1_graphkit_e120_gbnetwork_20260912/logs/select_%j.err
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
O=$BASE/overlays/nr1_graphkit_e120_gbnetwork_20260912
R=$BASE/results/nr1_graphkit_e120_gbnetwork_20260912
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
"$PY" "$O/select_nr1_eta_graphkit.py" --glob "$R/json/*_validation.json" --out "$R/json/validation_eta_selection.json"
cat "$R/json/validation_eta_selection.json"
