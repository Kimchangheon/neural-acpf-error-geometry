#!/bin/bash -l
# One-seed LUMINA metric-complete frozen NR1 rescore.
#SBATCH --job-name=lumina_nr1_complete
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr1_lumina_gbv06z_20260913/logs/complete_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr1_lumina_gbv06z_20260913/logs/complete_%j.err
set -euo pipefail
BASE=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC; O=$BASE/overlays/nr1_lumina_gbv06z_20260913; R=$BASE/results/nr1_lumina_gbv06z_20260913; DATA=/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet; PY=/home/hpc/iwi5/iwi5295h/conda-envs/gridfm-py312/bin/python
E=$("$PY" -c "import json;print(json.load(open('$R/json/validation_eta_selection.json'))['selected_eta'])"); export PYTHONPATH="$O:$BASE" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
srun "$PY" -u "$O/nr1_metric_complete_rescore.py" --model lumina --seed 42 --parquet "$DATA" --eta "$E" --workers 16 --batch 16 --chunk 16 --cache-dir "$R/cache/lumina_s42_test" --out-dir "$R/complete/lumina_s42"
