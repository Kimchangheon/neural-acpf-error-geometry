#!/bin/bash -l
# One-seed LUMINA NR1 pipeline: validation-only damping selection.
#SBATCH --job-name=lumina_nr1_val
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr1_lumina_gbv06z_20260913/logs/val_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr1_lumina_gbv06z_20260913/logs/val_%j.err
set -euo pipefail
BASE=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC; O=$BASE/overlays/nr1_lumina_gbv06z_20260913; R=$BASE/results/nr1_lumina_gbv06z_20260913
DATA=/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet
PY=/home/hpc/iwi5/iwi5295h/conda-envs/gridfm-py312/bin/python; C=$BASE/results/ckpt/gb_vsweep2_20260911/gbv06z_control_zerohead_best.pt
LARGS='--model_config /home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/lumina_ckpt/lumina_config_do0.json --mask_known_v --bus_physics_features --theta_anchor start --init_recipe sd002_zerohead'
mkdir -p "$R"/{logs,json,cache}; export PYTHONPATH="$O:$BASE" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
srun "$PY" -u "$O/nr1_multiprocess_baseline.py" --stage validation --model lumina --checkpoint "$C" --parquet "$DATA" --cache-dir "$R/cache/lumina_s42_val" --out "$R/json/lumina_s42_validation.json" --workers 16 --batch 16 --lumina-args "$LARGS"
