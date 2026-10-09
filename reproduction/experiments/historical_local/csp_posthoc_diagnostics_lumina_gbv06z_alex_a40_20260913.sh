#!/bin/bash -l
# CSP post-hoc retained-variance/off-error/identity diagnostic for LUMINA.
#SBATCH --job-name=lumina_csp_posthoc
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/csp_posthoc_lumina_gbv06z_20260913/logs/%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/csp_posthoc_lumina_gbv06z_20260913/logs/%j.err
set -euo pipefail
BASE=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC; O=$BASE/overlays/csp_posthoc_lumina_gbv06z_20260913; R=$BASE/results/csp_posthoc_lumina_gbv06z_20260913; PY=/home/hpc/iwi5/iwi5295h/conda-envs/gridfm-py312/bin/python; DATA=/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet; C=$BASE/results/ckpt/gb_vsweep2_20260911/gbv06z_control_zerohead_best.pt
LARGS='--model_config /home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/lumina_ckpt/lumina_config_do0.json --mask_known_v --bus_physics_features --theta_anchor start --init_recipe sd002_zerohead'; mkdir -p "$R"/{logs,json}; export PYTHONPATH="$O:$BASE" OMP_NUM_THREADS=16; srun "$PY" -u "$O/csp_posthoc_diagnostics.py" --parquet "$DATA" --lumina "$C" --lumina-args "$LARGS" --rank 16 --batch 16 --out-json "$R/json/lumina_gbv06z_csp_posthoc_k16.json" --out-csv "$R/json/lumina_gbv06z_csp_posthoc_k16.csv"
