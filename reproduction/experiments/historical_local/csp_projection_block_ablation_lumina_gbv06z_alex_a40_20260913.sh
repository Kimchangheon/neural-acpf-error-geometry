#!/bin/bash -l
# One-seed LUMINA magnitude/angle CSP block ablation (full-state, no restore).
#SBATCH --job-name=lumina_csp_blocks
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/csp_blocks_lumina_gbv06z_20260913/logs/%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/csp_blocks_lumina_gbv06z_20260913/logs/%j.err
set -euo pipefail
BASE=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC; O=$BASE/overlays/csp_blocks_lumina_gbv06z_20260913; R=$BASE/results/csp_blocks_lumina_gbv06z_20260913; DATA=/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet; PY=/home/hpc/iwi5/iwi5295h/conda-envs/gridfm-py312/bin/python; C=$BASE/results/ckpt/gb_vsweep2_20260911/gbv06z_control_zerohead_best.pt
LARGS='--model_config /home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/lumina_ckpt/lumina_config_do0.json --mask_known_v --bus_physics_features --theta_anchor start --init_recipe sd002_zerohead'; mkdir -p "$R"/{json,logs}; export PYTHONPATH="$O:$BASE" OMP_NUM_THREADS=16
srun "$PY" -u "$O/csp_projection_block_ablation.py" --model lumina --checkpoint "$C" --parquet "$DATA" --rank 16 --batch 16 --lumina-args "$LARGS" --out "$R/json/lumina_s42_blocks_k16_norestore.json"
