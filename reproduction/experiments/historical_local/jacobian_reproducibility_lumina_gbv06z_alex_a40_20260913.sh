#!/bin/bash -l
# Frozen Jacobian-gain audit for one LUMINA checkpoint.
#SBATCH --job-name=lumina_jac_audit
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/jacobian_lumina_gbv06z_20260913/logs/%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/jacobian_lumina_gbv06z_20260913/logs/%j.err
set -euo pipefail
BASE=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC; O=$BASE/overlays/jacobian_lumina_gbv06z_20260913; R=$BASE/results/jacobian_lumina_gbv06z_20260913; PY=/home/hpc/iwi5/iwi5295h/conda-envs/gridfm-py312/bin/python; DATA=/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet; C=$BASE/results/ckpt/gb_vsweep2_20260911/gbv06z_control_zerohead_best.pt
LARGS='--model_config /home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/lumina_ckpt/lumina_config_do0.json --mask_known_v --bus_physics_features --theta_anchor start --init_recipe sd002_zerohead'; mkdir -p "$R"/{logs,json}; LOCAL_DATA=${TMPDIR:?}/GBnetwork.parquet; cp "$DATA" "$LOCAL_DATA"; export PYTHONPATH="$O:$BASE" OMP_NUM_THREADS=16
srun "$PY" -u "$O/jacobian_subspace_reproducibility.py" --parquet "$LOCAL_DATA" --lumina "$C" --lumina-args "$LARGS" --rank 16 --scenarios 128 --directions 32 --seed 20260909 --basis-batch 16 --out-json "$R/json/jacobian_lumina_k16.json" --out-csv "$R/json/jacobian_lumina_k16.csv" --out-npz "$R/json/jacobian_lumina_gains_k16.npz"
