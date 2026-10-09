#!/bin/bash -l
# Validation-only damping selection for the cached custom-NR1 baseline.
#SBATCH --job-name=nr1_val
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr1_multiproc_gbnetwork_20260910/logs/val_%A_%a.out
#SBATCH --error=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr1_multiproc_gbnetwork_20260910/logs/val_%A_%a.err
set -euo pipefail
BASE=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC; O=$BASE/overlays/nr1_multiproc_gbnetwork_20260910; R=$BASE/results/nr1_multiproc_gbnetwork_20260910; DATA=/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet; PY=/home/hpc/iwi5/iwi5295h/conda-envs/gridfm-py312/bin/python
families=(g3 g3 g3 gridsfm gridsfm gridsfm); seeds=(41 42 43 41 42 43); i=${SLURM_ARRAY_TASK_ID}; m=${families[$i]}; s=${seeds[$i]}
case "$m:$s" in g3:41|g3:43) c=$R/checkpoints/pignn_global_GBnetwork_g3_ref_s${s}_40_best_model.ckpt;; g3:42) c=$BASE/overlays/residual_diagnostics_20260828/pignn_global_GBnetwork_g3_s42_40_best_model.ckpt;; gridsfm:42) c=$BASE/overlays/residual_diagnostics_20260828/pfv2_gridsfm_GBnetwork_b26_best.pt;; gridsfm:*) c=$R/checkpoints/gridsfm_GBnetwork_s${s}_40_best.pt;; esac
mkdir -p $R/{logs,json,cache}; export PYTHONPATH=$O:$BASE; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
srun $PY -u $O/nr1_multiprocess_baseline.py --stage validation --model $m --checkpoint $c --parquet $DATA --cache-dir $R/cache/${m}_s${s}_val --out $R/json/${m}_s${s}_validation.json --workers 16 --batch 16
