#!/bin/bash -l
#SBATCH --job-name=probe_alpha
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/probe_alpha.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/probe_alpha.err
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --time=00:25:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
cp "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/pf/case118_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet" "${TMPDIR}/p.parquet"
srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u probe_alpha.py --PARQUET "${TMPDIR}/p.parquet" \
  --ckpt "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/gridfm/smoke_graphkit_ls_best.pt" --BATCH 8 --batches 4
