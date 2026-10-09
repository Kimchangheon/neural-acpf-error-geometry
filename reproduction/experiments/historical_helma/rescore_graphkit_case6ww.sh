#!/bin/bash -l
#SBATCH --job-name=rescore_graphkit_case6ww
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/rescore_graphkit_case6ww.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/rescore_graphkit_case6ww.err
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
# Deliberately does NOT delete the /Users/changhunkim copy: a training job for the same grid
# may still be PENDING and needs that exact file, and removing it would fail
# that job at startup. The training jobs clean up after themselves.
cp "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/pf/case6ww_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet" "${TMPDIR}/p.parquet"

srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u rescore_pf.py \
  --PARQUET "${TMPDIR}/p.parquet" \
  --ckpt "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pf_gridfm_h100_20260811_135024/pf_graphkit_case6ww_b64_best.pt" \
  --grid case6ww --impl graphkit --BATCH 32 \
  --hidden_size 48 --num_layers 12 --n_heads 8 \
  --json_out "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/rescore/pf_gridfm_h100_20260811_135024/case6ww.json"
