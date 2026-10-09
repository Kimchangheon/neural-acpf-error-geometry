#!/bin/bash -l
#SBATCH --job-name=rescore_mirrorpin_case6470rte
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/rescore_mirrorpin_case6470rte.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/rescore_mirrorpin_case6470rte.err
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
cp "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/pf/case6470rte_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_25551_NR_branchrows_directSI_rg20.parquet" "${TMPDIR}/p.parquet"

srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u rescore_pf.py \
  --PARQUET "${TMPDIR}/p.parquet" \
  --ckpt "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pf_gridfm_h100_20260812_120439/pf_mirrorpin_case6470rte_b18_best.pt" \
  --grid case6470rte --impl mirrorpin --BATCH 32 \
  --hidden_size 48 --num_layers 12 --n_heads 8 \
  --json_out "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/rescore/pf_gridfm_h100_20260812_120439/case6470rte.json"
