#!/bin/bash -l
#SBATCH --job-name=resid_type
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/resid_type.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/resid_type.err
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
cp "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/pf/LVN_heo1_ppcY_backbone_dc_compile_cNR_ls0.60-1.40_u0clean_siNR_36000_NR_branchrows_directSI.parquet" "${TMPDIR}/p.parquet"
/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u resid_by_type.py "${TMPDIR}/p.parquet" /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pf_zq_probe2_203940/zq_off_LVN_heo1_b64_best.pt off
/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u resid_by_type.py "${TMPDIR}/p.parquet" /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pf_zq_probe2_203940/zq_on_LVN_heo1_b64_best.pt on
