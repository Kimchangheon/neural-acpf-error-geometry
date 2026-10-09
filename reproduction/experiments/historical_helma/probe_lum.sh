#!/bin/bash -l
#SBATCH --job-name=probe_lum
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/probe_lum.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/probe_lum.err
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
V=/home/vault/b313dc/b313dc11/data/pf
BIG=case9241pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_18280_NR_branchrows_directSI_rg20.parquet
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
echo "### case1354pegase"
cp "$V/case1354pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_32627_NR_branchrows_directSI_rg20.parquet" "${TMPDIR}/p.parquet"
srun $PY -u probe_batch_memory.py --model lumina --model_config /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/lumina_ckpt/lumina_config.json --PARQUET "${TMPDIR}/p.parquet" --batches 1,2,4,8,16 || true
rm -f "${TMPDIR}/p.parquet"
echo "### case9241pegase"
cp "$V/$BIG" "${TMPDIR}/b.parquet"
srun $PY -u probe_batch_memory.py --model lumina --model_config /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/lumina_ckpt/lumina_config.json --PARQUET "${TMPDIR}/b.parquet" --batches 1,2,4 || true
