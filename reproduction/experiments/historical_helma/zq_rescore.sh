#!/bin/bash -l
#SBATCH --job-name=zq_rescore
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/zq_rescore.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/zq_rescore.err
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
cp "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/pf/LVN_heo1_ppcY_backbone_dc_compile_cNR_ls0.60-1.40_u0clean_siNR_36000_NR_branchrows_directSI.parquet" "${TMPDIR}/p.parquet"
for a in off on; do
  echo "=== arm $a"
  srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u rescore_pf.py --PARQUET "${TMPDIR}/p.parquet" \
    --ckpt "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pf_zq_probe2_203940/zq_${a}_LVN_heo1_b64_best.pt" \
    --grid LVN_heo1 --impl mirror --BATCH 32 \
    --json_out "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pf_zq_probe2_203940/rescore_${a}.json"
done
