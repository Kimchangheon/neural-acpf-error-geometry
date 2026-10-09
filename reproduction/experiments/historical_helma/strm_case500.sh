#!/bin/bash -l
#SBATCH --job-name=strm_case500
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/strm_case500.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/strm_case500.err
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
# Streamed: never extracts the archive, so peak disk is the tarball plus the
# compact output rather than the ~21x expanded JSON.
srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u opfdata_stream.py --case_name pglib_opf_case500_goc --root /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/opfdata
df -h ~ | tail -1
