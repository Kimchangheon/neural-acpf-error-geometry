#!/bin/bash -l
#SBATCH --job-name=rb_case500
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/rb_case500.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/rb_case500.err
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
# Rebuild: the first stores predate the branch-row capture and the
# contiguous-slice fix. case14 is included as the verification anchor, since
# the PyG path has results for it.
srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u opfdata_stream.py --case_name pglib_opf_case500_goc --root /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/opfdata
