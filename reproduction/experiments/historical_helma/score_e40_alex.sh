#!/bin/bash -l
# Re-score with each run's own training overlay on PYTHONPATH.
#
# The first pass imported the repo-root GNSMsg_SelfAttention_armijo.py, which is
# a different file from the one each run trained under (sha256 203fa4b5.. vs
# a8ac278e..), so trained weights were driven through a forward pass they were
# not trained with: raw |V| RMSE came out at about twice the training log's.
# The overlay must come first on PYTHONPATH, per run.
#SBATCH --job-name=e40_score
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/e40_score/%x_%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/e40_score/%x_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
set -euo pipefail
B=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
DATA=/home/hpc/b313dc/b313dc11/data_staging/mfk_e40_20260905/GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet
mkdir -p $B/results/e40_score
test -r "$DATA"
cd $B
for spec in mfk8:8 mfk16:16 mfk8_pfw1:8 mfk8_pfw10:8; do
  v=${spec%%:*}; k=${spec##*:}
  ov=$B/overlays/g3_${v}_e40_20260905
  ck=$B/results/g3_${v}_e40_20260905/ckpt/pignn_global_GBnetwork_g3_${v}_e40_s42_40_best_model.ckpt
  test -r "$ck" || { echo "MISSING $ck"; continue; }
  test -r "$ov/GNSMsg_SelfAttention_armijo.py" || { echo "MISSING overlay $ov"; continue; }
  echo "=========== $v (k=$k) ==========="
  echo "[code] $(sha256sum $ov/GNSMsg_SelfAttention_armijo.py)"
  PYTHONPATH=$ov:$B "$PY" -u $B/score_e40.py --parquet "$DATA" --checkpoint "$ck" \
        --manifold_k "$k" --label "$v" || echo "FAILED $v"
done
