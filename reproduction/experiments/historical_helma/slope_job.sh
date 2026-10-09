#!/bin/bash -l
#SBATCH --job-name=slope_b
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/slope_b.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/slope_b.err
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
set -uo pipefail
B=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
V=/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out
export PYTHONPATH=$B:${PYTHONPATH:-}; cd $B
mkdir -p results/slope
source manifest_ppnr_v2.sh
for e in "${MANIFEST[@]}"; do
  g="${e%%|*}"; rest="${e#*|}"; rest="${rest#*|}"; file="${rest%|*}"
  [ -f "$V/$file" ] || { echo "SKIP $g (no parquet)"; continue; }
  for m in graphkit gridsfm lumina; do
    [ -f "results/slope/${m}_${g}.json" ] && continue
    # Prefer a rerun group (v2c/v2r/v2t/v2e) over the base v2 run.
    ck=""
    for grp in D_v2c A_v2r D_v2t D_v2e A_v2 B_v2 C_v2 D_v2; do
      c=$(ls results/ckpt/pfv2_${m}_${grp}/pfv2_${m}_${g}_b*_best.pt 2>/dev/null | head -1)
      [ -n "$c" ] && { ck="$c"; break; }
    done
    [ -z "$ck" ] && { echo "SKIP $m $g (no ckpt)"; continue; }
    echo "### $m $g  <- $(basename $(dirname $ck))"
    timeout 1800 $PY -u rescore_slope_r2.py --model $m --PARQUET "$V/$file" --grid "$g" \
      --ckpt "$ck" --model_config $B/lumina_ckpt/lumina_config.json \
      --max_test_samples 3000 --json_out "results/slope/${m}_${g}.json" 2>&1 \
      | grep -aE "RESCORE|Error|rror:" | cut -c1-260
  done
done
echo SLOPE_ALL_DONE
