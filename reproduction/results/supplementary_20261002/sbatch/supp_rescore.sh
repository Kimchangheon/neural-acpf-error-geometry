#!/bin/bash -l
#SBATCH --job-name=supp_rescore
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# Supplementary rescoring: converged (+200 ep, LR 3e-5) GraphKit on six grids
# and the four GBnetwork checkpoints, at k=16/32/64/128 with every projection
# channel (C, P, CSP, Pmag, CSPmag, Pang, CSPang) and the matching NR floors.
# Scorer protocol = paper Table 1 (full state, no restore_known; verified on
# k08_s42_e120 to ~1e-7).  One array task per row below.
set -euo pipefail
B=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
W=/hnvme/workspace/b313dc11-pf_v2
OUT=$B/results/supp_rescore_20261002
export PYTHONPATH=$B:${PYTHONPATH:-}
cd $B
mkdir -p $OUT/json $OUT/floor
C=$B/results/ckpt
# name | grid | checkpoint | batch | compute floor (1/0)
ROWS=(
 "conv_case89pegase|case89pegase|$C/pfv2_graphkit_ext3e5_20260929/gkx_case89pegase_e200_lr3e5_best.pt|64|1"
 "conv_case118|case118|$C/pfv2_graphkit_ext3e5_20260929/gkx_case118_e200_lr3e5_best.pt|64|1"
 "conv_case300|case300|$C/pfv2_graphkit_ext3e5_20260929/gkx_case300_e200_lr3e5_best.pt|64|1"
 "conv_LVN_heo1|LVN_heo1|$C/pfv2_graphkit_ext3e5_20260929/gkx_LVN_heo1_e200_lr3e5_best.pt|64|1"
 "conv_case1354pegase|case1354pegase|$C/pfv2_graphkit_ext3e5_20260929/gkx_case1354pegase_e200_lr3e5_best.pt|64|1"
 "conv_case1888rte|case1888rte|$C/pfv2_graphkit_ext3e5_20260929/gkx_case1888rte_e200_lr3e5_best.pt|52|1"
 "gb_base120|GBnetwork|$C/gk_e120_20260910/k08_s42_e120_best.pt|52|1"
 "gb_lr1e4|GBnetwork|$C/gb_ext_20260921/gkx_lr1e4_best.pt|52|0"
 "gb_lr3e5|GBnetwork|$C/gb_ext_20260921/gkx_lr3e5_best.pt|52|0"
 "gb_lr1e5|GBnetwork|$C/gb_ext_20260921/gkx_lr1e5_best.pt|52|0"
)
IFS="|" read -r name grid ck batch dofloor <<< "${ROWS[${SLURM_ARRAY_TASK_ID:?}]}"
test -s "$ck"
source $B/manifest_ppnr_v2.sh
for e in "${MANIFEST[@]}"; do [[ "$e" == "$grid|"* ]] && { r="${e#*|}"; r="${r#*|}"; P="${r%|*}"; }; done
L=$TMPDIR/$grid.parquet; cp "$W/$P" "$L"
sha256sum "$ck" > $OUT/json/${name}.sha256
for k in 16 32 64 128; do
  o=$OUT/json/${name}_r${k}.json
  [ -s "$o" ] || $PY -u output_intervention_metrics.py --model graphkit --checkpoint "$ck" --parquet "$L" --rank $k --batch $batch --out "$o"
  if [ "$dofloor" = 1 ]; then
    fo=$OUT/floor/${grid}_r${k}.json
    [ -s "$fo" ] || $PY -u csp16_truth_floor.py --parquet "$L" --rank $k --batch $batch --out "$fo"
  fi
done
echo "ROWDONE $name"
