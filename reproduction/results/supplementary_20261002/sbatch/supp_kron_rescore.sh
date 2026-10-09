#!/bin/bash -l
#SBATCH --job-name=supp_kron
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# LVN_heo1 Kron-reduced GraphKit (120 ep, and +200 ep at LR 3e-5), scored on
# all 722 buses (GK_KRON=1 expands V_e = R V_k), k=16..128, every channel.
# Floors are those of the unreduced grid (same NR references).
set -euo pipefail
B=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
W=/hnvme/workspace/b313dc11-pf_v2
OUT=$B/results/supp_rescore_20261002
export PYTHONPATH=$B:${PYTHONPATH:-} GK_KRON=1
cd $B
ROWS=("kron120|$B/results/ckpt/heo1_kron_20261001/heo1kron_base_best.pt"
      "kron_e200|$B/results/ckpt/heo1_kron_20261001/heo1kron_base_e200_lr3e5_best.pt")
IFS="|" read -r name ck <<< "${ROWS[${SLURM_ARRAY_TASK_ID:?}]}"
test -s "$ck"
source $B/manifest_ppnr_v2.sh
for e in "${MANIFEST[@]}"; do [[ "$e" == "LVN_heo1|"* ]] && { r="${e#*|}"; r="${r#*|}"; P="${r%|*}"; }; done
L=$TMPDIR/lvn.parquet; cp "$W/$P" "$L"
sha256sum "$ck" > $OUT/json/${name}_LVN_heo1.sha256
for k in 16 32 64 128; do
  o=$OUT/json/${name}_LVN_heo1_r${k}.json
  [ -s "$o" ] || $PY -u output_intervention_metrics.py --model graphkit --checkpoint "$ck" --parquet "$L" --rank $k --batch 64 --out "$o"
done
echo "ROWDONE $name"
