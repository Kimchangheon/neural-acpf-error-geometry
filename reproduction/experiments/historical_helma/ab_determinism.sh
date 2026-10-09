#!/bin/bash -l
#SBATCH --job-name=ab_determinism
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/ab_determinism.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/ab_determinism.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
# Control: same config, same seed, twice. If these two diverge from each other
# then the lazy-vs-preload divergence is GPU kernel non-determinism, not the
# preload path changing the data.
set -uo pipefail
B=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
Q=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/pf/case300_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37930_NR_branchrows_directSI.parquet
export PYTHONPATH=$B:${PYTHONPATH:-}
cd $B
OUT=$B/results/logs/ab_preload; mkdir -p "$OUT"
COMMON="--PARQUET $Q --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
--lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 \
--EPOCHS 3 --VAL_EVERY 1 --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 \
--BATCH 64 --LR 5e-4 --log_dir $OUT --ckpt_dir $B/results/ckpt/ab_preload \
--task pf --gridfm_impl graphkit --hidden_size 48 --num_layers 12 --n_heads 8 \
--zero_init_head --vn_feature_mode log --feature_transform signed_log \
--mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh --preload_ram"
for r in R1 R2; do
  echo "############ $r (identical config, seed 42)"
  $PY -u train_valid_test_gridfm.py $COMMON --run_name det_$r > "$OUT/$r.txt" 2>&1
  echo "  rc=$?"
  grep -aE "^Epoch" "$OUT/$r.txt" | sed -E 's/.*(train loss [0-9.eE+-]+).*(mag [0-9.eE+-]+).*/\1 \2/;s/^(Epoch +0 ).*valid loss ([0-9.eE+-]+).*/\1valid \2/' | head -4
done
echo "ALL DONE"
