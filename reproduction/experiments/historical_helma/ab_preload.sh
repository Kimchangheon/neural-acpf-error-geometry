#!/bin/bash -l
#SBATCH --job-name=ab_preload
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/ab_preload.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/ab_preload.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

set -uo pipefail
B=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
Q=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/pf/case300_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37930_NR_branchrows_directSI.parquet
export PYTHONPATH=$B:${PYTHONPATH:-}
cd $B
OUT=$B/results/logs/ab_preload; mkdir -p "$OUT" "$B/results/ckpt/ab_preload"
ls -la "$Q" || { echo "PARQUET MISSING"; exit 1; }

COMMON="--PARQUET $Q --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
--lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 \
--EPOCHS 4 --VAL_EVERY 1 --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 \
--BATCH 64 --LR 5e-4 --log_dir $OUT --ckpt_dir $B/results/ckpt/ab_preload \
--task pf --gridfm_impl graphkit --hidden_size 48 --num_layers 12 --n_heads 8 \
--zero_init_head --vn_feature_mode log --feature_transform signed_log \
--mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh"

echo "############ A: lazy parquet (baseline)"
t0=$SECONDS
$PY -u train_valid_test_gridfm.py $COMMON --run_name ab_lazy > "$OUT/A.txt" 2>&1
echo "  rc=$? WALL_A=$((SECONDS-t0))s"
grep -aE "^\[preload\]|^Epoch|Traceback|Error" "$OUT/A.txt" | tail -8

echo "############ B: --preload_ram"
t0=$SECONDS
$PY -u train_valid_test_gridfm.py $COMMON --run_name ab_preload --preload_ram > "$OUT/B.txt" 2>&1
echo "  rc=$? WALL_B=$((SECONDS-t0))s"
grep -aE "^\[preload\]|^Epoch|Traceback|Error" "$OUT/B.txt" | tail -10
echo "ALL DONE"
