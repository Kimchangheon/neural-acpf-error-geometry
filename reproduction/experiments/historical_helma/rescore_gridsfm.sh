#!/bin/bash -l
#SBATCH --job-name=rescore_gridsfm
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/rescore_gridsfm.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/rescore_gridsfm.err
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

set -uo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC

CKPTS=(
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_gridsfm_B_v2/pfv2_gridsfm_case39_b32_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_gridsfm_B_v2/pfv2_gridsfm_case57_b32_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_gridsfm_B_v2/pfv2_gridsfm_case89pegase_b32_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_gridsfm_B_v2/pfv2_gridsfm_SimBench_b32_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_gridsfm_B_v2/pfv2_gridsfm_case118_b32_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_gridsfm_B_v2/pfv2_gridsfm_case145_b32_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_gridsfm_B_v2/pfv2_gridsfm_iceland_b32_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_gridsfm_B_v2/pfv2_gridsfm_case_illinois200_b32_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_gridsfm_D_v2/pfv2_gridsfm_GBnetwork_b26_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_gridsfm_D_v2/pfv2_gridsfm_case2848rte_b20_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_gridsfm_D_v2/pfv2_gridsfm_case2869pegase_b20_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_gridsfm_D_v2/pfv2_gridsfm_case3120sp_b18_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_gridsfm_D_v2t/pfv2_gridsfm_ENTSO_E_RealGridTest_b9_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_gridsfm_D_v2t/pfv2_gridsfm_case6470rte_b9_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_gridsfm_D_v2t/pfv2_gridsfm_case6495rte_b9_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_gridsfm_D_v2t/pfv2_gridsfm_case6515rte_b9_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_gridsfm_D_v2t/pfv2_gridsfm_case9241pegase_b6_best.pt"
)
for ck in "${CKPTS[@]}"; do
  n=$(basename "$ck"); n=${n%_best.pt}; n=${n%_best_model.ckpt}
  rest=${n#pfv2_gridsfm_}                       # <grid>_b<N>[_<EPOCHS>]
  rest=$(echo "$rest" | sed -E "s/_[0-9]+$//")
  grid=${rest%_b*}; batch=${rest##*_b}
  file=$(awk -F'\t' -v g="$grid" '$1==g{print $2}' /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/rescore_map_gridsfm.tsv)
  if [ -z "$file" ] || [ ! -s "/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/$file" ]; then
    echo "### SKIP $grid (no parquet)"; continue
  fi
  prev=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/pfv2_gridsfm_rescore/rescore_gridsfm_${grid}_training_log.txt
  if [ -s "$prev" ] && grep -aq "| test loss\|Final test-set RMSE" "$prev"; then
    echo "### ALREADY SCORED $grid"; continue
  fi
  echo "### RESCORE $grid batch $batch from $ck"
  mkdir -p "${TMPDIR}/ckpt_unused"
  cp "/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/$file" "${TMPDIR}/$grid.parquet"
  srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test_gridsfm.py \
    --PARQUET "${TMPDIR}/$grid.parquet" \
    --run_name rescore_gridsfm_${grid} \
    --log_to_file --log_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/pfv2_gridsfm_rescore --ckpt_dir ${TMPDIR}/ckpt_unused \
    --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
    --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 \
    --no_preload_ram \
    --BATCH $batch --EPOCHS 0 --LR 1e-4 --VAL_EVERY 1 \
    --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 \
    --init_checkpoint "$ck" \
    --task pf --init_mode scratch --mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh || echo "### FAILED $grid"
  rm -f "${TMPDIR}/$grid.parquet"
done
echo "### DONE gridsfm"
