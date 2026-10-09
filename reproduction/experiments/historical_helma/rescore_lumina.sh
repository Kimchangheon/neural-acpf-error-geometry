#!/bin/bash -l
#SBATCH --job-name=rescore_lumina
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/rescore_lumina.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/rescore_lumina.err
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

set -uo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC

CKPTS=(
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_lumina_B_v2/pfv2_lumina_case39_b32_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_lumina_B_v2/pfv2_lumina_case57_b32_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_lumina_B_v2/pfv2_lumina_case89pegase_b32_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_lumina_B_v2/pfv2_lumina_SimBench_b32_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_lumina_B_v2/pfv2_lumina_case118_b32_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_lumina_B_v2/pfv2_lumina_case145_b32_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_lumina_B_v2/pfv2_lumina_iceland_b32_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_lumina_B_v2/pfv2_lumina_case_illinois200_b32_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_lumina_D_v2/pfv2_lumina_GBnetwork_b32_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_lumina_D_v2/pfv2_lumina_case2848rte_b32_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_lumina_D_v2/pfv2_lumina_case2869pegase_b32_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_lumina_D_v2/pfv2_lumina_case3120sp_b32_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_lumina_D_v2/pfv2_lumina_ENTSO_E_RealGridTest_b32_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_lumina_D_v2/pfv2_lumina_case6470rte_b32_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_lumina_D_v2/pfv2_lumina_case6495rte_b32_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_lumina_D_v2/pfv2_lumina_case6515rte_b32_best.pt"
  "/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/pfv2_lumina_D_v2t/pfv2_lumina_case9241pegase_b32_best.pt"
)
for ck in "${CKPTS[@]}"; do
  n=$(basename "$ck"); n=${n%_best.pt}; n=${n%_best_model.ckpt}
  rest=${n#pfv2_lumina_}                       # <grid>_b<N>[_<EPOCHS>]
  rest=$(echo "$rest" | sed -E "s/_[0-9]+$//")
  grid=${rest%_b*}; batch=${rest##*_b}
  file=$(awk -F'\t' -v g="$grid" '$1==g{print $2}' /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/rescore_map_lumina.tsv)
  if [ -z "$file" ] || [ ! -s "/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/$file" ]; then
    echo "### SKIP $grid (no parquet)"; continue
  fi
  prev=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/pfv2_lumina_rescore/rescore_lumina_${grid}_training_log.txt
  if [ -s "$prev" ] && grep -aq "| test loss\|Final test-set RMSE" "$prev"; then
    echo "### ALREADY SCORED $grid"; continue
  fi
  echo "### RESCORE $grid batch $batch from $ck"
  mkdir -p "${TMPDIR}/ckpt_unused"
  cp "/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/$file" "${TMPDIR}/$grid.parquet"
  srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test_lumina.py \
    --PARQUET "${TMPDIR}/$grid.parquet" \
    --run_name rescore_lumina_${grid} \
    --log_to_file --log_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/pfv2_lumina_rescore --ckpt_dir ${TMPDIR}/ckpt_unused \
    --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
    --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 \
    --no_preload_ram \
    --BATCH $batch --EPOCHS 0 --LR 1e-4 --VAL_EVERY 1 \
    --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 \
    --init_checkpoint "$ck" \
    --task pf --init_mode scratch --mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh --model_config /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/lumina_ckpt/lumina_config.json || echo "### FAILED $grid"
  rm -f "${TMPDIR}/$grid.parquet"
done
echo "### DONE lumina"
