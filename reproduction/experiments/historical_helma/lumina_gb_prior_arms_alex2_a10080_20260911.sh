#!/bin/bash -l
# Independent Alex2 A100-80 GBnetwork reproductions of Helma LUMINA arms:
# 836038 control, 836040 output-anchor-only, 836039 full prior-feature arm.
# Every task is scratch seed 42 and writes to its own directory below OUT.
#SBATCH --job-name=lum_gb_prior
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/lumina_gb_prior_arms_alex_20260911/slurm/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/lumina_gb_prior_arms_alex_20260911/slurm/%A_%a.err

set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
SOURCE=/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet
DATA=${TMPDIR:?TMPDIR is required}/GBnetwork.parquet
OUT=${BASE}/results/lumina_gb_prior_arms_alex_20260911

names=(control anchor_prior anchor_prior_z_ft)
flags=(
  ''
  '--v_anchor prior --vmag_prior_angle start'
  '--v_anchor prior --bus_vmag_prior chord --vmag_prior_angle start --init_recipe sd002_zerohead'
)
i=${SLURM_ARRAY_TASK_ID:?}
name=${names[$i]}
extra=${flags[$i]}
RUNOUT=${OUT}/${name}

mkdir -p "${OUT}/slurm" "${RUNOUT}/logs" "${RUNOUT}/ckpt"
test -r "${SOURCE}"
test -r "${BASE}/lumina_ckpt/lumina_config_do0.json"
cp "${SOURCE}" "${DATA}"
test "$(stat -c %s "${SOURCE}")" = "$(stat -c %s "${DATA}")"

export PYTHONPATH="${BASE}:${PYTHONPATH:-}"
cd "${BASE}"
echo "[arm] ${name}; Helma counterpart: $([ "${name}" = control ] && echo 836038 || ([ "${name}" = anchor_prior ] && echo 836040 || echo 836039))"
echo "[output] ${RUNOUT}"
srun "${PY}" -u train_valid_test_lumina.py \
  --PARQUET "${DATA}" --run_name "gbv_${name}_alex" \
  --log_to_file --log_dir "${RUNOUT}/logs" --ckpt_dir "${RUNOUT}/ckpt" \
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 \
  --BATCH 32 --EPOCHS 120 --LR 1e-3 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 --split_seed 42 \
  --task pf --init_mode scratch --physics_loss_form logcosh \
  --model_config "${BASE}/lumina_ckpt/lumina_config_do0.json" \
  --mask_known_v --mse_weight 1.0 --physics_weight 1e-2 \
  --bus_physics_features --theta_anchor start ${extra}
