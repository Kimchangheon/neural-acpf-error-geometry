#!/bin/bash -l
# Alex2 reproduction of Helma job 836041 (gbv06z_control_zerohead).
# This is intentionally the same single-seed, scratch, 120-epoch arm:
# no magnitude anchor, no chord-prior input feature, sd002 + zero terminal head.
#SBATCH --job-name=gbv06z_zerohead
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/lumina_gbv06z_control_zerohead_alex_20260911/logs/%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/lumina_gbv06z_control_zerohead_alex_20260911/logs/%j.err

set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
SOURCE=/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet
DATA=${TMPDIR:?TMPDIR is required}/GBnetwork.parquet
OUT=${BASE}/results/lumina_gbv06z_control_zerohead_alex_20260911

mkdir -p "${OUT}/logs" "${OUT}/ckpt"
test -r "${SOURCE}"
test -r "${BASE}/lumina_ckpt/lumina_config_do0.json"
cp "${SOURCE}" "${DATA}"
test "$(stat -c %s "${SOURCE}")" = "$(stat -c %s "${DATA}")"

export PYTHONPATH="${BASE}:${PYTHONPATH:-}"
cd "${BASE}"
echo "[reproduction] Helma 836041 gbv06z_control_zerohead"
srun "${PY}" -u train_valid_test_lumina.py \
  --PARQUET "${DATA}" --run_name gbv06z_control_zerohead_alex \
  --log_to_file --log_dir "${OUT}/logs" --ckpt_dir "${OUT}/ckpt" \
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 \
  --BATCH 32 --EPOCHS 120 --LR 1e-3 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 --split_seed 42 \
  --task pf --init_mode scratch --physics_loss_form logcosh \
  --model_config "${BASE}/lumina_ckpt/lumina_config_do0.json" \
  --mask_known_v --mse_weight 1.0 --physics_weight 1e-2 \
  --bus_physics_features --theta_anchor start --init_recipe sd002_zerohead
