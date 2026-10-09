#!/bin/bash -l
#SBATCH --job-name=ctrl_lumina_bkvlog
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/ctrl_lumina_bkvlog.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/ctrl_lumina_bkvlog.err
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
set -euo pipefail
export PYTHONPATH=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:${PYTHONPATH:-}
cd /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
LOCAL_PARQUET="${TMPDIR:?}/GBnetwork.parquet"
DATA=/home/hpc/b313dc/b313dc11/pb_data/GBnetwork.parquet
cp "$DATA" "${LOCAL_PARQUET}"
test "$(stat -c %s "${LOCAL_PARQUET}")" = "$(stat -c %s "$DATA")"

# Controlled repeat of pfv2_lumina_GBnetwork_b32 with ONE change:
# --bus_base_kv_mode log.  The MODEL is untouched -- this is adapter-side only,
# unlike ctrl_lumina_inln, which adds a LayerNorm the official architecture does
# not have and is kept only as a reference arm.
#
# Why: LUMINA has no input normalisation anywhere -- not in the model
# (Linear->ReLU) and not in the SDK's data pipeline.  On its HDF5 path the
# question never arises, because H5Bus carries no base_kv field at all and the
# loader fills that column with a constant 1.0.  We were feeding rated kV there
# instead: measured on a real batch it spans 6.6-400 (std 124.8) against every
# other bus column at 0-1.5, and carries zero scenario-to-scenario variation on
# a fixed grid.  "unit" reproduces the official HDF5 distribution exactly; "log"
# keeps the voltage-level information but at the scale of the other columns
# (0.82-2.60, std 0.38), the way PIGNN (vn_log) and graphkit
# (--vn_feature_mode log) both take it.
srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test_lumina.py \
  --PARQUET "${LOCAL_PARQUET}" \
  --run_name ctrl_lumina_bkvlog \
  --log_to_file --log_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/ctrl_20260906 --ckpt_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/ctrl_20260906 \
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 \
  --dataset_complex_dtype complex128 \
  --BATCH 32 --EPOCHS 40 --LR 1e-4 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 \
  --task pf --init_mode scratch --bus_base_kv_mode log \
  --mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh \
  --model_config /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/lumina_ckpt/lumina_config.json
