#!/bin/bash -l
#SBATCH --job-name=mask_lumina_sp
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/mask_lumina_sp.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/mask_lumina_sp.err
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

# Retrain of the v2 GBnetwork baseline with ONE addition: --mask_known_v.
# AC power flow specifies |V| at PV and slack and theta at slack; only the rest
# is solved for.  PIGNN enforces that inside its iteration and its PV/slack
# buses are exact for free; GridSFM feeds the setpoint into its voltage head and
# GraphKit writes it into the bus row, but neither HOLDS it, and LUMINA does not
# carry it on the bus at all.  On IEEE-14 giving LUMINA the setpoint as an input
# changed its |V| RMSE by nothing (.02321 vs .02321), which is why the value is
# now held rather than merely supplied.  Nothing held is a solved quantity, so
# no label leaks.  Everything else is copied from the v2 script verbatim.
srun /home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python -u train_valid_test_lumina.py \
  --PARQUET "${LOCAL_PARQUET}" \
  --run_name mask_lumina_sp \
  --log_to_file --log_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/logs/mask_20260908 --ckpt_dir /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/ckpt/mask_20260908 \
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 \
  --dataset_complex_dtype complex128 \
  --BATCH 32 --EPOCHS 40 --LR 1e-4 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 \
  --task pf --mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh \
  --mask_known_v --init_mode scratch --bus_inject_features --model_config /home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/lumina_ckpt/lumina_config.json
