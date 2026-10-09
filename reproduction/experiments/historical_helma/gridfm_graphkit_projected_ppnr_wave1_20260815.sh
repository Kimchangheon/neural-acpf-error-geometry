#!/bin/bash -l
#SBATCH --job-name=gkproj_allpf
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/gridfm_graphkit_projected_ppnr_wave1_20260815_%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/gridfm_graphkit_projected_ppnr_wave1_20260815_%A_%a.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-29%6

set -euo pipefail

BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
DATA=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/data/pf
GROUP=gridfm_graphkit_projected_ppnr_wave1_20260815

# Canonical Phase-0 pandapower/Newton corpus. LVN_heo1 is intentionally absent:
# its oracle file is tagged _cNR_, not _ppNR_.
MANIFEST=(
  "case4gs|4|case4gs_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet"
  "case5|5|case5_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet"
  "case6ww|6|case6ww_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet"
  "case9|9|case9_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet"
  "case14|14|case14_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet"
  "case24_ieee_rts|24|case24_ieee_rts_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet"
  "GBreducednetwork|29|GBreducednetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet"
  "case30|30|case30_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet"
  "case_ieee30|30|case_ieee30_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet"
  "case33bw|33|case33bw_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet"
  "case39|39|case39_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet"
  "case57|57|case57_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet"
  "case89pegase|89|case89pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet"
  "SimBench|94|SimBench_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_siNR_36000_NR_branchrows_directSI.parquet"
  "case118|118|case118_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet"
  "case145|145|case145_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet"
  "iceland|189|iceland_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet"
  "case_illinois200|200|case_illinois200_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet"
  "case300|300|case300_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet"
  "case1354pegase|1354|case1354pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_32627_NR_branchrows_directSI_rg20.parquet"
  "case1888rte|1888|case1888rte_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet"
  "GBnetwork|2224|GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_34424_NR_branchrows_directSI_rg20.parquet"
  "case2848rte|2848|case2848rte_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet"
  "case2869pegase|2869|case2869pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_35998_NR_branchrows_directSI_rg20.parquet"
  "case3120sp|3120|case3120sp_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_35465_NR_branchrows_directSI_rg20.parquet"
  "ENTSO_E_RealGridTest|6051|ENTSO_E_RealGridTest_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_siNR_36000_NR_branchrows_directSI.parquet"
  "case6470rte|6470|case6470rte_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_25551_NR_branchrows_directSI_rg20.parquet"
  "case6495rte|6495|case6495rte_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_23332_NR_branchrows_directSI_rg20.parquet"
  "case6515rte|6515|case6515rte_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_20135_NR_branchrows_directSI_rg20.parquet"
  "case9241pegase|9241|case9241pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_18280_NR_branchrows_directSI_rg20.parquet"
)

IFS='|' read -r grid buses file <<< "${MANIFEST[$SLURM_ARRAY_TASK_ID]}"

# Preserve the measured pilot envelope of roughly 2,700 bus-nodes/batch.
batch=$((2700 / buses))
(( batch < 1 )) && batch=1
(( batch > 16 )) && batch=16

if (( buses < 300 )); then
  max_train=1024; max_valid=256; max_test=256
elif (( buses < 1000 )); then
  max_train=512; max_valid=128; max_test=128
else
  max_train=256; max_valid=64; max_test=64
fi

source_parquet=${DATA}/${file}
task_scratch=${TMPDIR}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
local_parquet=${task_scratch}/${file}
run_name=${GROUP}_${grid}_w1_b${batch}_s42

mkdir -p "${task_scratch}" "${BASE}/results/logs/${GROUP}" "${BASE}/results/ckpt/${GROUP}"
test -s "${source_parquet}"
test -f "${source_parquet}"
test ! -L "${source_parquet}"
test "$(realpath "$(dirname "${source_parquet}")")" = "$(realpath "${DATA}")"
cp "${source_parquet}" "${local_parquet}"
test "$(stat -c %s "${source_parquet}")" = "$(stat -c %s "${local_parquet}")"
# Home is only a compute-visible staging cache; vault retains the authoritative
# source. Reclaim the cache copy after the byte-count-verified node-local copy.
rm -- "${source_parquet}"

export PYTHONPATH=${BASE}:${PYTHONPATH:-}
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${BASE}"

echo "[gpu] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "[grid] ${grid}, buses=${buses}, batch=${batch}"
echo "[samples] ${max_train}/${max_valid}/${max_test}, epochs=20, seed=42"
echo "[model] released GraphKit 48x12x8 with fused projected DPF operator, physics_weight=1"
echo "[staging] verified node-local copy; removed compute-visible cache source"

srun "${PY}" -u train_valid_test_gridfm.py \
  --PARQUET "${local_parquet}" --task pf --gridfm_impl graphkit_projected \
  --run_name "${run_name}" --log_to_file \
  --log_dir "${BASE}/results/logs/${GROUP}" \
  --ckpt_dir "${BASE}/results/ckpt/${GROUP}" \
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \
  --lazy_parquet --row_group_cache_size 4 --dataset_complex_dtype complex128 \
  --BATCH "${batch}" --EPOCHS 20 --LR 5e-4 --VAL_EVERY 1 \
  --train_ratio 0.3333 --valid_ratio 0.3333 \
  --max_train_samples "${max_train}" \
  --max_valid_samples "${max_valid}" \
  --max_test_samples "${max_test}" \
  --hidden_size 48 --num_layers 12 --n_heads 8 --dropout 0.0 \
  --feature_transform signed_log --vn_feature_mode log \
  --mse_weight 1.0 --physics_weight 1.0 \
  --physics_loss_form logcosh --convergence_tol_pu 1e-3 \
  --armijo_max_backtracks 8 \
  --projected_dpf_lr 0.003377 \
  --projected_dpf_beta1 0.979681 --projected_dpf_beta2 0.963442 \
  --projected_dpf_mu 1e-4 \
  --projected_dpf_max_vm_fraction 0.10 \
  --projected_dpf_max_angle_step 0.30 \
  --seed_value 42
