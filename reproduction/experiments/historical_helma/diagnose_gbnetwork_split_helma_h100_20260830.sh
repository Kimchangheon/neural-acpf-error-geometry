#!/bin/bash -l
# Moved to helma H100: the alex a40 partition is under maintenance
# (ReqNodeNotAvail).  Checkpoints were copied across; the parquet, the split
# and every flag are unchanged.
#
# Two measurements aimed at the two things the roughness run did NOT settle.
#
# (1) Local vs coupling split of the mismatch.  Writing
#     dS_i = [S_i^set - V_i conj(Y_ii V_i)] - V_i conj(sum_{j!=i} Y_ij V_j),
#     a spatially smooth error cancels in the second term but not the first.
#     Roughness can therefore only ever explain the coupling part, and the
#     raw run left a gap: the constant predictor is 2.0x smoother and 2.2x
#     better on admittance-weighted roughness but 9.5x better on mean|dQ|.
#     This says whether the missing factor is local.  It is also the obvious
#     candidate for the GridSFM anomaly -- weighted edge roughness 15x any
#     other model, yet the smallest residual of any model -- which broke the
#     admittance-weighted correlation (r=+0.10 with it, +0.68 without).
#
# (2) In-manifold fraction.  The sharp version of "smooth" is not slow
#     variation but membership: is the error a voltage pattern this grid can
#     actually produce?  The constant predictor errs by exactly minus the
#     scenario deviation, so its error lies entirely in the span of the
#     training-split reference variation, by construction.  A learned
#     prediction has no such guarantee.  Basis is the top-32 right singular
#     vectors of the centred training reference matrix; a random error would
#     score k/N = 32/2224 = 1.4%.
#
# (roughness header follows)
# Voltage-error roughness.  The constant per-bus predictor beats every model
# on median |dP|/|dQ| despite a larger |V| error, and the |Y_ii| breakdown
# shows the gap growing monotonically with stiffness (models win 5x below
# |Y_ii|=1e2 on P, lose 128x above 1e5).  The hypothesis is that the residual
# is a difference operator: a voltage error shared by neighbours cancels,
# one that differs is multiplied by the branch admittance.  This run measures
# rms|eps_i - eps_j| against rms|eps_i| directly; the ratio is ~sqrt(2) for
# per-bus noise and ~0 for a smooth field.
#
# (round 2 header follows)
# Round 2: the same six checkpoints plus the per-bus constant baseline, with
# three diagnostics the first round did not have.
#
#  * per-edge dtheta_ij, and the gauge decomposition of the bus-angle error.
#    Flows depend on theta_i - theta_j, so a model whose angles carry a
#    scenario-dependent common offset posts a large per-bus theta RMSE while
#    predicting every flow correctly.  Round 1 could not tell that apart, and
#    the baseline run made it urgent: the constant predictor loses to the
#    models on per-bus theta by 7x yet beats them on median |dP| by 12x.
#  * pooled regression slope/R2 (GridSFM section 6.2 as published) alongside
#    the per-bus version (the same regression on each bus deviation from its
#    own mean), because pooled R2 on a single-grid surrogate is inflated by
#    the memorisable static voltage profile.
#  * 1 - SSE/SST against the per-bus constant predictor, so the number the
#    report tables call R^2 is computed by the same code instead of by hand.
#SBATCH --job-name=gb_split
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/residual_split_20260829/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/residual_split_20260829/logs/%A_%a.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --array=0-7%4

set -euo pipefail
trap 'rc=$?; echo "[failure] line=${LINENO} command=${BASH_COMMAND} rc=${rc}" >&2; exit "${rc}"' ERR

BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
OVERLAY=${BASE}/overlays/residual_split_20260829
CKPTS=${OVERLAY}
DATA=/home/hpc/b313dc/b313dc11/data_staging/grl_oegr_gate_20260825/GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet
OUTDIR=${BASE}/results/residual_split_20260829
LOCAL_PARQUET=${TMPDIR:?TMPDIR is required}/GBnetwork.parquet

models=(g1 g2 g3 g4 g5 gridsfm baseline_perbus g3)
checkpoints=(
  ${CKPTS}/pignn_global_GBnetwork_g1_s42_40_best_model.ckpt
  ${CKPTS}/pignn_global_GBnetwork_g2_s42_40_best_model.ckpt
  ${CKPTS}/pignn_global_GBnetwork_g3_s42_40_best_model.ckpt
  ${CKPTS}/pignn_global_GBnetwork_g4_s42_40_best_model.ckpt
  ${CKPTS}/pignn_global_GBnetwork_g5_s42_40_best_model.ckpt
  ${CKPTS}/pfv2_gridsfm_GBnetwork_b26_best.pt
  none
  ${CKPTS}/pignn_global_GBnetwork_g3_mse_s42_40_best_model.ckpt
)

model=${models[${SLURM_ARRAY_TASK_ID:?}]}
checkpoint=${checkpoints[${SLURM_ARRAY_TASK_ID}]}
run=${model}_GBnetwork_detailed
[ "${SLURM_ARRAY_TASK_ID}" = "7" ] && run=g3_mse_GBnetwork_detailed
mkdir -p "${OUTDIR}/json" "${OUTDIR}/logs"
test -r "${DATA}"; test -x "${PY}"; test -r "${OVERLAY}/diagnose_residual_distributions.py"
test -r "${OVERLAY}/prediction_diagnostics.py"
if [ "${checkpoint}" != "none" ]; then test -r "${checkpoint}"; fi
cp "${DATA}" "${LOCAL_PARQUET}"
test "$(stat -c %s "${LOCAL_PARQUET}")" = "$(stat -c %s "${DATA}")"
export PYTHONPATH=${OVERLAY}:${BASE}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${BASE}"
echo "[diagnostic] model=${model} checkpoint=${checkpoint}"
echo "[data] GBnetwork 37022 rows; random_split 1/3,1/3,1/3 seed=42; test residuals complex128"
echo "[gpu] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
srun "${PY}" -u "${OVERLAY}/diagnose_residual_distributions.py" \
  --model "${model}" --checkpoint "${checkpoint}" --parquet "${LOCAL_PARQUET}" \
  --batch 23 --output "${OUTDIR}/json/${run}.json"
