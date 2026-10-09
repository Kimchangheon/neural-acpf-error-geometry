#!/bin/bash -l
# Post-hoc manifold projection, k sweep.
#
# The residual responds to the out-of-manifold part of the voltage error and is
# uncorrelated with the in-manifold part (corr +0.97 against -0.16 over eight
# configurations, helma job 802537).  Rather than penalise that component, this
# removes it by construction: the prediction is replaced by its projection onto
# the rank-k basis of the training-split reference variation,
#     v_proj = vbar + U_k U_k^T (v - vbar),
# and the circular equivalent for theta.  No test data enters the basis and no
# model is retrained; this is the rank-k generalisation of the per-bus
# calibration already validated.
#
# ACCEPTANCE CHECK: the projection is a hard constraint and cannot reduce the
# in-manifold error, so |V| RMSE after projection must equal the rms_in already
# measured -- 1.945e-02 for G3, 2.181e-02 for GridSFM, 2.331e-02 for the
# constant predictor at k=32.  If it does not, the implementation is wrong and
# the residual numbers must not be used.
#
# The sweep exists because k was never tuned: larger k represents more and
# suppresses less, and at k=N the projection is the identity.
#SBATCH --job-name=gb_project
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/residual_project_20260830/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/residual_project_20260830/logs/%A_%a.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --array=0-11%4

set -euo pipefail
trap 'rc=$?; echo "[failure] line=${LINENO} command=${BASH_COMMAND} rc=${rc}" >&2; exit "${rc}"' ERR

BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
OVERLAY=${BASE}/overlays/residual_split_20260829
DATA=/home/hpc/b313dc/b313dc11/data_staging/grl_oegr_gate_20260825/GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet
OUTDIR=${BASE}/results/residual_project_20260830
LOCAL_PARQUET=${TMPDIR:?TMPDIR is required}/GBnetwork.parquet

models=(g3 gridsfm baseline_perbus)
ckpts=(
  ${OVERLAY}/pignn_global_GBnetwork_g3_s42_40_best_model.ckpt
  ${OVERLAY}/pfv2_gridsfm_GBnetwork_b26_best.pt
  none
)
ks=(8 16 32 64)

t=${SLURM_ARRAY_TASK_ID:?}
mi=$((t / 4)); ki=$((t % 4))
model=${models[$mi]}; checkpoint=${ckpts[$mi]}; K=${ks[$ki]}
run=${model}_k${K}

mkdir -p "${OUTDIR}/json" "${OUTDIR}/logs"
test -r "${DATA}"; test -x "${PY}"; test -r "${OVERLAY}/diagnose_residual_distributions.py"
if [ "${checkpoint}" != "none" ]; then test -r "${checkpoint}"; fi
[ -f "${OUTDIR}/json/${run}.json" ] && { echo "HAVE ${run}"; exit 0; }
cp "${DATA}" "${LOCAL_PARQUET}"
test "$(stat -c %s "${LOCAL_PARQUET}")" = "$(stat -c %s "${DATA}")"
export PYTHONPATH=${OVERLAY}:${BASE}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${BASE}"
echo "[projection] model=${model} k=${K}"
echo "[data] GBnetwork 37022 rows; random_split 1/3,1/3,1/3 seed=42; basis from the TRAIN third only"
srun "${PY}" -u "${OVERLAY}/diagnose_residual_distributions.py" \
  --model "${model}" --checkpoint "${checkpoint}" --parquet "${LOCAL_PARQUET}" \
  --batch 23 --project_k "${K}" --output "${OUTDIR}/json/${run}.json"
echo "RUN_DONE ${run}"
