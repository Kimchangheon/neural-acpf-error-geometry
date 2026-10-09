#!/bin/bash -l
# Frozen, no-retraining GridSFM raw-versus-CSP_16 evaluation for ppNR-v2.
# One array task evaluates one manifest grid.  The only learned quantities used
# by CSP are the train-third prediction offset and train-third NR SVD bases.
#
# Checkpoint priority exactly follows the v2 report's recovery convention:
# continuation D_v2c, then the five-large-grid v2t rescore checkpoint, then
# base D/B, and finally the staged A-tier checkpoint collection.
#
# Submit from BASE/sbatch:
#   sbatch --array=0-30%12 score_gridsfm_31grid_csp16_helma.sh
#
# No model is trained by this script.
#SBATCH --job-name=gsfm31csp16
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/gridsfm_31grid_csp16_20260907/logs/%A_%a.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/gridsfm_31grid_csp16_20260907/logs/%A_%a.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
VAULT=/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out
PY=/home/hpc/b313dc/b313dc11/conda-envs/gridfm-py312/bin/python
OUT=${BASE}/results/gridsfm_31grid_csp16_20260907
export PYTHONPATH=${BASE}:${PYTHONPATH:-}
cd "${BASE}"
mkdir -p "${OUT}/json" "${OUT}/logs" "${OUT}/checkpoint_manifest"

source "${BASE}/manifest_ppnr_v2.sh"
entry="${MANIFEST[${SLURM_ARRAY_TASK_ID:?}]}"
grid="${entry%%|*}"
rest="${entry#*|}"; buses="${rest%%|*}"
rest="${rest#*|}"; parquet="${rest%|*}"
data="${VAULT}/${parquet}"
local_data="${TMPDIR:?TMPDIR is required}/${grid}.parquet"
# The vault is visible on the login host but is not reliably mounted inside a
# compute allocation.  Copying this immutable source parquet to node-local
# storage changes neither the corpus nor the split; it is the same staging
# convention used by the original checkpoint-rescore jobs.
cp "${data}" "${local_data}" || { echo "cannot stage parquet: ${data}"; exit 2; }
trap 'rm -f "${local_data}"' EXIT

# Do not select by test metrics.  This is a fixed provenance priority, chosen
# before this PB evaluation.  It preserves the report's continuation/rescore
# rule and uses no 300-epoch or DDP checkpoints.
ck=""
for pattern in \
  "${BASE}/results/ckpt/pfv2_gridsfm_D_v2c/pfv2_gridsfm_${grid}_b*_best.pt" \
  "${BASE}/results/ckpt/pfv2_gridsfm_D_v2t/pfv2_gridsfm_${grid}_b*_best.pt" \
  "${BASE}/results/ckpt/pfv2_gridsfm_D_v2/pfv2_gridsfm_${grid}_b*_best.pt" \
  "${BASE}/results/ckpt/pfv2_gridsfm_B_v2/pfv2_gridsfm_${grid}_b*_best.pt" \
  "${BASE}/results/ckpt_from_iwi/pfv2_gridsfm_A_v2/pfv2_gridsfm_${grid}_b*_best.pt"; do
  for candidate in ${pattern}; do
    [[ -f "${candidate}" ]] || continue
    ck="${candidate}"
    break 2
  done
done
[[ -n "${ck}" ]] || { echo "missing fixed v2 checkpoint for ${grid}"; exit 3; }

name="$(basename "${ck}")"
if [[ "${name}" =~ _b([0-9]+)_best\.pt$ ]]; then
  batch="${BASH_REMATCH[1]}"
else
  echo "cannot parse original batch size from ${name}"; exit 4
fi

sha256sum "${ck}" > "${OUT}/checkpoint_manifest/${grid}.sha256"
printf 'grid=%s\nbuses=%s\nparquet=%s\ncheckpoint=%s\ncheckpoint_sha256=%s\nbatch=%s\nsplit=random_split(seed=42; train=0.3333; valid=0.3333; test=remainder)\ncalibration=train-only per-bus magnitude/circular-angle offset\nbasis=train-only NR magnitude and gauge-aligned angle SVD\nrank=16\npb=complex128 GENCO structural-zero; all bus-scenario pairs\n' \
  "${grid}" "${buses}" "${parquet}" "${ck}" "$(cut -d' ' -f1 "${OUT}/checkpoint_manifest/${grid}.sha256")" "${batch}" \
  > "${OUT}/checkpoint_manifest/${grid}.txt"

echo "[frozen-evaluation] grid=${grid} buses=${buses} batch=${batch} checkpoint=${ck}"
"${PY}" -u output_intervention_metrics.py \
  --model gridsfm --checkpoint "${ck}" --parquet "${local_data}" --rank 16 --batch "${batch}" \
  --out "${OUT}/json/${grid}.json"

"${PY}" - "${OUT}/json/${grid}.json" <<'PY'
import json, math, sys
p=json.load(open(sys.argv[1]))
m=p['result']['metrics']
for state in ('raw', 'CSP16'):
    for key in ('vmag_rmse', 'mean_pb', 'max_pb'):
        value=float(m[state][key])
        if not math.isfinite(value) or value <= 0:
            raise SystemExit(f'non-finite/non-positive {state}.{key}: {value}')
print('[audit] finite positive raw/CSP16 metrics')
PY
