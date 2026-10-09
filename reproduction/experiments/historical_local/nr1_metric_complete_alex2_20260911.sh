#!/bin/bash -l
# Re-score the frozen NR1 test caches with all Table-1 metrics.  No model
# inference, training, data synthesis, or eta selection happens here.
#SBATCH --job-name=nr1_complete
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr1_multiproc_gbnetwork_20260910/logs/complete_%A_%a.out
#SBATCH --error=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr1_multiproc_gbnetwork_20260910/logs/complete_%A_%a.err
set -euo pipefail
BASE=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
O=$BASE/overlays/nr1_multiproc_gbnetwork_20260910
R=$BASE/results/nr1_multiproc_gbnetwork_20260910
DATA=/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out/GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet
PY=/home/hpc/iwi5/iwi5295h/conda-envs/gridfm-py312/bin/python
families=(g3 g3 g3 gridsfm gridsfm gridsfm); seeds=(41 42 43 41 42 43)
i=${SLURM_ARRAY_TASK_ID:?}; m=${families[$i]}; s=${seeds[$i]}
ETA=$("$PY" -c "import json; print(json.load(open('$R/json/validation_eta_selection.json'))['selected_eta'])")
export PYTHONPATH="$O:$BASE" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "[frozen] eta=$ETA; model=$m seed=$s; no neural inference"
srun "$PY" -u "$O/nr1_metric_complete_rescore.py" --model "$m" --seed "$s" --parquet "$DATA" --eta "$ETA" --workers 16 --batch 16 --chunk 16 \
  --cache-dir "$R/cache/${m}_s${s}_test" --out-dir "$R/complete/${m}_s${s}"
