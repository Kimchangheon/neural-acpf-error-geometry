#!/bin/bash -l
# One-seed LUMINA validation eta freeze.
#SBATCH --job-name=lumina_nr1_eta
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr1_lumina_gbv06z_20260913/logs/eta_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/nr1_lumina_gbv06z_20260913/logs/eta_%j.err
set -euo pipefail
BASE=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC; O=$BASE/overlays/nr1_lumina_gbv06z_20260913; R=$BASE/results/nr1_lumina_gbv06z_20260913; PY=/home/hpc/iwi5/iwi5295h/conda-envs/gridfm-py312/bin/python
"$PY" -c 'import json,glob; from pathlib import Path; f=glob.glob("'"$R"'/json/*_validation.json"); assert len(f)==1; x=json.load(open(f[0])); et=x["etas"]; m={str(e):sum(x["score"][s][str(e)]["mean_pb"] for s in ("C","CSP16"))/2 for e in et}; b=min(et,key=lambda e:m[str(e)]); Path("'"$R"'/json/validation_eta_selection.json").write_text(json.dumps({"rule":"minimum validation Mean PB over C+NR1 and CSP16+NR1; one LUMINA seed","candidates":m,"selected_eta":b,"inputs":f},indent=2))'
