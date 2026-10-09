#!/bin/bash -l
#SBATCH --job-name=gate_hetero_gb
#SBATCH --output=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/gate_hetero_gb_%j.out
#SBATCH --error=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/sbatch/Job_out/gate_hetero_gb_%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
set -euo pipefail
BASE=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
GROUP=${GROUP:?set GROUP}
python - "$BASE/results/logs/$GROUP" <<'PY'
import json, math, re, sys
from pathlib import Path
root = Path(sys.argv[1])
modes = ("baseline", "hetero")
def parse(mode):
    path = root / f"GBnetwork_{mode}_s42_training_log.txt"
    text = path.read_text()
    tail = text[text.rfind("Residual distribution over"):]
    def pair(label):
        m = re.search(rf"{label}\s+\|ΔP\|\s+([0-9.eE+-]+).*?\|ΔQ\|\s+([0-9.eE+-]+)", tail)
        if not m: raise SystemExit(f"cannot parse {label} from {path}")
        return tuple(map(float, m.groups()))
    maximum, mean, p95 = pair("max"), pair("mean"), pair("p95")
    vals = (*maximum, *mean, *p95)
    score = math.exp(sum(math.log(max(v, 1e-30)) for v in vals) / len(vals))
    return dict(path=str(path), max=maximum, mean=mean, p95=p95, score=score)
r = {m: parse(m) for m in modes}
b = r["baseline"]
passing = []
for mode in modes[1:]:
    x = r[mode]
    # Residual-primary gate: at least 5% better aggregate residual score, with
    # neither worst P nor worst Q allowed to deteriorate by more than 25%.
    if x["score"] <= 0.95*b["score"] and all(x["max"][i] <= 1.25*b["max"][i] for i in (0,1)):
        passing.append(mode)
print(json.dumps({"results": r, "passing": passing}, indent=2))
if not passing:
    raise SystemExit("GBnetwork residual gate failed; Heo1 will remain blocked")
PY
