#!/usr/bin/env python3
"""Aggregate the frozen three-seed GraphKit intervention rescores."""
import argparse, json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--indir', required=True); ap.add_argument('--out', required=True); a=ap.parse_args()
    paths=sorted(Path(a.indir).glob('graphkit_both_s*_output_interventions_k16.json'))
    if len(paths)!=3: raise RuntimeError(f'Need exactly three completed seed files, got {len(paths)}')
    vals={}
    for p in paths:
        x=json.loads(p.read_text())['result']['metrics']['restore_known']
        for condition,m in x.items():
            vals.setdefault(condition,{}).setdefault('seeds',[]).append({'file':p.name,**m})
    out={'n_seeds':3,'conditions':{}}
    keys=('vmag_rmse','angle_rmse_deg','within_r2','within_slope','mean_pb','max_pb')
    for cond,d in vals.items():
        out['conditions'][cond]={'per_seed':d['seeds'], 'mean_std':{k:{'mean':float(np.mean([s[k] for s in d['seeds']])), 'std':float(np.std([s[k] for s in d['seeds']],ddof=1))} for k in keys}}
    Path(a.out).write_text(json.dumps(out,indent=2))
    for cond,d in out['conditions'].items():
        print(cond, ' | '.join(f'{k}={v["mean"]:.7g} ± {v["std"]:.4g}' for k,v in d['mean_std'].items()))
if __name__=='__main__': main()
