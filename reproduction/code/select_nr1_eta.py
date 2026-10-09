#!/usr/bin/env python3
"""Freeze the global damping factor from validation-only NR1 JSON files."""
import argparse,json,glob
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--glob',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
files=sorted(glob.glob(a.glob)); assert len(files)==6, f'expected six validation files, got {len(files)}'
rows=[json.load(open(f)) for f in files]; etas=rows[0]['etas']; means={}
for e in etas:
 vals=[r['score'][state][str(e)]['mean_pb'] for r in rows for state in ('C','CSP16')]
 means[str(e)]=sum(vals)/len(vals)
best=min(etas,key=lambda x:means[str(x)])
Path(a.out).write_text(json.dumps({'rule':'minimize unweighted mean validation Mean PB over C+NR1 and CSP16+NR1 across two models x three seeds','candidates':means,'selected_eta':best,'inputs':files},indent=2))
