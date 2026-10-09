#!/usr/bin/env python3
"""Cached-prediction, multiprocessing one-step custom-NR baseline for GBnetwork.

The neural model is evaluated once per split.  The existing NumPy custom NR
then runs in forked CPU workers, mirroring ScenarioSynthesis_PPC's data
generation pattern.  This is a scorer; it never trains or generates labels.
"""
from __future__ import annotations
import argparse, csv, json, math, multiprocessing as mp, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from collate_blockdiag_optimized_complex_columns import collate_blockdiag
from controlled_error_geometry import calibrated_forward, make_model, split_dataset
from diagnose_residual_distributions import angle_diff, fit_per_bus_offset, manifold_basis
from manifold_projection import project_state
from output_intervention_metrics import _restore_known_setpoints

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("SCENARIO_SYNTHESIS_ROOT", str(ROOT.parent / "ScenarioSynthesis_PPC")))
from newton_raphson_improved import JacobianMatrix3p, VoltageCalculation3p

ETAS=(.25,.5,.75,1.)
G={}

def _pb(U,S,Y,bt):
    sc=U*np.conj(Y@U); dp=S.real-sc.real; dq=S.imag-sc.imag
    p=bt!=1; q=(bt!=1)&(bt!=2)
    rho=np.sqrt((dp*p)**2+(dq*q)**2)
    return float(rho.mean()),float(rho.max())

def _init(cache, y, bt):
    global G
    G={k:np.load(v,mmap_mode="r") for k,v in cache.items()}
    G["Y"]=np.load(y); G["bt"]=np.load(bt)

def _worker(bounds):
    a,b,etas=bounds; Y=G["Y"]; bt=G["bt"]; out={}
    for name in ("C","CSP16"):
        vals={str(e):[] for e in etas}; failures=0; elapsed=0.
        for i in range(a,b):
            vth=G[name][i]; U=np.asarray(vth[:,0])*np.exp(1j*np.asarray(vth[:,1])); S=np.asarray(G["S"][i])
            try:
                tic=time.perf_counter(); J=JacobianMatrix3p(Y,U)[0]
                U1,_=VoltageCalculation3p(bt,J,Y,U,S.real,S.imag); elapsed+=time.perf_counter()-tic
            except np.linalg.LinAlgError:
                failures+=1; U1=U
            ang0=np.angle(U); ang1=np.angle(U1); mag0=np.abs(U); mag1=np.abs(U1)
            slack=bt==1; pq=(bt!=1)&(bt!=2); non=~slack
            for eta in etas:
                ang=ang0.copy(); mag=mag0.copy(); ang[non]+=eta*np.angle(np.exp(1j*(ang1[non]-ang0[non])))
                mag[pq]+=eta*(mag1[pq]-mag0[pq]); Un=mag*np.exp(1j*ang)
                vals[str(eta)].append(_pb(Un,S,Y,bt))
        out[name]={"pb":vals,"failures":failures,"solve_seconds":elapsed}
    return out

def _cache(model_kind, ckpt, phase, parquet, batch, cache_dir):
    device=torch.device("cuda"); train,valid,test=split_dataset(parquet); ev=valid if phase=="validation" else test
    model,forward=make_model(model_kind,ckpt,parquet,batch,device); off=fit_per_bus_offset(model,forward,train,batch,device)
    Uv,Ut,vbar,tbar,*_=manifold_basis(train,batch,k=16); basis=(Uv.to(device=device,dtype=torch.float64),Ut.to(device=device,dtype=torch.float64),vbar.to(device=device,dtype=torch.float64),tbar.to(device=device,dtype=torch.float64))
    n=len(ev); loader=DataLoader(ev,batch_size=batch,shuffle=False,num_workers=0,collate_fn=collate_blockdiag)
    cache_dir.mkdir(parents=True,exist_ok=True); c=np.lib.format.open_memmap(cache_dir/'C.npy',mode='w+',dtype='float64',shape=(n,2224,2)); p=np.lib.format.open_memmap(cache_dir/'CSP16.npy',mode='w+',dtype='float64',shape=(n,2224,2)); s=np.lib.format.open_memmap(cache_dir/'S.npy',mode='w+',dtype='complex128',shape=(n,2224))
    k=0; csp_time=0.; y=bt=None
    with torch.no_grad():
      for b in loader:
        ns=len(b['sizes']); nbus=int(b['sizes'][0]); assert nbus==2224
        cal=calibrated_forward(forward,off,b)[0].reshape(ns,nbus,2)
        torch.cuda.synchronize(); tic=time.perf_counter(); pv,pt=project_state(cal[...,0],cal[...,1],basis); pv,pt=_restore_known_setpoints(pv,pt,b,nbus); torch.cuda.synchronize(); csp_time+=time.perf_counter()-tic
        c[k:k+ns]=cal.cpu().numpy(); p[k:k+ns]=torch.stack((pv,pt),-1).cpu().numpy(); s[k:k+ns]=b['S_start'][0].reshape(ns,nbus).cpu().numpy(); k+=ns
        if y is None:
          # ``collate_blockdiag`` stores a batch as a sparse block-diagonal Ybus.
          # Materialising the entire B*2224 square matrix is prohibitively large;
          # the grid is shared, so extract only its first block instead.
          ybus = b['Ybus']
          if ybus.is_sparse:
            co = ybus.coalesce(); ind, val = co.indices(), co.values()
            keep = (ind[0] < nbus) & (ind[1] < nbus)
            yy = torch.zeros((nbus, nbus), dtype=val.dtype, device=val.device)
            yy[ind[0, keep], ind[1, keep]] = val[keep]
          else:
            yy = ybus[:nbus, :nbus]
          y=yy.cpu().numpy().astype(np.complex128); rawbt=b['bus_type'][0].reshape(-1)[:nbus].cpu().numpy(); bt=np.where(rawbt==1,1,np.where(rawbt==2,2,3)).astype(np.int64)
    np.save(cache_dir/'Y.npy',y); np.save(cache_dir/'bt.npy',bt); return {"n":n,"csp_seconds":csp_time,"cache":{x:str(cache_dir/f'{x}.npy') for x in ('C','CSP16','S')}}

def _score(cache_meta, etas, workers):
    cache=cache_meta['cache']; y=str(Path(cache['C']).with_name('Y.npy')); bt=str(Path(cache['C']).with_name('bt.npy')); n=cache_meta['n']; chunks=[(i,min(i+16,n),etas) for i in range(0,n,16)]
    ctx=mp.get_context('fork'); tic=time.perf_counter()
    with ctx.Pool(workers,initializer=_init,initargs=(cache,y,bt),maxtasksperchild=100) as pool: rows=list(pool.imap_unordered(_worker,chunks,chunksize=1))
    out={"wall_seconds":time.perf_counter()-tic}
    for state in ('C','CSP16'):
      d={str(e):[] for e in etas}; maxs={str(e):[] for e in etas}; fails=solvet=0.
      for r in rows:
       fails+=r[state]['failures']; solvet+=r[state]['solve_seconds']
       for e,vals in r[state]['pb'].items(): d[e].extend(x[0] for x in vals); maxs[e].extend(x[1] for x in vals)
      out[state]={e:{"mean_pb":float(np.mean(v)),"max_pb":float(np.max(maxs[e])),"scenario_pb":v} for e,v in d.items()}; out[state]['failures']=int(fails); out[state]['sum_solve_seconds']=solvet
    return out

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=('validation','test'),required=True); ap.add_argument('--model',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--parquet',required=True); ap.add_argument('--cache-dir',required=True); ap.add_argument('--out',required=True); ap.add_argument('--etas',nargs='+',type=float,default=list(ETAS)); ap.add_argument('--workers',type=int,default=16); ap.add_argument('--batch',type=int,default=16); args=ap.parse_args()
 os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
 cache=_cache(args.model,args.checkpoint,args.stage,args.parquet,args.batch,Path(args.cache_dir)); score=_score(cache,tuple(args.etas),args.workers)
 Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps({"stage":args.stage,"model":args.model,"checkpoint":args.checkpoint,"etas":args.etas,"cache":cache,"score":score},indent=2))
if __name__=='__main__': main()
