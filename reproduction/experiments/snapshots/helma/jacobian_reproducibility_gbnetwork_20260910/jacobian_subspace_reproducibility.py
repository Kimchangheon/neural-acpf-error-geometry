#!/usr/bin/env python3
"""Reproduce the frozen random-JVP experiment and add audited model-error JVPs.

This deliberately reuses ``jacobian_subspace_alignment``'s reduced state,
analytic complex128 JVP, split, seed, and restricted train-SVD basis.  It is a
post-hoc scorer only: no model or data is changed.
"""
from __future__ import annotations

import argparse, csv, json, math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from collate_blockdiag_optimized_complex_columns import collate_blockdiag
from controlled_error_geometry import calibrated_forward, make_model, split_dataset
from diagnose_residual_distributions import angle_diff, fit_per_bus_offset, manifold_basis
from jacobian_subspace_alignment import _orthonormal_restricted_basis, reduced_jvp, summarize


def _reduced_residual(y, v, theta, sset, angle_unknown, mag_unknown):
    vc = v * torch.exp(1j * theta)
    ib = torch.sparse.mm(y.coalesce(), vc[:, None])[:, 0] if y.is_sparse else y @ vc
    dr = sset - vc * ib.conj()
    return torch.cat((dr.real[angle_unknown], dr.imag[mag_unknown]))


def _gain(y, v, theta, d, angle_unknown, mag_unknown, tol):
    denom = torch.linalg.vector_norm(d)
    if float(denom) <= tol:
        return None
    return float(torch.linalg.vector_norm(
        reduced_jvp(y, v, theta, d[None], angle_unknown, mag_unknown)[0]) / denom)


def _quantiles(values):
    if not values:
        return {"n": 0}
    x = np.asarray(values, dtype=np.float64)
    return {"n": int(x.size), "p05": float(np.quantile(x,.05)),
            "p25": float(np.quantile(x,.25)), "median": float(np.median(x)),
            "p75": float(np.quantile(x,.75)), "p95": float(np.quantile(x,.95)),
            "mean": float(x.mean()), "std": float(x.std(ddof=1))}


def _model_record():
    return {"parallel": [], "perp": [], "linear_norm": [], "nonlinear_norm": [],
            "excluded_parallel": 0, "excluded_perp": 0, "excluded_linearization": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True); ap.add_argument("--g3", required=True)
    ap.add_argument("--gridsfm", required=True); ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", required=True); ap.add_argument("--out-npz", required=True)
    ap.add_argument("--rank", type=int, default=16); ap.add_argument("--scenarios", type=int, default=128)
    ap.add_argument("--directions", type=int, default=32); ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--basis-batch", type=int, default=16); args=ap.parse_args()
    if args.rank != 16 or args.scenarios != 128 or args.directions != 32 or args.seed != 20260909:
        raise ValueError("Reproduction must use the frozen k=16, 128x32, seed-20260909 protocol.")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, _valid, test = split_dataset(args.parquet)
    uv, ut, _vbar, _tbar, ev, et = manifold_basis(train, args.basis_batch, k=args.rank)
    gen=torch.Generator().manual_seed(args.seed)
    local=torch.randperm(len(test), generator=gen)[:args.scenarios].tolist()
    selected=Subset(test,local)
    loader=DataLoader(selected,batch_size=1,shuffle=False,num_workers=0,collate_fn=collate_blockdiag)
    # Calibration is fitted once, from training predictions only, for actual-error diagnostics.
    models={}
    for label, kind, ckpt in (("PIGNN-GC","g3",args.g3),("GridSFM","gridsfm",args.gridsfm)):
        model, forward=make_model(kind,ckpt,args.parquet,args.basis_batch,device)
        models[label]=(forward,fit_per_bus_offset(model,forward,train,args.basis_batch,device))
    rng=torch.Generator(device="cpu").manual_seed(args.seed+1)
    random={k:[] for k in ("gain_parallel","gain_perp","raw_jp","raw_jperp","norm_parallel","norm_perp","ratio")}
    raw_scenario=[]; raw_direction=[]; raw_gp=[]; raw_go=[]
    actual={label:_model_record() for label in models}; selected_indices=[]; ranks=None
    tol=float(torch.finfo(torch.float64).eps)
    with torch.no_grad():
      for si,batch in enumerate(loader):
        nbus=int(batch["sizes"][0]); bt=batch["bus_type"][0].reshape(-1)[:nbus].to(torch.long)
        au_cpu=bt!=1; mu_cpu=(bt!=1)&(bt!=2); au=au_cpu.to(device); mu=mu_cpu.to(device)
        if ranks is None:
            qth,rt=_orthonormal_restricted_basis(ut,au_cpu,args.rank,device)
            qvm,rv=_orthonormal_restricted_basis(uv,mu_cpu,args.rank,device)
            ranks={"angle":rt,"magnitude":rv,"combined":rt+rv}
        nt=int(au.sum()); nstate=int(au.sum()+mu.sum())
        ref=batch["V_newton"][0,:nbus].to(device=device,dtype=torch.float64); rv0,rt0=ref[:,0],ref[:,1]
        y=batch["Ybus"].to(device=device,dtype=torch.complex128); sset=batch["S_start"][0,:nbus].to(device=device,dtype=torch.complex128)
        q=torch.randn(args.directions,nstate,generator=rng,dtype=torch.float64).to(device); q/=q.norm(dim=1,keepdim=True)
        par=torch.cat(((q[:,:nt]@qth)@qth.T,(q[:,nt:]@qvm)@qvm.T),1); perp=q-par
        npn=par.norm(dim=1); non=perp.norm(dim=1)
        if bool((npn<=tol).any() or (non<=tol).any()): raise RuntimeError("Frozen random direction produced zero projected norm")
        jp=reduced_jvp(y,rv0,rt0,par,au,mu).norm(dim=1); jo=reduced_jvp(y,rv0,rt0,perp,au,mu).norm(dim=1)
        gp=reduced_jvp(y,rv0,rt0,par/npn[:,None],au,mu).norm(dim=1); go=reduced_jvp(y,rv0,rt0,perp/non[:,None],au,mu).norm(dim=1)
        for key,val in (("gain_parallel",gp),("gain_perp",go),("raw_jp",jp),("raw_jperp",jo),("norm_parallel",npn),("norm_perp",non),("ratio",go/gp)):
            random[key].extend(val.cpu().tolist())
        raw_scenario.extend([si]*args.directions); raw_direction.extend(range(args.directions)); raw_gp.extend(gp.cpu().tolist()); raw_go.extend(go.cpu().tolist())
        for label,(forward,offset) in models.items():
            cal=calibrated_forward(forward,offset,batch)[0,:nbus].to(torch.float64)
            # Exact reduced-coordinate calibrated error: known PF coordinates are not J state variables.
            e=torch.cat((angle_diff(cal[:,1],rt0)[au],(cal[:,0]-rv0)[mu]))
            ep=torch.cat((((e[:nt]@qth)@qth.T),((e[nt:]@qvm)@qvm.T)))
            eo=e-ep; rec=actual[label]
            g=_gain(y,rv0,rt0,ep,au,mu,tol)
            if g is None: rec["excluded_parallel"]+=1
            else: rec["parallel"].append(g)
            g=_gain(y,rv0,rt0,eo,au,mu,tol)
            if g is None: rec["excluded_perp"]+=1
            else: rec["perp"].append(g)
            # First-order comparison on the same reduced perturbation only.
            jfull=reduced_jvp(y,rv0,rt0,e[None],au,mu)[0].norm()
            vpert=rv0.clone(); tpert=rt0.clone(); vpert[mu]=vpert[mu]+e[nt:]; tpert[au]=tpert[au]+e[:nt]
            nonlinear=(_reduced_residual(y,vpert,tpert,sset,au,mu)-_reduced_residual(y,rv0,rt0,sset,au,mu)).norm()
            if float(nonlinear)<=tol: rec["excluded_linearization"]+=1
            else: rec["linear_norm"].append(float(jfull)); rec["nonlinear_norm"].append(float(nonlinear))
        selected_indices.append(int(test.indices[local[si]]))
        if (si+1)%16==0: print(f"[reproduce] {si+1}/{args.scenarios}",flush=True)
    summary={k:summarize(v) for k,v in random.items() if k!="ratio"}
    actual_summary={}
    for label,rec in actual.items():
        ln=np.asarray(rec["linear_norm"]); nl=np.asarray(rec["nonlinear_norm"])
        actual_summary[label]={"gain_parallel":_quantiles(rec["parallel"]),"gain_perp":_quantiles(rec["perp"]),
          "ratio_of_medians":float(np.median(rec["perp"])/np.median(rec["parallel"])),
          "excluded_parallel":rec["excluded_parallel"],"excluded_perp":rec["excluded_perp"],
          "linearization":{"n":int(len(ln)),"excluded":rec["excluded_linearization"],"ratio":_quantiles((ln/nl).tolist()),
            "correlation":float(np.corrcoef(ln,nl)[0,1]) if len(ln)>1 else float("nan")}}
    payload={"protocol":{"source":"jacobian_subspace_alignment.py frozen protocol", "rank":16,"scenarios":128,"directions_per_scenario":32,"seed":20260909,
      "state":"[theta(non-slack, radians), |V|(PQ)]","residual":"[DeltaP(non-slack), DeltaQ(PQ)]","jacobian":"analytic complex128 JVP", "random_direction":"iid N(0,I), unit-normalized before P; Pq and (I-P)q unit-normalized for gain", "basis":"separate train-only magnitude/angle SVD, restricted to unknown coordinates then QR orthonormalized", "gauge":"same circular wrapped angle convention as manifold basis; no degrees inside J", "tolerance":tol},
      "dimensions":{"buses":nbus,"state":nstate,"subspace_ranks":ranks},"basis_explained_variance":{"magnitude":ev,"angle":et},"selected_test_indices":selected_indices,
      "random_direction":{"summary":summary,"R1_ratio_of_pooled_medians":float(np.median(random["gain_perp"])/np.median(random["gain_parallel"])),"R2_median_paired_ratio":float(np.median(random["ratio"]))},"actual_calibrated_error":actual_summary}
    Path(args.out_json).write_text(json.dumps(payload,indent=2,allow_nan=False))
    np.savez_compressed(args.out_npz,scenario=np.asarray(raw_scenario),direction=np.asarray(raw_direction),gain_parallel=np.asarray(raw_gp),gain_perp=np.asarray(raw_go),ratio=np.asarray(random["ratio"]))
    with Path(args.out_csv).open("w",newline="") as f:
      w=csv.writer(f); w.writerow(["section","model","statistic","value"])
      for key,s in summary.items():
       for stat,value in s.items(): w.writerow(["random","",f"{key}.{stat}",value])
      for key in ("R1_ratio_of_pooled_medians","R2_median_paired_ratio"): w.writerow(["random","",key,payload["random_direction"][key]])
      for model,s in actual_summary.items():
       for block in ("gain_parallel","gain_perp"):
        for stat,value in s[block].items(): w.writerow(["actual",model,f"{block}.{stat}",value])
       w.writerow(["actual",model,"ratio_of_medians",s["ratio_of_medians"]])
    print(json.dumps(payload,indent=2),flush=True)

if __name__ == "__main__": main()
