#!/usr/bin/env python3
"""Reference diagnostics and reconstruction-matched CSP controls on GBnetwork."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from collate_blockdiag_optimized_complex_columns import collate_blockdiag
from compare_solution_basis_controls import (_lowpass, _random_basis, _restore_known,
                                             _topology_operators)
from controlled_error_geometry import calibrated_forward, make_model, pb_per_scenario, split_dataset
from diagnose_residual_distributions import angle_diff, fit_per_bus_offset, manifold_basis
from manifold_projection import project_state


def record_new():
    return {"v_sse": 0., "t_sse": 0., "n": 0, "pb": [], "max_pb": 0.}


def record_add(r, v, t, ref, batch, device):
    pbs, mx = pb_per_scenario(v, t, batch, device, return_global_max=True)
    r["v_sse"] += float((v-ref[..., 0]).square().sum())
    r["t_sse"] += float(angle_diff(t, ref[..., 1]).square().sum())
    r["n"] += v.numel(); r["pb"].append(pbs); r["max_pb"] = max(r["max_pb"], mx)


def record_end(r):
    return {"vmag_rmse": math.sqrt(r["v_sse"]/r["n"]),
            "angle_rmse_deg": math.degrees(math.sqrt(r["t_sse"]/r["n"])),
            "mean_pb": float(np.concatenate(r["pb"]).mean()), "max_pb": r["max_pb"]}


def project_blocks(v, t, uv, ut, vbar, tbar):
    pv = vbar + ((v-vbar) @ uv) @ uv.T
    dt = angle_diff(t, tbar)
    pt = torch.atan2(torch.sin(tbar + (dt @ ut) @ ut.T),
                     torch.cos(tbar + (dt @ ut) @ ut.T))
    return pv, pt


def retained(d, p):
    return 1. - float((d-p).square().sum()) / float(d.square().sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True); ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--random-draws", type=int, default=8)
    ap.add_argument("--random-seed", type=int, default=20260909)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.rank != 16: raise ValueError("Solution-SVD comparison is frozen at k=16")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, _valid, test = split_dataset(args.parquet)
    model, forward = make_model("g3", args.checkpoint, args.parquet, args.batch, device)
    offset = fit_per_bus_offset(model, forward, train, args.batch, device)
    usv, ust, vbar, tbar, target_v, target_t = manifold_basis(train, args.batch, k=16)
    vbar=vbar.to(device=device,dtype=torch.float64); tbar=tbar.to(device=device,dtype=torch.float64)
    usv=usv.to(device=device,dtype=torch.float64); ust=ust.to(device=device,dtype=torch.float64)
    nbus=vbar.numel()
    lap, h, eigvals = _topology_operators(train, nbus, device)
    randoms=[(_random_basis(nbus,16,args.random_seed+2*j,device),
              _random_basis(nbus,16,args.random_seed+2*j+1,device)) for j in range(args.random_draws)]

    totals={"v":0.,"t":0.}; lap_energy={"v":torch.zeros(nbus,dtype=torch.float64,device=device),
                                        "t":torch.zeros(nbus,dtype=torch.float64,device=device)}
    h_off={"v":0.,"t":0.}; random_off=[{"v":0.,"t":0.} for _ in randoms]
    loader=DataLoader(train,batch_size=args.batch,shuffle=False,num_workers=0,collate_fn=collate_blockdiag)
    with torch.no_grad():
        for bi,b in enumerate(loader):
            sizes=b["sizes"].numpy().astype(int); ns=len(sizes)
            ref=b["V_newton"].to(device=device,dtype=torch.float64)[0].reshape(ns,nbus,2)
            dv=ref[...,0]-vbar; dt=angle_diff(ref[...,1],tbar)
            totals["v"]+=float(dv.square().sum()); totals["t"]+=float(dt.square().sum())
            cv=dv@lap; ct=dt@lap
            lap_energy["v"]+=cv.square().sum(0); lap_energy["t"]+=ct.square().sum(0)
            h_off["v"]+=float((dv-dv@h.T).square().sum()); h_off["t"]+=float((dt-dt@h.T).square().sum())
            for rr,(uv,ut) in zip(random_off,randoms):
                rr["v"]+=float((dv-(dv@uv)@uv.T).square().sum())
                rr["t"]+=float((dt-(dt@ut)@ut.T).square().sum())
            if bi and bi%100==0: print(f"[training geometry] {bi}/{len(loader)}",flush=True)

    def choose_k(block,target):
        ratio=torch.cumsum(lap_energy[block],0)/totals[block]
        hit=torch.nonzero(ratio>=target)
        if not len(hit): return nbus
        return int(hit[0])+1
    kv=choose_k("v",target_v); kt=choose_k("t",target_t)
    beta_v=min(1.,math.sqrt(max(0.,(1.-target_v)*totals["v"]/h_off["v"])))
    beta_t=min(1.,math.sqrt(max(0.,(1.-target_t)*totals["t"]/h_off["t"])))
    h_v=(1-beta_v)*torch.eye(nbus,dtype=torch.float64,device=device)+beta_v*h
    h_t=(1-beta_t)*torch.eye(nbus,dtype=torch.float64,device=device)+beta_t*h
    methods={
      "solution_svd_k16":("basis",usv,ust),
      "laplacian_k16":("basis",lap[:,:16],lap[:,:16]),
      "laplacian_reconstruction_matched":("basis",lap[:,:kv],lap[:,:kt]),
      "spatial_onehop":("filter",h,h),
      "spatial_reconstruction_matched":("filter",h_v,h_t),
    }
    for j,(uv,ut) in enumerate(randoms): methods[f"random_{j:02d}_k16"]=("basis",uv,ut)
    test_energy={m:{"v_total":0.,"t_total":0.,"v_off":0.,"t_off":0.} for m in methods}
    refs={m:{"pure":record_new(),"restore_known":record_new()} for m in methods}
    preds={m:record_new() for m in methods}
    loader=DataLoader(test,batch_size=args.batch,shuffle=False,num_workers=0,collate_fn=collate_blockdiag)
    with torch.no_grad():
      for bi,b in enumerate(loader):
        sizes=b["sizes"].numpy().astype(int); ns=len(sizes)
        ref=b["V_newton"].to(device=device,dtype=torch.float64)[0].reshape(ns,nbus,2)
        cal=calibrated_forward(forward,offset,b)[0].reshape(ns,nbus,2)
        dv=ref[...,0]-vbar; dt=angle_diff(ref[...,1],tbar)
        for name,(kind,ov,ot) in methods.items():
          if kind=="basis":
            rv,rt=project_blocks(ref[...,0],ref[...,1],ov,ot,vbar,tbar)
            pv,pt=project_blocks(cal[...,0],cal[...,1],ov,ot,vbar,tbar)
          else:
            rv=vbar+dv@ov.T; rt=torch.atan2(torch.sin(tbar+dt@ot.T),torch.cos(tbar+dt@ot.T))
            cdv=cal[...,0]-vbar; cdt=angle_diff(cal[...,1],tbar)
            pv=vbar+cdv@ov.T; pt=torch.atan2(torch.sin(tbar+cdt@ot.T),torch.cos(tbar+cdt@ot.T))
          q=test_energy[name]; q["v_total"]+=float(dv.square().sum()); q["t_total"]+=float(dt.square().sum())
          q["v_off"]+=float((rv-ref[...,0]).square().sum()); q["t_off"]+=float(angle_diff(rt,ref[...,1]).square().sum())
          record_add(refs[name]["pure"],rv,rt,ref,b,device)
          rvr,rtr=_restore_known(rv,rt,b,nbus); record_add(refs[name]["restore_known"],rvr,rtr,ref,b,device)
          pv,pt=_restore_known(pv,pt,b,nbus); record_add(preds[name],pv,pt,ref,b,device)
        if bi and bi%50==0: print(f"[test] {bi}/{len(loader)}",flush=True)

    output={
      "protocol":{
        "model":"PIGNN-GC/G3 seed42","checkpoint":args.checkpoint,"split":"seed42 paper split",
        "calibration":"train predictions only","solution_basis":"train NR only; k=16",
        "matching_rule":"blockwise: smallest Laplacian low-frequency rank reaching the corresponding SVD-16 training EVR; spatial beta in H_beta=(1-beta)I+beta H chosen analytically from training reconstruction energy",
        "selection_uses_validation_or_test":False,"prediction_pb":"restore known setpoints then complex128 structural-zero Mean PB",
        "reference_pb":"both pure operator and operational restore-known values reported"},
      "targets":{"svd16_training_evr_magnitude":target_v,"svd16_training_evr_angle":target_t},
      "selected":{"laplacian_rank_magnitude":kv,"laplacian_rank_angle":kt,
                  "spatial_beta_magnitude":beta_v,"spatial_beta_angle":beta_t},
      "training_retained":{
        "laplacian_matched":{"magnitude":float(lap_energy["v"][:kv].sum()/totals["v"]),"angle":float(lap_energy["t"][:kt].sum()/totals["t"])},
        "spatial_matched":{"magnitude":1-beta_v**2*h_off["v"]/totals["v"],"angle":1-beta_t**2*h_off["t"]/totals["t"]},
        "random_k16":[{"magnitude":1-r["v"]/totals["v"],"angle":1-r["t"]/totals["t"]} for r in random_off]},
      "methods":{m:{
        "test_reference_retained":{"magnitude":1-q["v_off"]/q["v_total"],"angle":1-q["t_off"]/q["t_total"]},
        "reference_pure":record_end(refs[m]["pure"]),"reference_restore_known":record_end(refs[m]["restore_known"]),
        "calibrated_prediction_intervention":record_end(preds[m])}
        for m,q in test_energy.items()},
      "laplacian_eigenvalues":{"first16":eigvals[:16].cpu().tolist(),"at_kv":float(eigvals[kv-1]),"at_kt":float(eigvals[kt-1])},
      "n_test":len(test)}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(output,indent=2,allow_nan=False))
    print(json.dumps(output,indent=2),flush=True)

if __name__=="__main__": main()
