#!/usr/bin/env python3
"""GBcorr calibration x basis CSP audit, using the paper implementation."""
import argparse,csv,json,math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from collate_blockdiag_optimized_complex_columns import collate_blockdiag
from controlled_error_geometry import split_dataset,make_model,calibrated_forward,pb_per_scenario
from diagnose_residual_distributions import manifold_basis,fit_per_bus_offset,angle_diff
from manifold_projection import project_state
from output_intervention_metrics import _restore_known_setpoints

def en(x): return float(x.square().sum())
def off_state(x,mean,U,circ=False):
 """Off-subspace component of a state, centred at its fitting mean."""
 d=angle_diff(x,mean) if circ else x-mean; return d-(d@U)@U.T
def off_error(e,U):
 """Off-subspace component of an already-centred prediction error.

 Error is a tangent-space difference, not a state.  Subtracting the training
 mean here would be dimensionally wrong and breaks the affine-projection
 trade-off identity.
 """
 return e-(e@U)@U.T
def init(): return dict(pb=[],max_pb=0.,vs=0.,ts=0.,n=0,sp=None,sr=None,sp2=None,sr2=None,spr=None,ns=0,ecv=0.,ect=0.,cspv=0.,cspt=0.,offv=0.,offt=0.)
def add(r,v,t,rv,rt,b,dev):
 p,m=pb_per_scenario(v,t,b,dev,return_global_max=True); r['pb'].append(p);r['max_pb']=max(r['max_pb'],m);r['vs']+=en(v-rv);r['ts']+=en(angle_diff(t,rt));r['n']+=v.numel()
 for k,x in [('sp',v.sum(0)),('sr',rv.sum(0)),('sp2',v.square().sum(0)),('sr2',rv.square().sum(0)),('spr',(v*rv).sum(0))]: r[k]=x if r[k] is None else r[k]+x
 r['ns']+=v.shape[0]
def finish(r):
 pb=torch.cat([torch.from_numpy(x) for x in r['pb']]).numpy();n=r['ns'];vp=(r['sp2']-r['sp'].square()/n).sum().item();vr=(r['sr2']-r['sr'].square()/n).sum().item();co=(r['spr']-r['sp']*r['sr']/n).sum().item()
 return dict(vmag_rmse=math.sqrt(r['vs']/r['n']),angle_rmse_deg=math.degrees(math.sqrt(r['ts']/r['n'])),within_r2=co*co/(vp*vr) if vp>0 and vr>0 else 0.,within_slope=co/vr if vr>0 else float('nan'),mean_pb=float(pb.mean()),max_pb=r['max_pb'],cal_error_energy=dict(magnitude=r['ecv'],angle=r['ect']),csp_error_energy=dict(magnitude=r['cspv'],angle=r['cspt']),error_off_energy=dict(magnitude=r['offv'],angle=r['offt']))
def main():
 a=argparse.ArgumentParser();a.add_argument('--control',required=True);a.add_argument('--gbcorr',required=True);a.add_argument('--checkpoint',required=True);a.add_argument('--out-json',required=True);a.add_argument('--out-csv',required=True);a.add_argument('--batch',type=int,default=16);a.add_argument('--rank',type=int,default=16);x=a.parse_args();dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 ct,_,ce=split_dataset(x.control);gt,_,ge=split_dataset(x.gbcorr)
 cb=manifold_basis(ct,x.batch,k=x.rank);gb=manifold_basis(gt,x.batch,k=x.rank)
 cb=tuple(z.to(dev,dtype=torch.float64) if hasattr(z,'to') else z for z in cb[:4]);gb=tuple(z.to(dev,dtype=torch.float64) if hasattr(z,'to') else z for z in gb[:4])
 model,fw=make_model('g3',x.checkpoint,x.gbcorr,x.batch,dev);co=fit_per_bus_offset(model,fw,ct,x.batch,dev);go=fit_per_bus_offset(model,fw,gt,x.batch,dev)
 names={('control','control'):init(),('gbcorr','control'):init(),('control','gbcorr'):init(),('gbcorr','gbcorr'):init()}; ref={"control":dict(tv=0.,tt=0.,ov=0.,ot=0.,n=0.,pv=0.,pt=0.,pb=[],mx=0.),"gbcorr":dict(tv=0.,tt=0.,ov=0.,ot=0.,n=0.,pv=0.,pt=0.,pb=[],mx=0.)}; loader=DataLoader(ge,batch_size=x.batch,shuffle=False,num_workers=0,collate_fn=collate_blockdiag)
 with torch.no_grad():
  for b in loader:
   ns=len(b['sizes']);N=int(b['sizes'][0]); z=b['V_newton'][0].to(dev,dtype=torch.float64).reshape(ns,N,2);rv,rt=z[...,0],z[...,1]
   for bn,ba in [('control',cb),('gbcorr',gb)]:
    Uv,Ut,vbar,tbar=ba;ov=off_state(rv,vbar,Uv);ot=off_state(rt,tbar,Ut,True);q=ref[bn];q['tv']+=en(rv-vbar);q['tt']+=en(angle_diff(rt,tbar));q['ov']+=en(ov);q['ot']+=en(ot);q['n']+=rv.numel();pv,pt=project_state(rv,rt,ba);q['pv']+=en(pv-rv);q['pt']+=en(angle_diff(pt,rt));pp,mm=pb_per_scenario(pv,pt,b,dev,return_global_max=True);q['pb'].append(pp);q['mx']=max(q['mx'],mm)
   for cn,ofs in [('control',co),('gbcorr',go)]:
    cal=calibrated_forward(fw,ofs,b)[0].reshape(ns,N,2);cv,cth=cal[...,0],cal[...,1]
    for bn,ba in [('control',cb),('gbcorr',gb)]:
     Uv,Ut,vbar,tbar=ba;pv,pt=project_state(cv,cth,ba);pv,pt=_restore_known_setpoints(pv,pt,b,N);r=names[(cn,bn)];add(r,pv,pt,rv,rt,b,dev);ev,et=cv-rv,angle_diff(cth,rt);r['ecv']+=en(ev);r['ect']+=en(et);r['cspv']+=en(pv-rv);r['cspt']+=en(angle_diff(pt,rt));r['offv']+=en(off_error(ev,Uv));r['offt']+=en(off_error(et,Ut))
 out={'protocol':{'control_train_test':[len(ct),len(ce)],'gbcorr_train_test':[len(gt),len(ge)],'rank_requested':x.rank,'rank_actual':{'control':[int(cb[0].shape[1]),int(cb[1].shape[1])],'gbcorr':[int(gb[0].shape[1]),int(gb[1].shape[1])]},'T':'control calibration + control basis','R':'GBcorr calibration + GBcorr basis','restoration':'existing restore_known applied after every projection'},'reference':{},'cells':{}}
 for bn,q in ref.items(): out['reference'][bn]=dict(test_retained_variance={'magnitude':1-q['ov']/q['tv'],'angle':1-q['ot']/q['tt']},reference_off_energy={'magnitude':q['ov'],'angle':q['ot']},oracle=dict(vmag_rmse=math.sqrt(q['pv']/q['n']),angle_rmse_deg=math.degrees(math.sqrt(q['pt']/q['n'])),mean_pb=float(torch.cat([torch.from_numpy(v) for v in q['pb']]).mean()),max_pb=q['mx']))
 out['training_explained_variance']={'control':{'magnitude':float(manifold_basis(ct,x.batch,k=x.rank)[4]),'angle':float(manifold_basis(ct,x.batch,k=x.rank)[5])},'gbcorr':{'magnitude':float(manifold_basis(gt,x.batch,k=x.rank)[4]),'angle':float(manifold_basis(gt,x.batch,k=x.rank)[5])}}
 for (cn,bn),r in names.items():
  m=finish(r);idn={}
  for bl,key in [('magnitude','v'),('angle','t')]:
   ro=ref[bn]['ov' if key=='v' else 'ot'];eo=m['error_off_energy'][bl];lhs=m['csp_error_energy'][bl]-m['cal_error_energy'][bl];rhs=ro-eo;idn[bl]=dict(reference_off=ro,error_off=eo,rhs=rhs,lhs=lhs,relative_discrepancy=abs(lhs-rhs)/max(abs(rhs),abs(lhs),1.),off_error_fraction=eo/m['cal_error_energy'][bl])
  m['identity']=idn;out['cells'][f'{cn}_cal__{bn}_basis']=m
 Path(x.out_json).write_text(json.dumps(out,indent=2));
 with open(x.out_csv,'w',newline='') as f:
  w=csv.writer(f);w.writerow(['section','cell','block','metric','value']);
  for k,v in out['reference'].items():
   for b,z in v['test_retained_variance'].items():w.writerow(['reference',k,b,'retained_variance',z])
  for k,v in out['cells'].items():
   for n,z in v.items():
    if not isinstance(z,dict):w.writerow(['cell',k,'',n,z])
   for b,z in v['identity'].items():
    for n,q in z.items():w.writerow(['identity',k,b,n,q])
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
