#!/usr/bin/env python3
"""Question-paired pooled-state alignment. Responses are NOT held constant across generators."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
from portability.probe_training import Corpus,make_probe
from portability.storage import atomic_json
from reproduce_saved_scores import metrics

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);ap.add_argument('--pairs',nargs='+',type=int,default=[25,50,100,250,500]);ap.add_argument('--device',default='cuda');args=ap.parse_args()
 torch.set_num_threads(8);device=torch.device(args.device)
 data={c:Corpus(args.root,c) for c in ['ministral3_8b','qwen3_8b']}
 for source,target in [('ministral3_8b','qwen3_8b'),('qwen3_8b','ministral3_8b')]:
  src,tgt=data[source],data[target]
  if src.shape[-1]!=tgt.shape[-1]:raise ValueError('Explicit new dimensional adapter required')
  source_ids={r['original_id']:i for i,r in enumerate(src.rows) if r['split']=='train'}
  pairs=[(source_ids[r['original_id']],i) for i,r in enumerate(tgt.rows) if r['split']=='train' and r['original_id'] in source_ids]
  for seed in args.seeds:
   rng=np.random.default_rng(seed);order=rng.permutation(len(pairs));te=tgt.indices('test');cal=rng.permutation(tgt.indices('train'))[:min(100,len(tgt.indices('train')))]
   xt=torch.tensor(np.asarray(tgt.array[te]).mean(1),device=device);xc=torch.tensor(np.asarray(tgt.array[cal]).mean(1),device=device)
   for count in args.pairs:
    if count>len(pairs):continue
    selected=[pairs[i] for i in order[:count]];si,ti=zip(*selected)
    x=torch.tensor(np.asarray(tgt.array[list(ti)]).mean(1),device=device);y=torch.tensor(np.asarray(src.array[list(si)]).mean(1),device=device)
    mx=x.mean(0);my=y.mean(0);a=x-mx;b=y-my
    gram=a@a.T;lam=.01*gram.diag().mean().clamp_min(1e-6)
    dual=torch.linalg.solve(gram+lam*torch.eye(count,device=device),b)
    torch.manual_seed(seed);q=torch.linalg.qr(torch.randn(x.shape[1],min(128,count-1,x.shape[1]),device=device)).Q
    u,_,vh=torch.linalg.svd((a@q).T@(b@q));rotation=u@vh
    methods={'ridge_dual':lambda z:((z-mx)@a.T)@dual+my,
             'subspace_procrustes':lambda z:(z-mx)+(((z-mx)@q)@(rotation-torch.eye(q.shape[1],device=device)))@q.T+my}
    for method,mapper in methods.items():
     for name in ['mean_linear','M0']:
      ckpath=args.root/'artifacts/probes'/f'{source}__{name}__s{seed}__scratch_full/best.pt'
      if not ckpath.exists():continue
      ck=torch.load(ckpath,map_location='cpu',weights_only=True);net=make_probe(name,src.shape).to(device);net.load_state_dict(ck['model']);net.eval()
      with torch.inference_mode():
       pred=net(mapper(xt)[:,None,:]).sigmoid().cpu().numpy();pc=net(mapper(xc)[:,None,:]).sigmoid().cpu().numpy()
      out=args.root/'artifacts/probes'/f'{source}_to_{target}__{name}__s{seed}__{method}_{count}';out.mkdir(parents=True,exist_ok=True)
      report=dict(method=method,pairs=count,target_labels_for_alignment=0,source_training_labels='see source config',caveat='Pairs match original QA question IDs; generated answers differ. This is not same-response representation alignment.',metrics=metrics(tgt.y[te],pred),pair_ids=[tgt.rows[i]['sample_id'] for i in ti])
      if len(np.unique(tgt.y[cal]))==2:
       fit=LogisticRegression().fit(logit(np.clip(pc,1e-6,1-1e-6))[:,None],tgt.y[cal]);pcal=fit.predict_proba(logit(np.clip(pred,1e-6,1-1e-6))[:,None])[:,1]
       report['alignment_plus_platt']=dict(target_labels=len(cal),metrics=metrics(tgt.y[te],pcal))
      atomic_json(out/'alignment.json',report)
      atomic_json(out/'predictions.json',[dict(sample_id=tgt.rows[i]['sample_id'],label=int(tgt.y[i]),probability=float(p)) for i,p in zip(te,pred)])
      print(out.name,flush=True)
if __name__=='__main__':main()
