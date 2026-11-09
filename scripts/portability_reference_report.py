#!/usr/bin/env python3
"""Matched reference gaps and deployment thresholds selected exclusively on source dev."""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'));sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from portability.probe_training import Corpus,make_probe,predict,POOLED,RAW
from portability.storage import atomic_json,sha256
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/portability_reference_v1'
def operating(y,p,t):
 y=np.asarray(y).astype(bool);pred=np.asarray(p)>=t;tp=int(np.sum(pred&y));fp=int(np.sum(pred&~y));fn=int(np.sum(~pred&y));tn=int(np.sum(~pred&~y))
 return dict(tp=tp,fp=fp,fn=fn,tn=tn,tpr=tp/(tp+fn) if tp+fn else None,fpr=fp/(fp+tn) if fp+tn else None,fnr=fn/(tp+fn) if tp+fn else None,precision=tp/(tp+fp) if tp+fp else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,accuracy=(tp+tn)/len(y))
def load(path):
 rows=json.loads(path.read_text());d={r['sample_id']:r for r in rows}
 if len(d)!=len(rows):raise ValueError('Duplicate sample IDs')
 return d
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bootstrap',type=int,default=500);ap.add_argument('--pooled-only',action='store_true');args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(8);device=torch.device('cuda');thresholds={}
 for source in ['ministral3_8b','qwen3_8b']:
  for name in POOLED+(tuple() if args.pooled_only else RAW):
   data=Corpus(ROOT,source,raw=name in RAW);idx=data.indices('dev')
   for seed in [42,43,44]:
    cp=ROOT/'artifacts/probes'/f'{source}__{name}__s{seed}__scratch_full/best.pt';cache=OUT/f'{source}__{name}__s{seed}__source_dev.json';digest=sha256(cp)
    if cache.exists():
     saved=json.loads(cache.read_text())
     if saved['checkpoint_sha256']!=digest:raise ValueError('Source checkpoint changed')
     p=np.array(saved['probabilities']);
     if saved['ids']!=[data.rows[i]['sample_id'] for i in idx]:raise ValueError('Development IDs changed')
    else:
     ck=torch.load(cp,map_location='cpu',weights_only=True);net=make_probe(name,ck['shape']).to(device);net.load_state_dict(ck['model']);p=predict(net,data,idx,device,8 if name in RAW else 64);del net;torch.cuda.empty_cache()
     atomic_json(cache,dict(checkpoint_sha256=digest,ids=[data.rows[i]['sample_id'] for i in idx],labels=data.y[idx].tolist(),probabilities=p.tolist()))
    y=data.y[idx];candidates=np.r_[0,np.unique(p),np.nextafter(1.,2.)];ops=[operating(y,p,t) for t in candidates]
    selected={'fixed_05':.5,'source_dev_max_f1':float(candidates[max(range(len(ops)),key=lambda i:(ops[i]['f1'],candidates[i]))])}
    for cap in [.05,.1]:
     eligible=[i for i,o in enumerate(ops) if o['fpr'] is not None and o['fpr']<=cap];i=max(eligible,key=lambda i:(ops[i]['tpr'],-ops[i]['fpr'],candidates[i]));selected[f'source_dev_fpr_{cap}']=float(candidates[i])
    thresholds[(source,name,seed)]=selected
 rows=[];deployment=[];missing=[]
 paths=list((ROOT/'artifacts/probes').glob('*_to_*__*/predictions.json'))+list((ROOT/'results/model_updates_v1').glob('*__frozen/predictions.json'))+list((ROOT/'results/distribution_v1').glob('*/predictions.json'))
 for path in paths:
  exp=path.parent.name
  if '__cv_' in exp or not ('__frozen' in exp or '_to_' in exp or exp.startswith('distribution_v1')):continue
  bits=exp.split('__');pair=bits[0];name=bits[1];seed=int(bits[2][1:])
  if name not in POOLED+RAW or (args.pooled_only and name in RAW):continue
  if '_to_' in pair:source,target=pair.split('_to_');ref=ROOT/'artifacts/probes'/f'{target}__{name}__s{seed}__scratch_full/predictions.json';scope='independent target test'
  elif pair.startswith('updates_v1'):source='qwen3_8b';target=pair;ref=ROOT/'artifacts/probes'/f'{target}__{name}__s{seed}__scratch_full/predictions.json';scope='independent target test'
  else:
   target=pair;source=next(g for g in ['ministral3_8b','qwen3_8b'] if target.endswith(g));ref=ROOT/'results/distribution_v1'/f'{target}__{name}__s{seed}__cv_scratch/predictions.json';scope='context-grouped OOF reference; not independent target test'
  d=load(path);ids=sorted(d);y=np.array([d[i]['label'] for i in ids]);p=np.array([d[i]['probability'] for i in ids])
  for policy,t in thresholds[(source,name,seed)].items():deployment.append(dict(experiment=exp,source=source,target=target,probe=name,seed=seed,policy=policy,threshold=t,n=len(ids),**operating(y,p,t)))
  if not ref.exists():missing.append(dict(experiment=exp,reference=str(ref.relative_to(ROOT))));continue
  r=load(ref)
  if r.keys()!=d.keys() or any(r[i]['label']!=d[i]['label'] for i in ids):raise ValueError(f'Reference IDs/labels mismatch: {exp}')
  q=np.array([r[i]['probability'] for i in ids]);a=roc_auc_score(y,p);b=roc_auc_score(y,q);rng=np.random.default_rng(seed);strata=[np.flatnonzero(y==c) for c in [0,1]];boot=[]
  for _ in range(args.bootstrap):
   ix=np.concatenate([rng.choice(s,len(s)) for s in strata]);boot.append(roc_auc_score(y[ix],q[ix])-roc_auc_score(y[ix],p[ix]))
  reference_config=ref.parent/'config.json'
  reference_budget=json.loads(reference_config.read_text()).get('total_target_labels') if reference_config.exists() else 'fold-specific; see evaluation.json'
  source_config=ROOT/'artifacts/probes'/f'{source}__{name}__s{seed}__scratch_full/config.json'
  source_budget=json.loads(source_config.read_text())['total_target_labels']
  rows.append(dict(source_training_labels=source_budget,reference_training_labels=reference_budget,experiment=exp,source=source,target=target,probe=name,seed=seed,n=len(ids),transfer_auroc=a,reference_auroc=b,transfer_gap=b-a,retention=(a-.5)/(b-.5) if b>.52 else None,retention_status='defined' if b>.52 else 'reference too close to or below chance',gap_ci95=np.quantile(boot,[.025,.975]).tolist(),reference_scope=scope,reference=str(ref.relative_to(ROOT))))
 atomic_json(OUT/'transfer_gaps.json',rows);atomic_json(OUT/'operating_points.json',deployment);atomic_json(OUT/'missing_references.json',missing)
 for filename,records in [('transfer_gaps.csv',rows),('operating_points.csv',deployment)]:
  if records:
   with (OUT/filename).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 print(json.dumps(dict(paired_references=len(rows),operating_points=len(deployment),missing_references=len(missing))),flush=True)
if __name__=='__main__':main()
