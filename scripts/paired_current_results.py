#!/usr/bin/env python3
"""Paired test-example bootstrap; training seeds held fixed, never retunes thresholds."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
root=Path(__file__).resolve().parents[1];base=root/'artifacts/probes';out=root/'results/current_run'
def auc(y,p):
 order=np.argsort(p,kind='stable');ys=y[order];ps=p[order];bounds=np.r_[0,np.flatnonzero(np.diff(ps))+1,len(ps)]
 ranks=np.repeat((bounds[:-1]+bounds[1:]+1)/2,np.diff(bounds));n1=ys.sum();n0=len(ys)-n1
 return float((ranks[ys==1].sum()-n1*(n1+1)/2)/(n1*n0))
def f1(y,p):
 pred=p>=.5;tp=np.sum(pred&(y==1));den=2*tp+np.sum(pred&(y==0))+np.sum(~pred&(y==1));return float(2*tp/den) if den else 0.
def load(name):
 rows=json.loads((base/name/'predictions.json').read_text());d={r['sample_id']:r for r in rows}
 if len(d)!=len(rows):raise ValueError('Duplicate predictions')
 return d
comparisons=[]
for corpus in ['ministral3_8b','qwen3_8b']:
 for probe in ['M0','M1','M2','M3','M4']:
  comparisons.append((corpus,probe+' warm minus scratch',f'{probe}__warm_full',f'{probe}__scratch_full'))
 comparisons.extend([(corpus,'M3 minus M2','M3__scratch_full','M2__scratch_full'),(corpus,'M1 minus best linear layer','M1__scratch_full','best_layer_linear__scratch_full')])
results=[]
for corpus,title,left,right in comparisons:
 a=[];b=[];labels=None;ids=None
 for seed in [42,43,44]:
  p,mode=left.split('__');d1=load(f'{corpus}__{p}__s{seed}__{mode}')
  p,mode=right.split('__');d2=load(f'{corpus}__{p}__s{seed}__{mode}')
  if d1.keys()!=d2.keys():raise ValueError('Paired IDs differ')
  sid=sorted(d1);y=np.array([d1[i]['label'] for i in sid])
  if any(d1[i]['label']!=d2[i]['label'] for i in sid):raise ValueError('Paired labels differ')
  if labels is not None and (ids!=sid or not np.array_equal(labels,y)):raise ValueError('Seed labels/IDs differ')
  labels=y;ids=sid;a.append([d1[i]['probability'] for i in sid]);b.append([d2[i]['probability'] for i in sid])
 a=np.array(a);b=np.array(b);strata=[np.flatnonzero(labels==c) for c in [0,1]];rng=np.random.default_rng(42);samples=[]
 for _ in range(1000):
  idx=np.r_[rng.choice(strata[0],len(strata[0])),rng.choice(strata[1],len(strata[1]))];y=labels[idx]
  samples.append([np.mean([auc(y,p[idx])-auc(y,q[idx]) for p,q in zip(a,b)]),np.mean([f1(y,p[idx])-f1(y,q[idx]) for p,q in zip(a,b)])])
 samples=np.array(samples);ca=(a.mean(0)>=.5)==labels;cb=(b.mean(0)>=.5)==labels
 n10=int(np.sum(ca&~cb));n01=int(np.sum(~ca&cb));pv=binomtest(n10,n10+n01,.5).pvalue if n10+n01 else 1.
 row=dict(corpus=corpus,comparison=title,n=len(labels),seeds=3,bootstrap=1000,auroc_delta=float(np.mean([auc(labels,p)-auc(labels,q) for p,q in zip(a,b)])),auroc_delta_ci95=np.quantile(samples[:,0],[.025,.975]).tolist(),f1_delta=float(np.mean([f1(labels,p)-f1(labels,q) for p,q in zip(a,b)])),f1_delta_ci95=np.quantile(samples[:,1],[.025,.975]).tolist(),ensemble_mcnemar_exact_p=float(pv),ensemble_discordant=[n10,n01],scope='Paired response bootstrap averaging three seed differences; conditions on fitted seeds. McNemar compares averaged-probability ensembles at 0.5.')
 results.append(row)
ordered=sorted(range(len(results)),key=lambda i:results[i]['ensemble_mcnemar_exact_p']);previous=0
for rank,i in enumerate(ordered):
 previous=max(previous,min(1,(len(results)-rank)*results[i]['ensemble_mcnemar_exact_p']));results[i]['ensemble_mcnemar_holm_p']=previous
(out/'paired_comparisons.json').write_text(json.dumps(results,indent=2));print('Saved',len(results),'paired comparisons')
