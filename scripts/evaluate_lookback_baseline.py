#!/usr/bin/env python3
"""Matched response-level Lookback features and original default logistic classifier."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'));sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold,GroupShuffleSplit
from portability.probe_training import Corpus
from portability.storage import atomic_json,sha256,ShardDataset
from reproduce_saved_scores import metrics,bootstrap
ROOT=Path(__file__).resolve().parents[1]
def load(corpus):
 folder=ROOT/'artifacts/lookback_v1'/corpus
 if not (folder/'complete.json').exists():raise ValueError('Incomplete feature replay '+corpus)
 lookup={}
 for p in sorted(folder.glob('shard_*.json')):
  c=json.loads(p.read_text());a=np.load(folder/c['array'])
  if sha256(folder/c['array'])!=c['sha256'] or len(a)!=len(c['rows']):raise ValueError('Corrupt Lookback features')
  for r,v in zip(c['rows'],a):
   if r['sample_id'] in lookup:raise ValueError('Duplicate feature ID')
   lookup[r['sample_id']]=v.reshape(-1)
 data=Corpus(ROOT,corpus);eligible={i for i,r in enumerate(data.rows) if r['sample_id'] in lookup}
 x=np.zeros((len(data.rows),len(next(iter(lookup.values())))),dtype=np.float32)
 for i in eligible:x[i]=lookup[data.rows[i]['sample_id']]
 ids=lambda split:np.array([i for i in data.indices(split) if i in eligible],dtype=int)
 return data,x,ids

def save(corpus,mode,data,idx,p,config):
 out=ROOT/'results/lookback_v1'/f'{corpus}__{mode}';out.mkdir(parents=True,exist_ok=True)
 rows=[dict(sample_id=data.rows[i]['sample_id'],label=int(data.y[i]),probability=float(v)) for i,v in zip(idx,p)]
 atomic_json(out/'predictions.json',rows);atomic_json(out/'config.json',config)
 report=dict(metrics=metrics(data.y[idx],p),confidence_intervals=bootstrap(data.y[idx],p,500,42),per_judge={j:metrics(np.array([data.rows[i]['judge_labels'][j] for i in idx]),p) for j in ['mistral','qwen','phi']},coverage=dict(supported_valid_test=len(idx),canonical_valid_test=len(data.indices('test'))),caveat='Original Lookback per-head feature/default logistic method adapted to modern models and response-level consensus labels; not reproduction of original span benchmarks. Unsupported truncated prompt partitions excluded; compare probes on exactly these IDs.')
 atomic_json(out/'evaluation.json',report)
 # Recompute all existing frozen probes on same cases; never compare different subsets.
 comparisons=[];expected={r['sample_id']:r['label'] for r in rows}
 for path in (ROOT/'results/distribution_v1').glob(f'{corpus}__*/predictions.json'):
  if '__cv_' in path.parent.name:continue
  values={r['sample_id']:r for r in json.loads(path.read_text())}
  if not expected.keys()<=values.keys():continue
  if any(values[s]['label']!=y for s,y in expected.items()):raise ValueError('Matched probe label mismatch')
  comparisons.append(dict(experiment=path.parent.name,metrics=metrics([r['label'] for r in rows],[values[r['sample_id']]['probability'] for r in rows])))
 if corpus.startswith('updates_v1'):
  paths=(ROOT/'results/model_updates_v1').glob(f'{corpus}__*__frozen/predictions.json')
 else:paths=(ROOT/'artifacts/probes').glob(f'{corpus}__*__scratch_full/predictions.json') if corpus in ['qwen3_8b','ministral3_8b'] else []
 for path in paths:
  values={r['sample_id']:r for r in json.loads(path.read_text())}
  if not expected.keys()<=values.keys():continue
  if any(values[s]['label']!=y for s,y in expected.items()):raise ValueError('Matched source/update probe labels differ')
  comparisons.append(dict(experiment=path.parent.name,metrics=metrics([r['label'] for r in rows],[values[r['sample_id']]['probability'] for r in rows])))
 atomic_json(out/'matched_probe_comparisons.json',comparisons)
 print(json.dumps(dict(corpus=corpus,mode=mode,metrics=report['metrics'])),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--include-updates',action='store_true');ap.add_argument('--targets',nargs='+',help='Explicit existing target corpora; avoids rerunning completed domain comparisons');args=ap.parse_args()
 for generator in ['ministral3_8b','qwen3_8b']:
  if args.targets and generator!='qwen3_8b':continue
  parent,xp,ip=load(generator);train=ip('train');dev=ip('dev');t0=time.time();model=LogisticRegression(max_iter=1000,random_state=42).fit(xp[train],parent.y[train]);seconds=time.time()-t0
  if model.n_iter_[0]>=1000:raise ValueError('Lookback classifier did not converge')
  config=dict(method='Lookback Lens response-level adaptation',estimator='sklearn LogisticRegression default C1,L2,lbfgs,unscaled features,max_iter1000',source=generator,train_ids=[parent.rows[i]['sample_id'] for i in train],dev_ids=[parent.rows[i]['sample_id'] for i in dev],training_seconds=seconds,train_labels=len(train),classifier_parameters=xp.shape[1]+1,label_one='hallucination, inverse of original paper factual-positive convention',source_features_config=json.loads((ROOT/'artifacts/lookback_v1'/generator/'config.json').read_text()))
  save(generator,'source',parent,ip('test'),model.predict_proba(xp[ip('test')])[:,1],config)
  targets=[f'distribution_v1_{d}_{generator}' for d in ['hotpotqa','pubmedqa']]
  if generator=='qwen3_8b' and args.include_updates:targets += [f'updates_v1_qwen3_8b_step{s:06d}' for s in [50,150,300]]
  if args.targets:targets=args.targets
  for name in targets:
   data,x,ix=load(name);test=ix('test')
   if x.shape[1]!=xp.shape[1]:raise ValueError('Feature geometry mismatch; requires explicit head mapping')
   save(name,'frozen_source',data,test,model.predict_proba(x[test])[:,1],config)
   if name.startswith('updates_v1'):
    tr=ix('train');fresh=LogisticRegression(max_iter=1000,random_state=42).fit(x[tr],data.y[tr])
    if fresh.n_iter_[0]>=1000:raise ValueError('Target Lookback classifier did not converge')
    save(name,'target_trained',data,test,fresh.predict_proba(x[test])[:,1],dict(method='target-trained Lookback',train_ids=[data.rows[i]['sample_id'] for i in tr],train_labels=len(tr),classifier_parameters=x.shape[1]+1))
   else:
    meta=ShardDataset(ROOT/'artifacts'/f'stage1_{name}/hidden_states').records;groups=np.array([hashlib.sha256(meta[i]['context'].encode()).hexdigest() for i in test]);y=data.y[test]
    for seed in [42,43,44]:
     predictions=np.full(len(test),np.nan);folds=[]
     for fold,(fit,test_local) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(test,y,groups)):
      tr,va=next(GroupShuffleSplit(1,test_size=.25,random_state=seed+fold).split(fit,groups=groups[fit]))
      local_train=fit[tr];train_idx=test[local_train]
      # Same600/200 context split as our probe controls; dev reserved, defaults require no tuning.
      net=LogisticRegression(max_iter=1000,random_state=seed).fit(x[train_idx],data.y[train_idx]);predictions[test_local]=net.predict_proba(x[test[test_local]])[:,1]
      if net.n_iter_[0]>=1000:raise ValueError('OOF logistic classifier did not converge')
      folds.append(dict(fold=fold,train_ids=[data.rows[i]['sample_id'] for i in train_idx],validation_ids=[data.rows[i]['sample_id'] for i in test[fit[va]]],test_ids=[data.rows[i]['sample_id'] for i in test[test_local]]))
     if not np.isfinite(predictions).all():raise ValueError('OOF coverage mismatch')
     save(name,f'target_oof_s{seed}',data,test,predictions,dict(method='Lookback target-trained grouped OOF reference',seed=seed,folds=folds,scope='OOF reference, not independent final holdout; folds built on supported attention cohort'))
if __name__=='__main__':main()
