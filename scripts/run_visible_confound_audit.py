#!/usr/bin/env python3
"""Source-trained visible controls and within-stratum diagnostics; no LLM inference."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'));sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.linear_model import LogisticRegression
from portability.probe_training import Corpus
from portability.storage import atomic_json,ShardDataset
from reproduce_saved_scores import metrics
from update_confound_baselines import features
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/visible_confound_audit_v1'
def main():
 OUT.mkdir(parents=True,exist_ok=True);reports=[]
 names=['input_tokens','output_tokens','context_response_overlap','refusal_pattern','mean_logprob','min_logprob','mean_entropy']
 for source in ['ministral3_8b','qwen3_8b']:
  parent=Corpus(ROOT,source);xp=features(parent);train=parent.indices('train');models={}
  for signal,cols in [(n,[i]) for i,n in enumerate(names)]+[('combined_visible',[0,1,2,3])]:
   mean=xp[train][:,cols].mean(0);std=np.maximum(xp[train][:,cols].std(0),1e-8);net=LogisticRegression(max_iter=1000,random_state=42).fit((xp[train][:,cols]-mean)/std,parent.y[train]);models[signal]=(net,cols,mean,std)
  targets=[source]+[f'distribution_v1_{d}_{source}' for d in ['hotpotqa','pubmedqa']]
  if source=='qwen3_8b':targets +=[f'{prefix}_qwen3_8b_step{s:06d}' for prefix in ['updates_v1','updates_v1r43','updates_v1r44'] for s in [50,150,300]]
  for target in targets:
   data=parent if target==source else Corpus(ROOT,target);x=xp if target==source else features(data);idx=data.indices('test');y=data.y[idx];sid=[data.rows[i]['sample_id'] for i in idx];metadata=ShardDataset(ROOT/'artifacts'/f'stage1_{target}/hidden_states').records
   for signal,(net,cols,mean,std) in models.items():
    p=net.predict_proba((x[idx][:,cols]-mean)/std)[:,1];out=OUT/f'{target}__{signal}';out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'predictions.json',[dict(sample_id=s,label=int(t),probability=float(v)) for s,t,v in zip(sid,y,p)])
    config=dict(source=source,target=target,signal=signal,feature_columns=[names[c] for c in cols],train_ids=[parent.rows[i]['sample_id'] for i in train],training_labels=len(train),target_training_labels=0,mean=mean.tolist(),std=std.tolist(),coefficients=net.coef_.tolist(),intercept=net.intercept_.tolist())
    atomic_json(out/'config.json',config);reports.append(dict(source=source,target=target,signal=signal,metrics=metrics(y,p)))
   probe_paths=list((ROOT/'results/distribution_v1').glob(f'{target}__*/predictions.json')) if target.startswith('distribution_v1') else list((ROOT/'results/model_updates_v1').glob(f'{target}__*__frozen/predictions.json')) if target.startswith('updates_v1') else list((ROOT/'artifacts/probes').glob(f'{target}__*__scratch_full/predictions.json'))
   strat=[]
   groups={'answerable':np.array([not metadata[i]['is_unanswerable'] for i in idx]),'unanswerable':np.array([metadata[i]['is_unanswerable'] for i in idx]),'refusal_pattern':x[idx,3].astype(bool),'no_refusal_pattern':~x[idx,3].astype(bool)}
   for low,high in [(0,16),(17,32),(33,64),(65,96)]:groups[f'output_tokens_{low}_{high}']=(x[idx,1]>=low)&(x[idx,1]<=high)
   for path in probe_paths:
    if '__cv_' in path.parent.name:continue
    lookup={r['sample_id']:r for r in json.loads(path.read_text())}
    if set(lookup)!=set(sid) or any(lookup[s]['label']!=int(t) for s,t in zip(sid,y)):raise ValueError('Confound/probe test alignment differs')
    p=np.array([lookup[s]['probability'] for s in sid])
    for group,mask in groups.items():
     if mask.sum():strat.append(dict(experiment=path.parent.name,stratum=group,metrics=metrics(y[mask],p[mask])))
   atomic_json(OUT/f'{target}__strata.json',strat);print('Visible controls/strata:',target,flush=True)
 atomic_json(OUT/'summary.json',dict(controls=reports,caveat='Visible controls fit source training labels only. Answerability is native input metadata, not response factuality. Conditional metrics are descriptive observational controls, not proof that causal confounding is removed. Undefined AUROC in one-class strata remains null. Small/imbalanced strata require cautious interpretation.'))
if __name__=='__main__':main()
