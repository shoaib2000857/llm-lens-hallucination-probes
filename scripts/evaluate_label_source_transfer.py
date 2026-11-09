#!/usr/bin/env python3
"""Cross-label/model/domain predictions and honest human evaluation when labels exist."""
import csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'));sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from portability.probe_training import Corpus,make_probe,predict
from portability.storage import atomic_json,sha256
from reproduce_saved_scores import metrics
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/label_source_transfer_v1'
def main():
 torch.set_num_threads(8);device=torch.device('cuda');OUT.mkdir(parents=True,exist_ok=True)
 package=ROOT/'human_annotation/revision_v2'
 keys={r['annotation_id']:r for r in json.loads((package/'selection_key.json').read_text())};humans={};incomplete=0
 for r in csv.DictReader((package/'annotation.csv').open()):
  a,b,c=r['annotator_a'],r['annotator_b'],r['adjudicator']
  if a not in ('0','1') or b not in ('0','1'):incomplete+=1;continue
  value=c if c in ('0','1') else a if a==b else None
  if value is not None:humans[keys[r['annotation_id']]['sample_id']]=int(value)
 reports=[]
 for target in ['ministral3_8b','qwen3_8b']+[f'distribution_v1_{d}_{g}' for d in ['hotpotqa','pubmedqa'] for g in ['ministral3_8b','qwen3_8b']]:
  data=Corpus(ROOT,target);idx=data.indices('test');ids=[data.rows[i]['sample_id'] for i in idx]
  for source in ['ministral3_8b','qwen3_8b']:
   for name in ['mean_linear','M1']:
    for label in ['mistral','qwen','phi','consensus']:
     for seed in [42,43,44]:
      folder=ROOT/'artifacts/probes'/f'{source}__{name}__s{seed}__{("scratch_full" if label=="consensus" else "label_"+label)}'
      cp=folder/'best.pt';digest=sha256(cp);out=OUT/f'{source}_to_{target}__{name}__s{seed}__label_{label}';out.mkdir(parents=True,exist_ok=True)
      if (out/'predictions.json').exists():
       rows=json.loads((out/'predictions.json').read_text());provenance=json.loads((out/'provenance.json').read_text())
       if provenance['checkpoint_sha256']!=digest or [r['sample_id'] for r in rows]!=ids:raise ValueError('Changed checkpoint or cohort')
       p=np.array([r['probability'] for r in rows])
      else:
       ck=torch.load(cp,map_location='cpu',weights_only=True);net=make_probe(name,ck['shape']).to(device);net.load_state_dict(ck['model']);p=predict(net,data,idx,device,64);del net;torch.cuda.empty_cache()
       atomic_json(out/'predictions.json',[dict(sample_id=s,probability=float(v)) for s,v in zip(ids,p)])
       atomic_json(out/'provenance.json',dict(checkpoint_sha256=digest,training_label=label,training_corpus=source,target_corpus=target,cohort='all-three-valid target test only',manifest_sha256=sha256(ROOT/'artifacts'/f'stage1_{target}/pooled/manifest.json')))
      targets={j:metrics(np.array([data.rows[i]['label'] if j=='consensus' else data.rows[i]['judge_labels'][j] for i in idx]),p) for j in ['mistral','qwen','phi','consensus']}
      selected=[k for k,s in enumerate(ids) if s in humans];human_report=dict(status='pending_human_annotations',n=0)
      if selected:
       hy=np.array([humans[ids[k]] for k in selected]);hp=p[selected];human_report=dict(status='partial_or_complete',metrics=metrics(hy,hp),n=len(selected),caveat='Stratified annotation sample: not a population error-rate estimate.')
      row=dict(experiment=out.name,training_label=label,source=source,target=target,probe=name,seed=seed,test_targets=targets,human_evaluation=human_report)
      atomic_json(out/'evaluation.json',row);reports.append(row)
  print('Evaluated label-source transfer:',target,flush=True)
 atomic_json(OUT/'summary.json',dict(evaluations=reports,human_adjudicated_or_agreed=len(humans),human_incomplete=incomplete,caveat='Same target test cases across label sources; independent source development stopping. Human fields remain pending until genuine annotations are supplied. Cross-model/domain tests do not recalibrate or change score orientation.'))
 print(json.dumps(dict(evaluations=len(reports),human_labels=len(humans))),flush=True)
if __name__=='__main__':main()
