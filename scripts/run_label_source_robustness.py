#!/usr/bin/env python3
"""Train judge-specific probes on identical consensus-valid splits; cross-label evaluation."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from portability.probe_training import Corpus,train,predict
from portability.storage import atomic_json
from reproduce_saved_scores import metrics
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=25);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
 torch.set_num_threads(8);device=torch.device('cuda')
 for corpus in ['ministral3_8b','qwen3_8b']:
  original=Corpus(ROOT,corpus)
  for source in ['mistral','qwen','phi']:
   data=copy.copy(original);data.rows=copy.deepcopy(original.rows)
   for row in data.rows:
    if row['label'] is not None:
     label=row['judge_labels'][source]
     if label not in (0,1):raise ValueError('Invalid judge on common cohort')
     row['label']=label
   data.y=np.array([r['label'] if r['label'] is not None else -1 for r in data.rows],dtype=np.float32)
   if args.smoke:
    for split in ['train','dev','test']:
     selected=set(data.indices(split)[:32 if split=='train' else 8])
     for i,r in enumerate(data.rows):
      if r['split']==split and i not in selected:r['label']=None
   for name in ['mean_linear','M1']:
    for seed in ([42] if args.smoke else [42,43,44]):
     out=ROOT/'artifacts/probes'/f'{corpus}__{name}__s{seed}__label_{source}{"_smoke" if args.smoke else ""}'
     out.mkdir(parents=True,exist_ok=True)
     protocol=dict(training_label=source,cohort='all three judges valid',label_sha256=hashlib.sha256(json.dumps([(r['sample_id'],r['label']) for r in data.rows]).encode()).hexdigest(),human_labels_used=False)
     if (out/'label_protocol.json').exists() and json.loads((out/'label_protocol.json').read_text())!=protocol:raise ValueError('Changed labels on resume')
     atomic_json(out/'label_protocol.json',protocol)
     net=train(data,name,'full',seed,out,device,args.epochs,4,64)
     if net is None:continue
     idx=data.indices('test');p=predict(net,data,idx,device,64)
     rows=[dict(sample_id=data.rows[i]['sample_id'],probability=float(v),label=int(data.y[i])) for i,v in zip(idx,p)]
     report={}
     for judge in ['mistral','qwen','phi','consensus']:
      y=np.array([original.rows[i]['label'] if judge=='consensus' else original.rows[i]['judge_labels'][judge] for i in idx])
      report[judge]=metrics(y,p)
     atomic_json(out/'predictions.json',rows);atomic_json(out/'evaluation.json',dict(training_label=source,test_targets=report,n=len(idx),scope='Same held-out responses and common valid cohort; human evaluation pending. No native answerability labels substituted.'))
     print(json.dumps(dict(experiment=out.name,metrics=report)),flush=True)
     del net;torch.cuda.empty_cache()
if __name__=='__main__':main()
