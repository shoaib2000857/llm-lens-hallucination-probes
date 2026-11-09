#!/usr/bin/env python3
"""Current-run scores and integrity audit from saved predictions; no inference."""
import json,statistics,csv
from pathlib import Path
from collections import Counter
root=Path(__file__).resolve().parents[1];base=root/'artifacts/probes';out=root/'results/current_run';out.mkdir(exist_ok=True)
headlines=[];issues=[];counts=Counter()
for folder in sorted(base.iterdir()):
 if not folder.is_dir():continue
 counts['directories']+=1
 for name in ['complete.json','evaluation.json','skipped.json','alignment.json']:
  if (folder/name).exists():counts[name]+=1
 if (folder/'config.json').exists():
  cfg=json.loads((folder/'config.json').read_text())
  if 'train_ids' in cfg:
   tr=set(cfg['train_ids']);va=set(cfg['validation_ids'])
   if tr&va:issues.append({'experiment':folder.name,'issue':'training_validation_overlap'})
   if len(tr)!=cfg['train_labels'] or len(va)!=cfg['validation_labels']:issues.append({'experiment':folder.name,'issue':'label_count_mismatch'})
   if cfg['budget']!='full' and len(tr)+len(va)!=int(cfg['budget']):issues.append({'experiment':folder.name,'issue':'incorrect_label_budget'})
   if not ((folder/'complete.json').exists() or (folder/'skipped.json').exists()):issues.append({'experiment':folder.name,'issue':'unfinished_training'})
   if (folder/'predictions.json').exists():
    rows=json.loads((folder/'predictions.json').read_text());test={r['sample_id'] for r in rows}
    if test&(tr|va):issues.append({'experiment':folder.name,'issue':'test_leakage'})
for corpus in ['ministral3_8b','qwen3_8b']:
 for probe in ['mean_linear','last_linear','last_mlp','best_layer_linear','M0','M1','M2','M3','M4']:
  vals=[]
  for seed in [42,43,44]:
   folder=base/f'{corpus}__{probe}__s{seed}__scratch_full'
   m=json.loads((folder/'evaluation.json').read_text())['metrics'];rows=json.loads((folder/'predictions.json').read_text())
   tp=sum(r['label']==1 and r['probability']>=.5 for r in rows);tn=sum(r['label']==0 and r['probability']<.5 for r in rows)
   pos=sum(r['label'] for r in rows);neg=len(rows)-pos
   vals.append(dict(m,accuracy=(tp+tn)/len(rows),balanced_accuracy=(tp/pos+tn/neg)/2))
  row=dict(corpus=corpus,probe=probe,seeds=len(vals),n=vals[0]['n'],positives=vals[0]['positives'])
  for metric in ['accuracy','balanced_accuracy','auroc','average_precision','f1_05','ece','brier']:
   v=[r[metric] for r in vals];row[metric]=statistics.mean(v);row[metric+'_std']=statistics.stdev(v)
  row['majority_class_accuracy']=max(row['positives'],row['n']-row['positives'])/row['n'];headlines.append(row)
(out/'headline.json').write_text(json.dumps(headlines,indent=2));(out/'integrity.json').write_text(json.dumps(dict(counts=counts,issues=issues),indent=2))
with (out/'headline.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(headlines[0]));w.writeheader();w.writerows(headlines)
lines=['# Current-run target-trained scores','', 'Three seeds; accuracy/F1 use threshold 0.5. SQuAD v2 consensus labels only.','', '| Model | Probe | Accuracy | AUROC | F1 |','|---|---|---:|---:|---:|']
for r in headlines:
 lines.append(f"| {r['corpus']} | {r['probe']} | {100*r['accuracy']:.1f}% | {r['auroc']:.3f} | {r['f1_05']:.3f} |")
(out/'headline.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines));print('INTEGRITY',json.dumps(dict(counts=counts,issues=issues)))
