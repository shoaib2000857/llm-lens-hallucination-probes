#!/usr/bin/env python3
"""CPU-only seed summaries and fixed-threshold curves from immutable saved scores."""
import csv,json,sys,statistics
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from portability_reference_report import operating
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/portability_summary_v1'
def write(name,rows):
 if not rows:return
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def summarize(records,keys,values):
 grouped=defaultdict(list)
 for r in records:grouped[tuple(r[k] for k in keys)].append(r)
 out=[]
 for key,rs in sorted(grouped.items()):
  row=dict(zip(keys,key));row['seed_count']=len(rs)
  for metric in values:
   xs=[r[metric] for r in rs if r.get(metric) is not None]
   row[metric+'_mean']=statistics.mean(xs) if xs else None;row[metric+'_std']=statistics.stdev(xs) if len(xs)>1 else None
  out.append(row)
 return out
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 base=ROOT/'results/portability_reference_v1';gaps=json.loads((base/'transfer_gaps.json').read_text());ops=json.loads((base/'operating_points.json').read_text())
 for r in gaps:r['transfer_mode']='__'.join(r['experiment'].split('__')[3:]) or 'frozen_distribution'
 for r in ops:r['transfer_mode']='__'.join(r['experiment'].split('__')[3:]) or 'frozen_distribution'
 gs=summarize(gaps,['source','target','probe','transfer_mode','reference_scope'],['transfer_auroc','reference_auroc','transfer_gap','retention'])
 os=summarize(ops,['source','target','probe','transfer_mode','policy'],['threshold','tpr','fpr','precision','fnr','f1','accuracy'])
 write('transfer_gap_seed_summary.csv',gs);write('operating_point_seed_summary.csv',os)
 label=[]
 for r in json.loads((ROOT/'results/label_source_transfer_v1/summary.json').read_text())['evaluations']:
  for judge,metrics in r['test_targets'].items():
   label.append(dict(source=r['source'],target=r['target'],probe=r['probe'],training_label=r['training_label'],evaluation_label=judge,seed=r['seed'],**{k:metrics[k] for k in ['auroc','average_precision','f1_05','ece','brier']}))
 ls=summarize(label,['source','target','probe','training_label','evaluation_label'],['auroc','average_precision','f1_05','ece','brier']);write('label_source_seed_summary.csv',ls)
 count=0
 with (OUT/'threshold_portability_curves.csv').open('w',newline='') as f:
  fields=['experiment','source','target','probe','seed','threshold','source_dev_tpr','source_dev_fpr','target_test_tpr','target_test_fpr','target_test_precision','target_test_fnr'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  seen=set()
  for r in ops:
   exp=r['experiment']
   if exp in seen:continue
   seen.add(exp);cp=base/f"{r['source']}__{r['probe']}__s{r['seed']}__source_dev.json";dev=json.loads(cp.read_text())
   locations=[ROOT/'artifacts/probes'/exp/'predictions.json',ROOT/'results/model_updates_v1'/exp/'predictions.json',ROOT/'results/distribution_v1'/exp/'predictions.json']
   path=next((p for p in locations if p.exists()),None)
   if path is None:raise ValueError('Missing scores '+exp)
   scores=json.loads(path.read_text());y=[x['label'] for x in scores];p=[x['probability'] for x in scores]
   for n in range(101):
    threshold=n/100;s=operating(dev['labels'],dev['probabilities'],threshold);t=operating(y,p,threshold)
    w.writerow(dict(experiment=exp,source=r['source'],target=r['target'],probe=r['probe'],seed=r['seed'],threshold=threshold,source_dev_tpr=s['tpr'],source_dev_fpr=s['fpr'],target_test_tpr=t['tpr'],target_test_fpr=t['fpr'],target_test_precision=t['precision'],target_test_fnr=t['fnr']));count+=1
 note='''# Saved-evidence analysis\n\nAll summaries condition on existing fitted models. Standard deviations describe probe-training seeds; they are not response confidence intervals or independent generator/domain replicates. Existing paired stratified bootstrap intervals remain in portability_reference_v1. Undefined retention/precision are left blank; missingness must not be interpreted as zero.\n\nTransfer-mode groups remain separate: raw direct transfer, explicit depth mapping and alignment are not pooled as zero-shot. Distribution references are grouped OOF estimates. Source/reference training budgets are preserved in the underlying gap table.\n\nThreshold curves use the fixed probability grid0.00–1.00, selected without target labels. Source development and target test operating outcomes are descriptive; never select a deployment threshold from the target-test curve. No score inversion is introduced.\n\nLabel-source summaries compare the same valid target-test cohort across training labels and seeds. These labels are judges/consensus, not human factuality labels.\n'''
 (OUT/'README.md').write_text(note)
 print(json.dumps(dict(gap_summary_cells=len(gs),operating_summary_cells=len(os),label_summary_cells=len(ls),threshold_curve_rows=count)))
if __name__=='__main__':main()
