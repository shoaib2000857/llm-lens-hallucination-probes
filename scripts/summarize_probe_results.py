#!/usr/bin/env python3
"""Machine-derived CSV/Markdown/LaTeX plus seed mean/std; no manual metric transcription."""
import csv,json,re
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];rows=[]
for folder in sorted((root/'artifacts/probes').glob('*')):
 if not folder.is_dir():continue
 if (folder/'evaluation.json').exists():
  d=json.loads((folder/'evaluation.json').read_text());cfg=json.loads((folder/'config.json').read_text()) if (folder/'config.json').exists() else {}
  rows.append(dict(experiment=folder.name,mode=d['mode'],target_labels=cfg.get('total_target_labels',0),**d['metrics']))
 if (folder/'alignment.json').exists():
  d=json.loads((folder/'alignment.json').read_text());rows.append(dict(experiment=folder.name,mode='alignment',target_labels=0,**d['metrics']))
  if 'alignment_plus_platt' in d:rows.append(dict(experiment=folder.name+'__platt',mode='alignment_plus_calibration',target_labels=d['alignment_plus_platt']['target_labels'],**d['alignment_plus_platt']['metrics']))
 p=folder/'calibration/calibration.json'
 if p.exists():
  for d in json.loads(p.read_text()):
   if 'metrics' in d:rows.append(dict(experiment=folder.name+f"__{d['method']}_{d['budget']}",mode=d['mode'],target_labels=d['budget'],**d['metrics']))
out=root/'results/stage3';out.mkdir(parents=True,exist_ok=True)
if rows:
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (out/'metrics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 header=['experiment','mode','target_labels','auroc','average_precision','f1_05','ece','brier']
 (out/'metrics.md').write_text('| '+' | '.join(header)+' |\n| '+' | '.join(['---']*len(header))+' |\n'+'\n'.join('| '+' | '.join(str(r.get(k)) for k in header)+' |' for r in rows)+'\n')
 def esc(v):return str(v).replace('_',r'\_')
 (out/'metrics.tex').write_text('\\begin{tabular}{lllrrrrr}\n'+' & '.join(map(esc,header))+r' \\'+'\n'+ '\n'.join(' & '.join(esc(r.get(k)) for k in header)+r' \\' for r in rows)+'\n\\end{tabular}\n')
 groups={}
 for r in rows:groups.setdefault(re.sub(r'__s\d+__','__sSEED__',r['experiment']),[]).append(r)
 aggregate=[]
 for name,group in groups.items():
  row=dict(experiment=name,seeds=len(group),mode=group[0]['mode'])
  for metric in ['auroc','average_precision','f1_05','ece','brier']:
   values=[r[metric] for r in group if r.get(metric) is not None]
   row[metric+'_mean']=float(np.mean(values)) if values else None
   row[metric+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else None
  aggregate.append(row)
 (out/'seed_aggregate.json').write_text(json.dumps(aggregate,indent=2,allow_nan=False))
print('Saved',len(rows),'evaluation rows')
