#!/usr/bin/env python3
"""Generate machine-readable and Markdown result tables from distribution outputs."""
import csv,json,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1];base=root/'results/distribution_v1';rows=[]
for path in sorted(base.glob('*/evaluation.json')):
    r=json.loads(path.read_text());rows.append(dict(experiment=path.parent.name,mode=r['mode'],**r['metrics']))
fields=['experiment','mode','n','positives','accuracy_05','majority_class_accuracy','auroc','average_precision','f1_05','ece','brier','mean_probability','diagnostic_inverted_auroc']
with (base/'metrics.csv.tmp').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
(base/'metrics.csv.tmp').replace(base/'metrics.csv')
lines=['# Distribution round results','', 'Judge-consensus labels; diagnostic inversion is not deployment performance. Controls are out-of-fold estimates.','', '| Experiment | Mode | N | AUROC | AP | F1@0.5 | Accuracy |','|---|---|---:|---:|---:|---:|---:|']
for r in rows:
    vals=[r.get(k) for k in ['auroc','average_precision','f1_05','accuracy_05']]
    lines.append('| '+r['experiment']+' | '+r['mode']+' | '+str(r['n'])+' | '+' | '.join('NA' if v is None else f'{v:.4f}' for v in vals)+' |')
(base/'metrics.md.tmp').write_text('\n'.join(lines)+'\n');(base/'metrics.md.tmp').replace(base/'metrics.md')
print(f'Generated tables for {len(rows)} evaluations')
