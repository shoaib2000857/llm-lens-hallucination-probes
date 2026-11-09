#!/usr/bin/env python3
"""Machine-readable model-update tables and evidence-bounded performance/drift plots."""
import csv,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from portability.storage import atomic_json
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'results/model_updates_v1';out.mkdir(parents=True,exist_ok=True);rows=[]
paths=list(out.glob('*/evaluation.json'))+list((ROOT/'artifacts/probes').glob('updates_v1*/evaluation.json'))
for path in sorted(paths):
    r=json.loads(path.read_text());parts=path.parent.name.split('__');step=int(re.search(r'step(\d+)',parts[0]).group(1))
    rows.append(dict(experiment=path.parent.name,step=step,probe=parts[1],seed=int(parts[2][1:]),mode=r['mode'],**r['metrics']))
fields=['experiment','step','probe','seed','mode','n','positives','auroc','average_precision','f1_05','ece','brier','mean_probability','diagnostic_inverted_auroc']
with (out/'metrics.csv.tmp').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
(out/'metrics.csv.tmp').replace(out/'metrics.csv')
atomic_json(out/'summary.json',dict(evaluations=len(rows),rows=rows,caveat='One parent, one adaptation domain, one LoRA trajectory. Replication and independent human validation remain pending; inversion is diagnostic only.'))
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print('Tables saved; matplotlib unavailable, plots pending.');raise SystemExit(0)
import statistics
from portability.probe_training import POOLED,RAW
for metric in ['auroc','average_precision','f1_05','ece']:
    fig,ax=plt.subplots(figsize=(7,4))
    for probe in POOLED+RAW:
        source=[]
        for seed in [42,43,44]:
            p=ROOT/'artifacts/probes'/f'qwen3_8b__{probe}__s{seed}__scratch_full/evaluation.json'
            if p.exists():source.append(json.loads(p.read_text())['metrics'].get(metric))
        source=[x for x in source if x is not None];points={0:source}
        for row in rows:
            if row['probe']==probe and row['mode']=='zero_shot_parameter_transfer' and row.get(metric) is not None:points.setdefault(row['step'],[]).append(row[metric])
        xs=sorted(k for k,v in points.items() if v);ys=[statistics.mean(points[x]) for x in xs]
        if xs:ax.plot(xs,ys,'o-',label=probe)
    ax.set(xlabel='Post-deployment LoRA optimizer updates',ylabel=metric,title='Frozen deployment probes: one controlled trajectory');ax.legend(ncol=4,fontsize=8);fig.tight_layout();fig.savefig(out/f'{metric}_vs_steps.pdf');plt.close(fig)
if (out/'representation_drift.json').exists():
    drift=json.loads((out/'representation_drift.json').read_text());cka={r['step']:r['mean_linear_cka'] for r in drift['checkpoints']}
    for metric in ['auroc','ece']:
        fig,ax=plt.subplots(figsize=(6,4))
        for probe in POOLED+RAW:
            points={}
            for row in rows:
                if row['probe']==probe and row['mode']=='zero_shot_parameter_transfer' and row.get(metric) is not None and row['step'] in cka:points.setdefault(row['step'],[]).append(row[metric])
            xs=sorted(points)
            if xs:ax.plot([1-cka[s] for s in xs],[statistics.mean(points[s]) for s in xs],'o-',label=probe)
        ax.set(xlabel='1 − mean layer CKA on fixed development replay',ylabel=metric,title='Exploratory association across three updates');ax.legend(ncol=4,fontsize=8);fig.tight_layout();fig.savefig(out/f'{metric}_vs_drift.pdf');plt.close(fig)
print(f'Saved {len(rows)} evaluation rows and plots.')
