#!/usr/bin/env python3
"""CPU-only tables, figures and paired intervals from saved Lookback scores."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/publication_cpu_v1'

def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = ROOT / 'results/lookback_v1'
    rows = []
    for file in sorted(base.glob('*/evaluation.json')):
        a = json.loads(file.read_text())
        rows.append(dict(experiment=file.parent.name, **a['metrics'], **a['coverage']))
    if not rows:
        raise ValueError('No completed evaluations')
    write_csv('lookback_metrics.csv', rows)
    columns = ['experiment', 'n', 'auroc', 'average_precision', 'f1_05', 'ece']
    md = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---']*len(columns)) + ' |']
    for row in rows:
        md.append('| ' + ' | '.join(f'{row[c]:.3f}' if isinstance(row[c], float) else str(row[c]) for c in columns) + ' |')
    (OUT/'lookback_metrics.md').write_text('\n'.join(md)+'\n')
    tex = ['\\begin{tabular}{lrrrrr}', 'Setting & N & AUROC & AP & F1 & ECE \\\\']
    for row in rows:
        name = row['experiment'].replace('_', '\\_')
        tex.append(name + ' & ' + ' & '.join(str(row[c]) if c=='n' else f'{row[c]:.3f}' for c in columns[1:]) + ' \\\\')
    tex.append('\\end{tabular}')
    (OUT/'lookback_metrics.tex').write_text('\n'.join(tex)+'\n')
    comparisons = []
    for frozen in sorted(base.glob('*__frozen_source/predictions.json')):
        corpus = frozen.parent.name.rsplit('__', 1)[0]
        source = json.loads(frozen.read_text())
        ids = [r['sample_id'] for r in source]
        if len(set(ids)) != len(ids): raise ValueError('Duplicate source IDs')
        y = np.array([r['label'] for r in source]); p = np.array([r['probability'] for r in source])
        refs = sorted(base.glob(corpus+'__target_*/predictions.json'))
        for ref in refs:
            records = json.loads(ref.read_text()); target = {r['sample_id']: r for r in records}
            if len(target)!=len(records) or set(target)!=set(ids): raise ValueError('Reference cohort mismatch')
            if any(target[r['sample_id']]['label']!=r['label'] for r in source): raise ValueError('Reference label mismatch')
            q = np.array([target[s]['probability'] for s in ids])
            positive, negative = np.flatnonzero(y==1), np.flatnonzero(y==0)
            if not len(positive) or not len(negative): raise ValueError('One-class cohort')
            rng = np.random.default_rng(42); deltas=[]
            for _ in range(1000):
                idx = np.concatenate([rng.choice(positive,len(positive),replace=True),rng.choice(negative,len(negative),replace=True)])
                deltas.append([roc_auc_score(y[idx],q[idx])-roc_auc_score(y[idx],p[idx]), average_precision_score(y[idx],q[idx])-average_precision_score(y[idx],p[idx])])
            low, high = np.quantile(deltas,[.025,.975],axis=0)
            comparisons.append(dict(corpus=corpus,reference=ref.parent.name,n=len(y),transfer_auroc=roc_auc_score(y,p),reference_auroc=roc_auc_score(y,q),auroc_gap=roc_auc_score(y,q)-roc_auc_score(y,p),auroc_gap_ci_low=low[0],auroc_gap_ci_high=high[0],ap_gap=average_precision_score(y,q)-average_precision_score(y,p),ap_gap_ci_low=low[1],ap_gap_ci_high=high[1],bootstrap_replicates=1000))
            print('Paired comparison complete:',ref.parent.name,flush=True)
    write_csv('lookback_paired_transfer_gaps.csv',comparisons)
    updates=[r for r in rows if r['experiment'].startswith('updates_v1_')]
    fig, axes=plt.subplots(1,3,figsize=(11,3.5))
    for mode,label in [('frozen_source','Frozen source'),('target_trained','Fresh target')]:
        selected=sorted([r for r in updates if r['experiment'].endswith('__'+mode)],key=lambda r:r['experiment'])
        steps=[int(r['experiment'].split('step')[1].split('__')[0]) for r in selected]
        for ax,metric,title in zip(axes,['auroc','f1_05','ece'],['AUROC','F1 at 0.5','Calibration error']):
            ax.plot(steps,[r[metric] for r in selected],marker='o',label=label);ax.set_xlabel('Adaptation steps');ax.set_ylabel(title);ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout()
    for ext in ['svg','pdf','png']:fig.savefig(OUT/f'lookback_model_update.{ext}',dpi=180)
    plt.close(fig)
    (OUT/'README.md').write_text('Saved-score analysis only; no generator inference or human labels used. Paired stratified bootstrap has 1,000 replicates, seed42, identical sample IDs and labels. Intervals condition on fitted classifiers; OOF cases are not an independent final holdout and intervals do not include retraining uncertainty. Comparisons are exploratory, without multiplicity-adjusted significance claims. Lookback uses our modern response-level protocol, not original span-benchmark reproduction. AP means average precision. F1 uses the unmodified threshold0.5. Coverage exclusions remain explicit in the metric table.\n')
    print(json.dumps(dict(evaluations=len(rows),paired_comparisons=len(comparisons),output=str(OUT))),flush=True)

if __name__ == '__main__': main()
