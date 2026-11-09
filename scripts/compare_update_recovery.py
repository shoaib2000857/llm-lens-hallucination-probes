#!/usr/bin/env python3
"""Paired stratified case bootstrap of matched seed-ensemble recovery predictions."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from portability.storage import atomic_json
from portability.probe_training import POOLED, RAW
from reproduce_saved_scores import metrics
ROOT=Path(__file__).resolve().parents[1]

def paired(y,a,b,repeats,seed):
    observed_a=metrics(y,a);observed_b=metrics(y,b);keys=['auroc','average_precision','f1_05','ece','brier'];rng=np.random.default_rng(seed);groups=[np.flatnonzero(y==c) for c in [0,1]]
    if any(not len(g) for g in groups):return dict(status='undefined_single_class')
    values={k:[] for k in keys}
    for _ in range(repeats):
        idx=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);ma=metrics(y[idx],a[idx]);mb=metrics(y[idx],b[idx])
        for k in keys:values[k].append(ma[k]-mb[k])
    return dict(status='complete',difference_direction='method_a minus method_b',method_a_metrics=observed_a,method_b_metrics=observed_b,differences={k:dict(observed=observed_a[k]-observed_b[k],ci95=np.quantile(v,[.025,.975]).tolist()) for k,v in values.items()})

def ensemble(paths):
    if not all(p.exists() for p in paths):return None
    rows=[json.loads(p.read_text()) for p in paths];keys=[[(r['sample_id'],r['label']) for r in rs] for rs in rows]
    if any(k!=keys[0] for k in keys):raise ValueError('Seed prediction order/labels differ')
    return keys[0],np.asarray([r['label'] for r in rows[0]]),np.mean([[r['probability'] for r in rs] for rs in rows],axis=0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--bootstrap',type=int,default=500);ap.add_argument('--seeds',type=int,nargs='+',default=[42,43,44]);args=ap.parse_args()
    root=args.root;out=root/'results/model_update_recovery_v1';out.mkdir(parents=True,exist_ok=True);reports=[];missing=[]
    for step in [50,150,300]:
        corpus=f'updates_v1_qwen3_8b_step{step:06d}'
        for probe in POOLED+RAW:
            frozen=ensemble([root/'results/model_updates_v1'/f'{corpus}__{probe}__s{s}__frozen/predictions.json' for s in args.seeds])
            if frozen is None:raise ValueError('Missing frozen baseline')
            for budget in ['100','500','full']:
                methods={a:ensemble([root/'artifacts/probes'/f'{corpus}__{probe}__s{s}__{a}_{budget}/predictions.json' for s in args.seeds]) for a in ['scratch','warm']}
                pairs=[('warm','scratch')]+([('warm','frozen'),('scratch','frozen')] if budget=='full' else [])
                methods['frozen']=frozen
                for a,b in pairs:
                    key=f'{corpus}__{probe}__{budget}__{a}_minus_{b}'
                    if methods[a] is None or methods[b] is None:missing.append(key);continue
                    if methods[a][0]!=methods[b][0]:raise ValueError('Paired comparison IDs/labels differ')
                    result=dict(corpus=corpus,probe=probe,budget=budget,method_a=a,method_b=b,seeds=args.seeds,bootstrap_repeats=args.bootstrap,**paired(methods[a][1],methods[a][2],methods[b][2],args.bootstrap,42))
                    atomic_json(out/(key+'.json'),result);reports.append(result)
                    print('Compared '+key,flush=True)
    atomic_json(out/'summary.json',dict(comparisons=reports,missing=missing,caveat='Intervals are paired stratified case-bootstrap intervals conditional on these trained seed ensembles and judge labels. They do not capture adaptation trajectory variance. Multiple comparisons are exploratory; no multiple-testing correction or significance claim is applied.'))
    print(f'Saved {len(reports)} paired comparisons; {len(missing)} missing configurations explicitly listed.',flush=True)
if __name__=='__main__': main()
