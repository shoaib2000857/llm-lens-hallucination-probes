#!/usr/bin/env python3
"""Separate trajectory variance from probe-training seed variance; no pseudo-replication."""
import csv,json,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from portability.storage import atomic_json,sha256
from portability.probe_training import Corpus,POOLED,RAW
ROOT=Path(__file__).resolve().parents[1]

def main():
    issues=[];rows=[];drift=[];coverage={};source=Corpus(ROOT,'qwen3_8b')
    for seed,prefix in [(42,'updates_v1'),(43,'updates_v1r43'),(44,'updates_v1r44')]:
        for step in [50,150,300]:
            corpus=f'{prefix}_qwen3_8b_step{step:06d}';base=ROOT/'artifacts'/f'stage1_{corpus}';data=Corpus(ROOT,corpus)
            if [(r['original_id'],r['split']) for r in data.rows]!=[(r['original_id'],r['split']) for r in source.rows]:issues.append(corpus+': question/split alignment differs')
            cfg=json.loads((base/'config.json').read_text());adapter=ROOT/cfg['adapter']
            if sha256(adapter)!=cfg['adapter_sha256']:issues.append(corpus+': adapter checksum differs')
            completion=json.loads((base/'completion.json').read_text());agreement=json.loads((ROOT/'results/stage2'/corpus/'agreement.json').read_text())
            if len(data.rows)!=7000 or not completion['full_dataset_complete'] or any(j['total']!=7000 for j in agreement['judges'].values()):issues.append(corpus+': extraction/judge coverage differs')
            test={data.rows[i]['sample_id']:int(data.y[i]) for i in data.indices('test')};coverage[corpus]=dict(generated=len(data.rows),valid=int((data.y>=0).sum()),test=len(test),test_positives=sum(test.values()))
            for probe in POOLED+RAW:
                for probe_seed in [42,43,44]:
                    folder=ROOT/'results/model_updates_v1'/f'{corpus}__{probe}__s{probe_seed}__frozen'
                    if not (folder/'evaluation.json').exists():issues.append(folder.name+': missing frozen result');continue
                    paths=[folder]+[p for p in (ROOT/'artifacts/probes').glob(f'{corpus}__{probe}__s{probe_seed}__*') if (p/'evaluation.json').exists()]
                    for path in paths:
                        predictions=json.loads((path/'predictions.json').read_text())
                        if len(predictions)!=len(test) or {r['sample_id']:r['label'] for r in predictions}!=test:issues.append(path.name+': test alignment differs')
                        report=json.loads((path/'evaluation.json').read_text())
                        if (path/'complete.json').exists():
                            training=json.loads((path/'config.json').read_text());tr=set(training['train_ids']);va=set(training['validation_ids'])
                            if tr&va or (tr|va)&set(test):issues.append(path.name+': training/validation/test overlap')
                        rows.append(dict(trajectory_seed=seed,probe_seed=probe_seed,step=step,probe=probe,experiment=path.name,mode=report['mode'],**report['metrics']))
        path=ROOT/'results/model_updates_v1/representation_drift.json' if seed==42 else ROOT/'results/trajectory_replication_v1'/f's{seed}/representation_drift.json'
        if not path.exists():issues.append(f's{seed}: missing fixed-text drift')
        else:
            report=json.loads(path.read_text())
            for r in report['checkpoints']:drift.append(dict(trajectory_seed=seed,step=r['step'],mean_linear_cka=r['mean_linear_cka']))
    summaries=[]
    for step in [50,150,300]:
        for probe in POOLED+RAW:
            per_trajectory=[]
            for seed in [42,43,44]:
                cells=[r for r in rows if r['trajectory_seed']==seed and r['step']==step and r['probe']==probe and r['mode']=='zero_shot_parameter_transfer']
                if len(cells)!=3:issues.append(f'{seed}/{step}/{probe}: expected three probe seeds');continue
                per_trajectory.append(dict(trajectory_seed=seed,**{m:statistics.mean(r[m] for r in cells) for m in ['auroc','average_precision','f1_05','ece','brier']}))
            if len(per_trajectory)==3:
                summaries.append(dict(step=step,probe=probe,per_trajectory=per_trajectory,across_trajectory={m:dict(mean=statistics.mean(r[m] for r in per_trajectory),std=statistics.stdev(r[m] for r in per_trajectory)) for m in ['auroc','average_precision','f1_05','ece','brier']}))
    out=ROOT/'results/trajectory_replication_v1';out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'summary.json',dict(status='pass' if not issues else 'fail',issues=issues,coverage=coverage,rows=rows,frozen_summary=summaries,representation_drift=drift,caveat='Three optimizer sampling/initialization seeds on the same fixed HotpotQA adaptation examples, one parent and one domain; no independent domain or data-sampling replication. First average probe-training seeds within each trajectory, then report variability across three generator trajectories. Do not count nine probe/trajectory combinations as nine independent generator trajectories. Judge labels are not human factuality validation. Recovery budgets100/500/full are replicated for pooled architectures; Raw recovery100/500/full is available on trajectory42;500/full is available on trajectories43/44.'))
    with (out/'metrics.csv.tmp').open('w',newline='') as f:
        fields=['trajectory_seed','probe_seed','step','probe','experiment','mode','n','positives','auroc','average_precision','f1_05','ece','brier'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
    (out/'metrics.csv.tmp').replace(out/'metrics.csv')
    print(json.dumps(dict(status='pass' if not issues else 'fail',evaluations=len(rows),issues=issues)),flush=True)
    if issues:raise SystemExit(1)
if __name__=='__main__':main()
