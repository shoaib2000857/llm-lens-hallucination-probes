#!/usr/bin/env python3
"""Check completed result counts, IDs and context separation; export seed means."""
import collections,json,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1];issues=[];counts={};summary=[]
for base in sorted((root/'artifacts').glob('stage1_distribution_v1_*')):
    records=[r for p in sorted((base/'hidden_states').glob('shard_*.meta.json')) for r in json.loads(p.read_text())]
    if not records:
        records=[r for p in sorted((base/'hidden_states').glob('*.commit.json')) for r in json.loads((base/'hidden_states'/json.loads(p.read_text())['metadata']).read_text())]
    ids={r['sample_id'] for r in records};context={r['sample_id']:r['context'] for r in records}
    complete=json.loads((base/'completion.json').read_text());agreement=json.loads((root/'results/stage2'/base.name.removeprefix('stage1_')/'agreement.json').read_text())
    if len(ids)!=1000 or len(records)!=1000 or not complete['full_dataset_complete']:issues.append(base.name+': incomplete extraction')
    if any(j['total']!=1000 for j in agreement['judges'].values()):issues.append(base.name+': incomplete judges')
    counts[base.name]=dict(generated=len(records),valid=agreement['complete_three_judge'],positives=agreement['consensus_positive'],prompt_truncated=sum(r.get('prompt_was_truncated',False) for r in records))
    corpus=base.name.removeprefix('stage1_')
    for path in (root/'artifacts/probes').glob(corpus+'__cv*/config.json'):
        cfg=json.loads(path.read_text());tr=set(cfg['train_ids']);va=set(cfg['validation_ids'])
        predictions=json.loads((path.parent/'oof_predictions.json').read_text());te={r['sample_id'] for r in predictions}
        if tr&va or tr&te or va&te:issues.append(path.parent.name+': ID overlap')
        gs=[{context[i] for i in s} for s in [tr,va,te]]
        if any(a&b for i,a in enumerate(gs) for b in gs[i+1:]):issues.append(path.parent.name+': context overlap')
        if not (path.parent/'complete.json').exists():issues.append(path.parent.name+': unfinished fit')
groups=collections.defaultdict(list)
for path in (root/'results/distribution_v1').glob('*/evaluation.json'):
    r=json.loads(path.read_text());name=path.parent.name.split('__');groups[(name[0],name[1],r['mode'])].append(r['metrics'])
    rows=json.loads((path.parent/'predictions.json').read_text())
    if len({x['sample_id'] for x in rows})!=len(rows) or len(rows)!=r['metrics']['n']:issues.append(path.parent.name+': predictions mismatch')
for (corpus,probe,mode),rows in sorted(groups.items()):
    if len(rows)!=3:issues.append(f'{corpus}/{probe}/{mode}: missing seeds')
    fields={}
    for k in ['auroc','average_precision','f1_05','accuracy_05','ece','brier']:
        vals=[r[k] for r in rows if r.get(k) is not None]
        fields[k]=dict(mean=statistics.mean(vals),std=statistics.stdev(vals) if len(vals)>1 else 0) if vals else None
    summary.append(dict(corpus=corpus,probe=probe,mode=mode,n=rows[0]['n'],positives=rows[0]['positives'],metrics=fields))
fits=sum(1 for p in (root/'artifacts/probes').glob('distribution_v1*__cv*/complete.json'))
configured_fits=sum(1 for p in (root/'artifacts/probes').glob('distribution_v1*__cv*/config.json'))
if fits!=configured_fits:issues.append(f'Configured/completed fits differ: {configured_fits}/{fits}')
if fits<600 or sum(map(len,groups.values()))<216:issues.append('Original minimum coverage of600 fits/216 evaluations is incomplete')
report=dict(status='pass' if not issues else 'fail',issues=issues,corpora=counts,completed_control_fits=fits,evaluations=sum(map(len,groups.values())),seed_summary=summary)
(root/'results/distribution_v1/audit_summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='seed_summary'},indent=2))
if issues:raise SystemExit(1)
