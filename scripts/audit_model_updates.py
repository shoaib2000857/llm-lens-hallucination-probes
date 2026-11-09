#!/usr/bin/env python3
"""Validate completed trajectory coverage, exact label alignment and split separation."""
import argparse, collections, json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from portability.storage import atomic_json, sha256
from portability.probe_training import Corpus, POOLED, RAW
ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, default=ROOT); args = ap.parse_args()
    root = args.root; issues = []; counts = {}; groups = collections.defaultdict(list)
    source = Corpus(root, 'qwen3_8b')
    for step in [50, 150, 300]:
        name = f'updates_v1_qwen3_8b_step{step:06d}'; target = Corpus(root, name)
        base = root/'artifacts'/f'stage1_{name}'
        if [r['original_id'] for r in target.rows] != [r['original_id'] for r in source.rows]: issues.append(name + ': original question IDs changed')
        if [r['split'] for r in target.rows] != [r['split'] for r in source.rows]: issues.append(name + ': original splits changed')
        cfg = json.loads((base/'config.json').read_text())
        if sha256(Path(cfg['adapter'])) != cfg['adapter_sha256']: issues.append(name + ': adapter checksum changed')
        done = json.loads((base/'completion.json').read_text())
        if len(target.rows) != 7000 or not done['full_dataset_complete']: issues.append(name + ': extraction incomplete')
        agreement = json.loads((root/'results/stage2'/name/'agreement.json').read_text())
        if any(j['total'] != 7000 for j in agreement['judges'].values()): issues.append(name + ': judge coverage incomplete')
        test = {target.rows[i]['sample_id']: int(target.y[i]) for i in target.indices('test')}
        counts[name] = dict(generated=len(target.rows), valid=int((target.y >= 0).sum()), test=len(test), test_positives=sum(test.values()))
        paths = list((root/'results/model_updates_v1').glob(name+'__*/evaluation.json')) + list((root/'artifacts/probes').glob(name+'__*/evaluation.json'))
        for path in paths:
            pred = json.loads((path.parent/'predictions.json').read_text()); report = json.loads(path.read_text())
            if len(pred) != len(test) or {r['sample_id']:r['label'] for r in pred} != test or len({r['sample_id'] for r in pred}) != len(pred): issues.append(path.parent.name+': prediction ID/label mismatch')
            if report['metrics']['n'] != len(pred): issues.append(path.parent.name+': metric count mismatch')
            if (path.parent/'complete.json').exists():
                train_cfg = json.loads((path.parent/'config.json').read_text()); tr=set(train_cfg['train_ids']); va=set(train_cfg['validation_ids']); te=set(test)
                if tr & va or tr & te or va & te: issues.append(path.parent.name+': split overlap')
                if not tr <= {target.rows[i]['sample_id'] for i in target.indices('train')}: issues.append(path.parent.name+': training outside train split')
                expected_labels={r['sample_id']:r['label'] for r in target.rows}
                if any(expected_labels.get(i) not in (0,1) for i in tr|va): issues.append(path.parent.name+': invalid training/development label')
            parts=path.parent.name.split('__'); groups[(step,parts[1],report['mode'])].append(report['metrics'])
        for probe in POOLED+RAW:
            if len(groups[(step,probe,'zero_shot_parameter_transfer')]) != 3: issues.append(f'{name}/{probe}: missing frozen seeds')
    summary=[]
    for (step,probe,mode), rows in sorted(groups.items()):
        values={}
        for k in ['auroc','average_precision','f1_05','ece','brier','mean_probability']:
            vals=[r[k] for r in rows if r.get(k) is not None]
            values[k]=dict(mean=statistics.mean(vals),std=statistics.stdev(vals) if len(vals)>1 else 0) if vals else None
        summary.append(dict(step=step,probe=probe,mode=mode,runs=len(rows),metrics=values))
    count=sum(len(v) for v in groups.values())
    report=dict(status='pass' if not issues else 'fail',issues=issues,corpora=counts,evaluations=count,seed_summary=summary,
                caveat='Recovery summary pools label budgets; use individual configs for budget-matched comparisons. Judge labels are not human factuality labels.')
    atomic_json(root/'results/model_updates_v1/audit_summary.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['seed_summary']},indent=2))
    if issues: raise SystemExit(1)
if __name__=='__main__': main()
