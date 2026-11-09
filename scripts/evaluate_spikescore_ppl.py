#!/usr/bin/env python3
"""Compare saved SpikeScore against existing probes on exactly matched valid IDs."""
import json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from portability.probe_training import Corpus
from portability.storage import atomic_json
from reproduce_saved_scores import metrics

def main():
    out=ROOT/'results/spikescore_ppl_v1';reports=[]
    if not (out/'complete.json').exists():raise ValueError('Generation incomplete')
    for folder in sorted(out.iterdir()):
        if not folder.is_dir():continue
        data=Corpus(ROOT,folder.name);labels={r['sample_id']:r['label'] for r in data.rows}
        selected=json.loads((folder/'selection.json').read_text())['ids']
        rows=[r for file in sorted(folder.glob('batch_*.json')) for r in json.loads(file.read_text())]
        if [r['sample_id'] for r in rows]!=selected:raise ValueError('Subset ID alignment mismatch')
        valid=[r for r in rows if labels[r['sample_id']] is not None];y=[labels[r['sample_id']] for r in valid];scores=[r['spikescore'] for r in valid]
        report=dict(corpus=folder.name,selected=len(rows),valid=len(valid),positives=sum(y),raw_auroc=roc_auc_score(y,scores) if len(set(y))==2 else None,raw_average_precision=average_precision_score(y,scores) if len(set(y))==2 else None,turns_at_token_cap=sum(t['hit_cap'] for r in rows for t in r['turns']),truncated_prompts=sum(t['prompt_truncated'] for r in rows for t in r['turns']),matched_probes=[])
        if len(set(y))==2:
            ya=np.asarray(y);sa=np.asarray(scores);groups=[np.flatnonzero(ya==c) for c in [0,1]];rng=np.random.default_rng(42);values=[]
            for _ in range(1000):
                ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);values.append([roc_auc_score(ya[ix],sa[ix]),average_precision_score(ya[ix],sa[ix])])
            low,high=np.quantile(values,[.025,.975],axis=0)
            report['bootstrap_ci95']=dict(auroc=[float(low[0]),float(high[0])],average_precision=[float(low[1]),float(high[1])],replicates=1000,seed=42)
        base=ROOT/'artifacts/probes' if folder.name=='qwen3_8b' else ROOT/'results/distribution_v1'
        for file in sorted(base.glob(folder.name+'__*/predictions.json')):
            if '__cv_' in file.parent.name:continue
            # Judge-specific training runs store their own evaluation targets.
            # Main source comparison uses consensus-trained full-data probes.
            if folder.name=='qwen3_8b' and not file.parent.name.endswith('__scratch_full'):continue
            values={r['sample_id']:r for r in json.loads(file.read_text())}
            if not all(r['sample_id'] in values for r in valid):continue
            if any(values[r['sample_id']]['label']!=labels[r['sample_id']] for r in valid):raise ValueError('Probe label mismatch')
            report['matched_probes'].append(dict(experiment=file.parent.name,metrics=metrics(y,[values[r['sample_id']]['probability'] for r in valid])))
        reports.append(report)
    atomic_json(out/'evaluation.json',dict(results=reports,caveat='Label-blind200-case subset per corpus; original response consensus labels, not human factuality. Score is unbounded; AUROC/AP only, no invented probability calibration. Adapted PPL backbone and capped continuation protocol, not original paper reproduction.'))
    print(json.dumps(reports),flush=True)
if __name__=='__main__':main()
