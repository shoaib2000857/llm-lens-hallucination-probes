#!/usr/bin/env python3
"""Five-fold target-trained and same-generator warm-start controls, grouped by context."""
import argparse,hashlib,json,sys,types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold,GroupShuffleSplit
from portability.storage import ShardDataset,atomic_json
from portability.probe_training import Corpus,train,predict,POOLED,RAW
from reproduce_saved_scores import metrics,bootstrap
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--corpora',nargs='+',required=True);ap.add_argument('--probes',nargs='+',default=list(POOLED));ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);ap.add_argument('--epochs',type=int,default=25);ap.add_argument('--bootstrap',type=int,default=500);ap.add_argument('--adaptations',nargs='+',choices=['scratch','warm'],default=['scratch','warm']);ap.add_argument('--raw-batch',type=int,default=8);ap.add_argument('--device',default='cuda');args=ap.parse_args()
    torch.set_num_threads(8);device=torch.device(args.device)
    for corpus in args.corpora:
        generator=next(g for g in ['ministral3_8b','qwen3_8b'] if corpus.endswith('_'+g))
        data=Corpus(ROOT,corpus);valid=data.indices('test')
        meta=ShardDataset(ROOT/'artifacts'/f'stage1_{corpus}'/'hidden_states').records
        if [r['sample_id'] for r in meta]!=[r['sample_id'] for r in data.rows]:raise ValueError('Context metadata order differs')
        groups=np.asarray([hashlib.sha256(meta[i]['context'].encode()).hexdigest() for i in valid])
        y=data.y[valid]
        if min(np.bincount(y.astype(int),minlength=2))<5:raise ValueError('Five-fold control needs at least five examples of each class')
        for seed in args.seeds:
            folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(valid,y,groups))
            for name in args.probes:
                if name not in POOLED+RAW:raise ValueError('Unknown probe')
                data=Corpus(ROOT,corpus,raw=name in RAW)
                batch=args.raw_batch if name in RAW else 64
                for adaptation in args.adaptations:
                    predictions=[];fold_manifest=[]
                    for fold,(fit_local,test_local) in enumerate(folds):
                        tr_local,va_local=next(GroupShuffleSplit(1,test_size=.25,random_state=seed+fold).split(fit_local,groups=groups[fit_local]))
                        mapping={'train':valid[fit_local[tr_local]],'dev':valid[fit_local[va_local]],'test':valid[test_local]}
                        group_sets=[set(groups[np.isin(valid,mapping[s])]) for s in ['train','dev','test']]
                        if any(a&b for i,a in enumerate(group_sets) for b in group_sets[i+1:]):raise ValueError('Context leakage across folds')
                        data.indices=types.MethodType(lambda self,split,m=mapping:m[split],data)
                        out=ROOT/'artifacts/probes'/f'{corpus}__cv{fold}__{name}__s{seed}__{adaptation}_full'
                        source=ROOT/'artifacts/probes'/f'{generator}__{name}__s{seed}__scratch_full' if adaptation=='warm' else None
                        net=train(data,name,'full',seed,out,device,args.epochs,4,batch,source)
                        if net is None:raise ValueError('Single-class fold; inspect data, do not silently omit test examples')
                        p=predict(net,data,mapping['test'],device,batch)
                        rows=[dict(sample_id=data.rows[i]['sample_id'],label=int(data.y[i]),probability=float(v),fold=fold) for i,v in zip(mapping['test'],p)]
                        atomic_json(out/'oof_predictions.json',rows);predictions.extend(rows)
                        fold_manifest.append(dict(fold=fold,counts={s:len(v) for s,v in mapping.items()},checkpoint=str(out.relative_to(ROOT))))
                        del net;torch.cuda.empty_cache()
                    if len(predictions)!=len(valid) or len({r['sample_id'] for r in predictions})!=len(valid):raise ValueError('OOF coverage mismatch')
                    predictions.sort(key=lambda r:r['sample_id']);yt=np.asarray([r['label'] for r in predictions]);pt=np.asarray([r['probability'] for r in predictions])
                    out=ROOT/'results/distribution_v1'/f'{corpus}__{name}__s{seed}__cv_{adaptation}'
                    out.mkdir(parents=True,exist_ok=True)
                    atomic_json(out/'predictions.json',predictions)
                    report=dict(mode='target_trained_oof' if adaptation=='scratch' else ('same_family_warm_start_oof' if corpus.startswith('samefamily_') else 'same_generator_domain_warm_start_oof'),metrics=metrics(yt,pt),folds=fold_manifest,
                                confidence_intervals=bootstrap(yt,pt,args.bootstrap,seed),caveat='Five-fold target controls: each example predicted by a probe that never trains or selects on its context. Approximately 600 training + 200 development labels per fold. These are OOF estimates, not an independent final held-out test. Source warm checkpoint previously trained on SQuAD; target label budgets matched to scratch.')
                    report['metrics']['accuracy_05']=float(np.mean((pt>=.5)==yt))
                    report['unique_target_labels_across_cv']=len(valid)
                    atomic_json(out/'evaluation.json',report);print(json.dumps(dict(experiment=out.name,metrics=report['metrics'])),flush=True)
if __name__=='__main__':main()
