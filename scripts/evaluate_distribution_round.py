#!/usr/bin/env python3
"""Frozen same-generator distribution portability; no target fitting or inversion choice."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from portability.storage import atomic_json,sha256
from portability.probe_training import Corpus,make_probe,POOLED,RAW,registry
from run_probe_suite import evaluate
from reproduce_saved_scores import metrics
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--corpora',nargs='+',required=True);ap.add_argument('--probes',nargs='+',default=list(POOLED+RAW));ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);ap.add_argument('--bootstrap',type=int,default=500);ap.add_argument('--batch',type=int,default=64);ap.add_argument('--raw-batch',type=int,default=8);ap.add_argument('--device',default='cuda');args=ap.parse_args()
    torch.set_num_threads(8);device=torch.device(args.device)
    for corpus in args.corpora:
        generator=next((g for g in ['ministral3_8b','qwen3_8b'] if corpus.endswith('_'+g)),None)
        if generator is None:raise ValueError('Unknown source generator')
        cfg=json.loads((ROOT/'artifacts'/f'stage1_{corpus}'/'config.json').read_text())
        original=json.loads((ROOT/'configs'/f'stage1_{generator}.json').read_text())
        if (cfg['model'],cfg['model_revision'])!=(original['model'],original['model_revision']):raise ValueError('Distribution transfer requires identical generator checkpoint')
        src=Corpus(ROOT,generator);target=Corpus(ROOT,corpus);tr=src.indices('train');te=target.indices('test');signals={}
        for signal in ['mean_token_logprob','min_token_logprob','mean_token_entropy','n_gen_tokens']:
            x=np.asarray([src.rows[i][signal] for i in tr]);xt=np.asarray([target.rows[i][signal] for i in te])
            mu=x.mean();sd=max(x.std(),1e-8);fit=LogisticRegression(random_state=42).fit(((x-mu)/sd)[:,None],src.y[tr])
            p=fit.predict_proba(((xt-mu)/sd)[:,None])[:,1]
            signals[signal]=dict(metrics=metrics(target.y[te],p),source_training_labels=len(tr),target_training_labels=0)
        signal_out=ROOT/'results/distribution_v1'/f'{corpus}__confidence';signal_out.mkdir(parents=True,exist_ok=True)
        atomic_json(signal_out/'signals.json',dict(mode='source_fitted_confidence_transfer',source_corpus=generator,target_corpus=corpus,signals=signals))
        del src,target
        for name in args.probes:
            data=Corpus(ROOT,corpus,name in RAW)
            for seed in args.seeds:
                checkpoint=ROOT/'artifacts/probes'/f'{generator}__{name}__s{seed}__scratch_full'/'best.pt'
                digest=sha256(checkpoint)
                out=ROOT/'results/distribution_v1'/f'{corpus}__{name}__s{seed}'
                config=dict(corpus=corpus,probe=name,seed=seed,source_checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=digest,
                            target_feature_manifest_sha256=sha256(ROOT/'artifacts'/f'stage1_{corpus}'/'pooled/manifest.json'),bootstrap=args.bootstrap,
                            mode='zero_shot_distribution_transfer',target_training_labels=0)
                out.mkdir(parents=True,exist_ok=True)
                if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config:raise ValueError('Evaluation provenance changed')
                atomic_json(out/'config.json',config)
                if (out/'complete.json').exists():continue
                ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
                if list(ck['shape'])!=list(data.shape):raise ValueError('Same-generator representation shape changed')
                net=make_probe(name,ck['shape']).to(device);net.load_state_dict(ck['model'])
                registry(data,out,dict(source=str(checkpoint),probe=name,seed=seed,total_target_labels=0),'running')
                started=time.time();p=evaluate(net,data,device,args.raw_batch if name in RAW else args.batch,out,config['mode'],seed,args.bootstrap)
                report=json.loads((out/'evaluation.json').read_text());idx=data.indices('test')
                report['metrics']['accuracy_05']=float(np.mean((p>=.5)==data.y[idx]))
                report['metrics']['majority_class_accuracy']=float(max(data.y[idx].mean(),1-data.y[idx].mean()))
                report['diagnostic_note']='Inverted AUROC uses target labels; it is not a deployment score. Labels are judge consensus, not human factuality.'
                atomic_json(out/'evaluation.json',report)
                atomic_json(out/'complete.json',dict(seconds=time.time()-started,config=config))
                registry(data,out,dict(source=str(checkpoint),probe=name,seed=seed,total_target_labels=0),'complete',time.time()-started)
                print(json.dumps(dict(experiment=out.name,metrics=report['metrics'])),flush=True)
                del net,data;torch.cuda.empty_cache()
                data=Corpus(ROOT,corpus,name in RAW)
if __name__=='__main__':main()
