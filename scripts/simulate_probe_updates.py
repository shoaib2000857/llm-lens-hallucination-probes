#!/usr/bin/env python3
"""Offline deployed-monitor policies; decisions use development labels or replay drift only.

Recovery checkpoints are existing source-warm-start fits, not sequentially warm-started
fits. Costs describe this explicit recovery rule. No policy is chosen by test scores.
"""
import argparse, csv, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
from portability.storage import atomic_json, sha256
from portability.probe_training import Corpus, make_probe, predict, POOLED
from reproduce_saved_scores import metrics
ROOT=Path(__file__).resolve().parents[1]
POLICIES=('never','platt_only','every_checkpoint','every_2_checkpoints','validation_drop','representation_drift')

def linear_cka(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    a=a-a.mean(0);b=b-b.mean(0);ka=a@a.T;kb=b@b.T
    denom=np.linalg.norm(ka)*np.linalg.norm(kb)
    return float(np.sum(ka*kb)/max(denom,1e-12))

def should_update(policy,ordinal,auc,reference_auc,drift,args):
    if policy=='every_checkpoint': return True
    if policy=='every_2_checkpoints': return ordinal%2==0
    if policy=='validation_drop': return auc is not None and reference_auc is not None and reference_auc-auc>args.auc_drop
    if policy=='representation_drift': return drift>args.drift_threshold
    return False

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--probes',nargs='+',choices=POOLED,default=list(POOLED));ap.add_argument('--seeds',type=int,nargs='+',default=[42,43,44]);ap.add_argument('--recovery-budget',default='500');ap.add_argument('--monitor-samples',type=int,default=256);ap.add_argument('--calibration-labels',type=int,default=100);ap.add_argument('--auc-drop',type=float,default=.05);ap.add_argument('--drift-threshold',type=float,default=.05);ap.add_argument('--device',default='cuda');ap.add_argument('--threads',type=int,default=8);ap.add_argument('--trajectory-prefix',default='updates_v1');ap.add_argument('--replay',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--cache-root',type=Path);args=ap.parse_args()
    # Optional separate trajectories/outputs preserve the original policy run.
    torch.set_num_threads(args.threads);root=args.root;device=torch.device(args.device);steps=[50,150,300]
    data={0:Corpus(root,'qwen3_8b')};data.update({s:Corpus(root,f'{args.trajectory_prefix}_qwen3_8b_step{s:06d}') for s in steps})
    common=set.intersection(*[{d.rows[i]['original_id'] for i in d.indices('dev')} for d in data.values()])
    monitor=list(np.random.default_rng(42).permutation(sorted(common)))[:args.monitor_samples]
    if len(monitor)!=args.monitor_samples: raise ValueError('Insufficient shared valid development questions')
    dev_idx={s:np.asarray([{r['original_id']:i for i,r in enumerate(d.rows)}[oid] for oid in monitor]) for s,d in data.items()}
    out=args.output or root/'results/model_update_schedules_v1';out.mkdir(parents=True,exist_ok=True)
    protocol=dict(steps=steps,probes=args.probes,seeds=args.seeds,recovery_budget=args.recovery_budget,monitor_original_ids=monitor,calibration_labels=args.calibration_labels,auc_drop=args.auc_drop,drift_threshold=args.drift_threshold,device=args.device,protocol='source_warm_recovery_policies_v1',feature_manifests={str(s):sha256(root/'artifacts'/f'stage1_{d.name}/pooled/manifest.json') for s,d in data.items()})
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=protocol:raise ValueError('Schedule protocol changed; use a separate root for different settings')
    atomic_json(out/'config.json',protocol)
    replay=args.replay or root/'artifacts/model_updates_v1/replay'
    for s in [0,*steps]:
        meta=json.loads((replay/f'step_{s:06d}.json').read_text()) if (replay/f'step_{s:06d}.json').exists() else None
        if meta is not None and sha256(replay/f'step_{s:06d}.npy')!=meta['sha256']:raise ValueError('Replay checksum mismatch')
    arrays={s:np.load(replay/f'step_{s:06d}.npy',mmap_mode='r') for s in [0,*steps]}
    kernels={}
    for s,x in arrays.items():
        ks=[]
        for l in range(x.shape[1]):
            a=np.asarray(x[:,l],dtype=np.float64); a=a-a.mean(0); ks.append(a@a.T)
        kernels[s]=np.stack(ks)
    drift={}
    for a in [0,*steps]:
        for b in steps:
            if a<b:drift[a,b]=1-float(np.mean([float(np.sum(ka*kb)/max(np.linalg.norm(ka)*np.linalg.norm(kb),1e-12)) for ka,kb in zip(kernels[a],kernels[b])]))
    results=[]
    for probe in args.probes:
        for seed in args.seeds:
            checkpoints={0:root/'artifacts/probes'/f'qwen3_8b__{probe}__s{seed}__scratch_full'}
            checkpoints.update({s:root/'artifacts/probes'/f'{data[s].name}__{probe}__s{seed}__warm_{args.recovery_budget}' for s in steps})
            predictions={};costs={};label_ids={};weights={}
            for version,folder in checkpoints.items():
                if not (folder/'best.pt').exists(): raise ValueError(f'Missing recovery checkpoint {folder}')
                weights[version]=sha256(folder/'best.pt')
                if version:
                    cfg=json.loads((folder/'config.json').read_text());label_ids[version]=set(cfg['train_ids']+cfg['validation_ids']);costs[version]=json.loads((folder/'complete.json').read_text())['seconds_this_invocation']
                    if len(label_ids[version])!=int(args.recovery_budget):raise ValueError('Recovery label budget mismatch')
                for target_step in [0,*steps]:
                    if version>target_step:continue
                    target=data[target_step];cache=(args.cache_root or out/'cache')/f'{probe}__s{seed}__v{version:06d}__t{target_step:06d}.json'
                    cache.parent.mkdir(exist_ok=True)
                    prov=dict(checkpoint_sha256=weights[version],manifest_sha256=protocol['feature_manifests'][str(target_step)],monitor_original_ids=monitor,device=args.device)
                    if cache.exists():
                        saved=json.loads(cache.read_text())
                        if saved['provenance']!=prov:raise ValueError('Schedule prediction cache provenance mismatch')
                    else:
                        ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);net=make_probe(probe,ck['shape']).to(device);net.load_state_dict(ck['model']);tr=np.random.default_rng(seed).permutation(target.indices('train'))[:args.calibration_labels];te=target.indices('test')
                        saved=dict(provenance=prov,dev=predict(net,target,dev_idx[target_step],device,64).tolist(),test=predict(net,target,te,device,64).tolist(),train=predict(net,target,tr,device,64).tolist(),train_indices=tr.tolist(),test_ids=[target.rows[i]['sample_id'] for i in te])
                        atomic_json(cache,saved);del net
                        if device.type=='cuda':torch.cuda.empty_cache()
                    if saved['test_ids']!=[target.rows[i]['sample_id'] for i in target.indices('test')]:raise ValueError('Schedule test ID mismatch')
                    predictions[version,target_step]=saved
            for policy in POLICIES:
                version=0;reference_auc=metrics(data[0].y[dev_idx[0]],predictions[0,0]['dev'])['auroc'];used=set();updates=0;cost=0.;reports=[]
                for ordinal,s in enumerate(steps,1):
                    target=data[s];before=predictions[version,s];monitor_auc=metrics(target.y[dev_idx[s]],before['dev'])['auroc'];distance=drift[version,s]
                    if policy=='validation_drop':used.update((s,target.rows[i]['sample_id']) for i in dev_idx[s])
                    update=should_update(policy,ordinal,monitor_auc,reference_auc,distance,args)
                    if update:
                        version=s;updates+=1;used.update((s,i) for i in label_ids[s]);cost+=costs[s];reference_auc=metrics(target.y[dev_idx[s]],predictions[s,s]['dev'])['auroc']
                    saved=predictions[version,s];p=np.asarray(saved['test']);calibration_status=None
                    if policy=='platt_only':
                        idx=np.asarray(saved['train_indices']);y=target.y[idx];used.update((s,target.rows[i]['sample_id']) for i in idx)
                        if len(np.unique(y))==2:
                            z=logit(np.clip(saved['train'],1e-6,1-1e-6));zt=logit(np.clip(p,1e-6,1-1e-6));fit=LogisticRegression(random_state=seed).fit(z[:,None],y);p=fit.predict_proba(zt[:,None])[:,1];calibration_status='fit'
                        else:calibration_status='single_class_no_change'
                    report=dict(step=s,detector_version=version,updated=update,decision_development_auroc=monitor_auc,decision_replay_drift=distance,calibration_status=calibration_status,metrics=metrics(target.y[target.indices('test')],p))
                    reports.append(report)
                aggregate={}
                for metric in ['auroc','average_precision','f1_05','ece','brier']:
                    vals=[r['metrics'][metric] for r in reports if r['metrics'][metric] is not None];aggregate[metric]=dict(mean=float(np.mean(vals)),worst=float(max(vals) if metric in ['ece','brier'] else min(vals))) if vals else None
                result=dict(probe=probe,seed=seed,policy=policy,update_count=updates,total_target_labels=len(used),recovery_training_seconds_last_invocations=cost,aggregate=aggregate,checkpoints=reports)
                atomic_json(out/f'{probe}__s{seed}__{policy}.json',result);results.append(result)
    atomic_json(out/'summary.json',dict(results=results,policies=list(POLICIES),caveat='Offline simulation with three observed updates, not a 100-update deployment. Recovery uses existing independently source-warm-started 500-label fits, not sequential initialization. Thresholds are illustrative configuration choices, not validated universal thresholds. Target-label cost counts checkpoint-specific outcomes; initial deployment cost is excluded. Replay inference and monitoring compute are not included in recovery training seconds, which count only recorded last invocations. No policy selection uses test labels.'))
    with (out/'metrics.csv.tmp').open('w',newline='') as f:
        fields=['probe','seed','policy','update_count','total_target_labels','recovery_training_seconds_last_invocations','step','detector_version','updated','auroc','average_precision','f1_05','ece','brier'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader()
        for r in results:
            for c in r['checkpoints']:w.writerow({**r,**c,**c['metrics']})
    (out/'metrics.csv.tmp').replace(out/'metrics.csv')
    print(f'Completed {len(results)} policy simulations. No generator inference or new hidden tensors.',flush=True)
if __name__=='__main__':main()
