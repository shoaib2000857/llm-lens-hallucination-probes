#!/usr/bin/env python3
"""Matched scratch/warm-start, frozen transfer, calibration and optional full-token probes."""
import argparse,csv,json,os,sys,time,platform,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score,roc_auc_score
from scipy.optimize import minimize_scalar
from scipy.special import expit,logit
from portability.storage import atomic_json
from portability.probe_training import Corpus,train,predict,make_probe,POOLED,RAW
from reproduce_saved_scores import metrics,bootstrap

def evaluate(net,corpus,device,batch,out,mode,seed,bootstrap_n=0,source_layers=None):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 idx=corpus.indices('test');p=predict(net,corpus,idx,device,batch,source_layers)
 rows=[dict(sample_id=corpus.rows[i]['sample_id'],label=int(corpus.y[i]),probability=float(v)) for i,v in zip(idx,p)]
 atomic_json(out/'predictions.json',rows)
 report=dict(mode=mode,metrics=metrics(corpus.y[idx],p),per_judge={})
 if bootstrap_n:report['confidence_intervals']=bootstrap(corpus.y[idx],p,bootstrap_n,seed)
 for judge in ['mistral','qwen','phi']:
  y=[corpus.rows[i]['judge_labels'].get(judge) for i in idx]
  valid=[j for j,v in enumerate(y) if v in [0,1]]
  if valid:report['per_judge'][judge]=metrics(np.array(y)[valid].astype(int),p[valid])
 atomic_json(out/'evaluation.json',report)
 return p

def calibrate(net,target,device,batch,out,budgets,seed,source_layers=None,ptest=None):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 tr=np.random.default_rng(seed).permutation(target.indices('train'));te=target.indices('test')
 # This is a calibration experiment: only selected target training labels are used.
 ptest=predict(net,target,te,device,batch,source_layers) if ptest is None else ptest;ptr=predict(net,target,tr[:max(budgets)],device,batch,source_layers)
 reports=[]
 for budget in budgets:
  if budget>len(tr):continue
  y=target.y[tr[:budget]];p=ptr[:budget];z=logit(np.clip(p,1e-6,1-1e-6));zt=logit(np.clip(ptest,1e-6,1-1e-6))
  if len(np.unique(y))<2:reports.append(dict(budget=budget,status='single_class_calibration'));continue
  threshold=max(np.unique(np.r_[0,p,1]),key=lambda v:f1_score(y,p>=v,zero_division=0))
  platt=LogisticRegression(C=1.0,random_state=seed).fit(z[:,None],y)
  temp=minimize_scalar(lambda t:float(np.mean(np.logaddexp(0,z/t)-y*z/t)),bounds=(.05,20),method='bounded').x
  iso=IsotonicRegression(out_of_bounds='clip').fit(p,y)
  flip=roc_auc_score(y,p)<.5
  methods={'threshold':ptest,'platt':platt.predict_proba(zt[:,None])[:,1],'temperature':expit(zt/temp),'isotonic':iso.predict(ptest),'label_selected_sign':1-ptest if flip else ptest}
  for method,pred in methods.items():
   m=metrics(target.y[te],pred)
   if method=='threshold':m['f1_calibrated_threshold']=float(f1_score(target.y[te],pred>=threshold,zero_division=0))
   reports.append(dict(budget=budget,method=method,mode='calibrated_transfer',metrics=m,threshold=float(threshold) if method=='threshold' else None,selected_inversion=bool(flip) if method=='label_selected_sign' else None))
   atomic_json(out/f'{method}_{budget}_predictions.json',[dict(sample_id=target.rows[i]['sample_id'],label=int(target.y[i]),probability=float(v)) for i,v in zip(te,pred)])
 atomic_json(out/'calibration.json',reports)

def confidence_baselines(corpus,out):
 tr=corpus.indices('train');te=corpus.indices('test');result={}
 for name in ['mean_token_logprob','min_token_logprob','mean_token_entropy','n_gen_tokens']:
  x=np.array([corpus.rows[i][name] for i in tr],dtype=float);xt=np.array([corpus.rows[i][name] for i in te],dtype=float)
  mean=x.mean();std=max(x.std(),1e-8)
  fit=LogisticRegression().fit(((x-mean)/std)[:,None],corpus.y[tr]);p=fit.predict_proba(((xt-mean)/std)[:,None])[:,1]
  result[name]=dict(metrics=metrics(corpus.y[te],p),training_labels=len(tr),method='training-set-fitted univariate logistic baseline')
 atomic_json(out/'confidence_baselines.json',result)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--probes',nargs='+',default=list(POOLED));ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);ap.add_argument('--budgets',nargs='+',default=['25','50','100','250','500','1000','full']);ap.add_argument('--epochs',type=int,default=25);ap.add_argument('--patience',type=int,default=4);ap.add_argument('--threads',type=int,default=8);ap.add_argument('--batch',type=int,default=64);ap.add_argument('--raw-batch',type=int,default=2);ap.add_argument('--bootstrap',type=int,default=500);ap.add_argument('--device',default='cuda');ap.add_argument('--skip-calibration',action='store_true');args=ap.parse_args()
 torch.set_num_threads(args.threads);device=torch.device(args.device)
 base=args.root/'artifacts/probes';base.mkdir(parents=True,exist_ok=True)
 atomic_json(base/f'invocation_{time.time_ns()}.json',dict(command=sys.argv,python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,threads=args.threads,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip()))
 for name in args.probes:
  if name not in (*POOLED,*RAW):raise ValueError('Unknown probe')
  raw=name in RAW;batch=args.raw_batch if raw else args.batch
  corpora={c:Corpus(args.root,c,raw) for c in ['ministral3_8b','qwen3_8b']}
  for seed in args.seeds:
   trained={}
   for c,data in corpora.items():
    folder=base/f'{c}__{name}__s{seed}__scratch_full';net=train(data,name,'full',seed,folder,device,args.epochs,args.patience,batch)
    if net is None:raise ValueError('Full-data training has one class')
    evaluate(net,data,device,batch,folder,'target_trained',seed,args.bootstrap)
    if seed==args.seeds[0] and name==args.probes[0]:confidence_baselines(data,folder)
    trained[c]=folder;del net;torch.cuda.empty_cache() if device.type=='cuda' else None
   for source,target in [('ministral3_8b','qwen3_8b'),('qwen3_8b','ministral3_8b')]:
    src,tgt=corpora[source],corpora[target]
    ck=torch.load(trained[source]/'best.pt',map_location='cpu',weights_only=True);net=make_probe(name,ck['shape']).to(device);net.load_state_dict(ck['model'])
    # M2/M3 have layer positional buffers. Never silently reshape 34 and 36 layers.
    mapped=src.shape[0] if name in ('M2','M3') and src.shape[0]!=tgt.shape[0] else None
    mode='depth_resampled_transfer' if mapped else 'zero_shot_transfer'
    out=base/f'{source}_to_{target}__{name}__s{seed}__{mode}'
    frozen_pred=evaluate(net,tgt,device,batch,out,mode,seed,args.bootstrap,mapped)
    if not args.skip_calibration:calibrate(net,tgt,device,batch,out/'calibration',[10,25,50,100,250,500],seed,mapped,ptest=frozen_pred)
    del net;torch.cuda.empty_cache() if device.type=='cuda' else None
    for budget in args.budgets:
     if budget!='full' and int(budget)>len(tgt.indices('train')):continue
     for adaptation in ['scratch','warm']:
      if adaptation=='scratch' and budget=='full':continue
      folder=base/f'{target}__{name}__s{seed}__{adaptation}_{budget}'
      net=train(tgt,name,budget,seed,folder,device,args.epochs,args.patience,batch,trained[source] if adaptation=='warm' else None)
      if net is not None:evaluate(net,tgt,device,batch,folder,'warm_start' if adaptation=='warm' else 'target_trained',seed)
      del net;torch.cuda.empty_cache() if device.type=='cuda' else None
 print('Probe suite completed',flush=True)
if __name__=='__main__':main()
