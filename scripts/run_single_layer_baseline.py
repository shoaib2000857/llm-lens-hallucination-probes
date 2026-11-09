#!/usr/bin/env python3
"""Fit independent linear heads jointly; select layer/epoch using dev AUROC only."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'));sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,torch
from sklearn.metrics import roc_auc_score
from portability.probe_training import Corpus,atomic_torch
from portability.storage import atomic_json
from reproduce_saved_scores import metrics

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--device',default='cuda');ap.add_argument('--epochs',type=int,default=25);ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);args=ap.parse_args()
 torch.set_num_threads(8);device=torch.device(args.device)
 for corpus in ['ministral3_8b','qwen3_8b']:
  data=Corpus(args.root,corpus);tr=data.indices('train');va=data.indices('dev');te=data.indices('test');l,_,d=data.shape
  for seed in args.seeds:
   out=args.root/'artifacts/probes'/f'{corpus}__best_layer_linear__s{seed}__scratch_full';out.mkdir(parents=True,exist_ok=True)
   if (out/'evaluation.json').exists():continue
   torch.manual_seed(seed);w=torch.nn.Parameter(torch.randn(l,d,device=device)*.01);b=torch.nn.Parameter(torch.zeros(l,device=device));opt=torch.optim.AdamW([w,b],lr=2e-4)
   start=0;best=-1.;stale=0
   if (out/'last.pt').exists():
    ck=torch.load(out/'last.pt',map_location=device,weights_only=True);w.data.copy_(ck['w']);b.data.copy_(ck['b']);opt.load_state_dict(ck['optimizer']);start=ck['epoch']+1;best=ck['best'];stale=ck['stale']
   def probabilities(idx):
    values=[]
    with torch.no_grad():
     for s in range(0,len(idx),64):
      x,_=data.batch(idx[s:s+64],device);values.append(torch.sigmoid((x*w).sum(-1)+b).cpu().numpy())
    return np.concatenate(values)
   positives=float(data.y[tr].sum());pw=torch.tensor((len(tr)-positives)/positives,device=device)
   for epoch in range(start,args.epochs):
    if stale>=4:break
    order=np.random.default_rng(seed+epoch).permutation(tr)
    for s in range(0,len(order),64):
     x,y=data.batch(order[s:s+64],device);opt.zero_grad(set_to_none=True);z=(x*w).sum(-1)+b
     loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y[:,None].expand_as(z),pos_weight=pw);loss.backward();opt.step()
    p=probabilities(va);scores=[roc_auc_score(data.y[va],p[:,i]) for i in range(l)];layer=int(np.argmax(scores));score=float(scores[layer])
    if score>best:
     best=score;stale=0;atomic_torch(out/'best.pt',dict(w=w.detach().cpu(),b=b.detach().cpu(),layer=layer,epoch=epoch))
    else:stale+=1
    atomic_torch(out/'last.pt',dict(w=w.detach().cpu(),b=b.detach().cpu(),optimizer=opt.state_dict(),epoch=epoch,best=best,stale=stale))
   ck=torch.load(out/'best.pt',map_location=device,weights_only=True);w.data.copy_(ck['w']);b.data.copy_(ck['b']);p=probabilities(te)[:,ck['layer']]
   atomic_json(out/'config.json',dict(total_target_labels=len(tr)+len(va),seed=seed,selected_layer=int(ck['layer']),deployed_parameters=d+1,trained_parameters=l*(d+1),selection='dev AUROC; test never used',epochs=args.epochs))
   atomic_json(out/'predictions.json',[dict(sample_id=data.rows[i]['sample_id'],label=int(data.y[i]),probability=float(v)) for i,v in zip(te,p)])
   atomic_json(out/'evaluation.json',dict(mode='target_trained',metrics=metrics(data.y[te],p)))
   print(out.name,flush=True)
if __name__=='__main__':main()
