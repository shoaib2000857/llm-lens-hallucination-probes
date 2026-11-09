#!/usr/bin/env python3
"""Small global MLP adapters on existing unlabeled identical-text anchors."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch
from portability.storage import atomic_json,sha256
from portability.probe_training import Corpus,make_probe
from reproduce_saved_scores import metrics,bootstrap

def main():
 torch.set_num_threads(4);out=ROOT/'results/mlp_alignment_v1';out.mkdir(exist_ok=True)
 print('RESOURCE PLAN:CPU-only4threads,<8GiB RAM,<1GiB outputs,18adapter fits,existing500same-text anchors per direction; no new LLM inference.',flush=True)
 for source,target in [('qwen3_8b','ministral3_8b'),('ministral3_8b','qwen3_8b')]:
  folder=ROOT/'results/fixed_text_alignment_v1'/f'{source}_to_{target}';cfg=json.loads((folder/'config.json').read_text());src=Corpus(ROOT,source);tgt=Corpus(ROOT,target);lookup={r['sample_id']:i for i,r in enumerate(src.rows)}
  x=np.load(folder/'target_anchor_features.npy');y=np.asarray(src.array[[lookup[s] for s in cfg['anchor_ids']]],dtype=np.float32).mean(1);te=tgt.indices('test');z=np.asarray(tgt.array[te],dtype=np.float32).mean(1)
  for count in [25,100,500]:
   for seed in [42,43,44]:
    dest=out/f'{source}_to_{target}_{count}_s{seed}.json'
    if dest.exists():continue
    torch.manual_seed(seed);perm=np.random.default_rng(seed).permutation(count);cut=max(2,int(count*.8));tr=perm[:cut];va=perm[cut:];mx=x[tr].mean(0);my=y[tr].mean(0);sx=np.maximum(x[tr].std(0),.01);sy=np.maximum(y[tr].std(0),.01)
    xx=torch.tensor((x[:count]-mx)/sx);yy=torch.tensor((y[:count]-my)/sy);net=torch.nn.Sequential(torch.nn.Linear(x.shape[1],256),torch.nn.GELU(),torch.nn.Linear(256,y.shape[1]));opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01);best=float('inf');stale=0
    for epoch in range(100):
     net.train();opt.zero_grad();loss=(net(xx[tr])-yy[tr]).square().mean();loss.backward();opt.step();net.eval()
     with torch.no_grad():v=float((net(xx[va])-yy[va]).square().mean())
     if v<best:best=v;state={k:p.detach().clone() for k,p in net.state_dict().items()};stale=0
     else:stale+=1
     if stale>=10:break
    net.load_state_dict(state);net.eval()
    with torch.no_grad():mapped=net(torch.tensor((z-mx)/sx)).numpy()*sy+my
    reports=[]
    for probe in ['mean_linear','M0']:
     ck=torch.load(ROOT/'artifacts/probes'/f'{source}__{probe}__s{seed}__scratch_full/best.pt',map_location='cpu',weights_only=True);head=make_probe(probe,src.shape).eval();head.load_state_dict(ck['model'])
     with torch.no_grad():p=head(torch.tensor(mapped[:,None,:])).sigmoid().numpy()
     reports.append(dict(probe=probe,metrics=metrics(tgt.y[te],p),ci95=bootstrap(tgt.y[te],p,500,seed),predictions=[dict(sample_id=tgt.rows[i]['sample_id'],label=int(tgt.y[i]),probability=float(v)) for i,v in zip(te,p)]))
    torch.save(dict(state=state,mx=mx,my=my,sx=sx,sy=sy),dest.with_suffix('.pt'))
    atomic_json(dest,dict(source=source,target=target,anchors=count,seed=seed,parameters=sum(p.numel() for p in net.parameters()),validation_mse=best,epochs=epoch+1,anchor_cache_sha256=sha256(folder/'target_anchor_features.npy'),target_labels_used=0,reports=reports,caveat='Unlabeled source-training identical-text anchors only;80/20anchor fitting/validation included in budget. Global mean space only; no per-layer geometry claim.'))
    print(dest.name,flush=True)
 atomic_json(out/'complete.json',dict(status='complete'))
if __name__=='__main__':main()
