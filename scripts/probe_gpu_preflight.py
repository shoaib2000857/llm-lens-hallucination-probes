#!/usr/bin/env python3
"""Bounded real-tensor GPU forward/backward gate, run only after acquiring project GPU lock."""
import argparse,json,sys,time,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from portability.probe_training import Corpus,make_probe,POOLED,RAW
from portability.storage import atomic_json
ap=argparse.ArgumentParser();ap.add_argument('--raw',action='store_true');ap.add_argument('--batch',type=int,default=2);args=ap.parse_args()
root=Path(__file__).resolve().parents[1];torch.set_num_threads(8);device=torch.device('cuda');records=[]
for corpus in ['ministral3_8b','qwen3_8b']:
 data=Corpus(root,corpus,args.raw);idx=data.indices('train')[:args.batch]
 for name in RAW if args.raw else POOLED:
  torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
  net=make_probe(name,data.shape).to(device);x,y=data.batch(idx,device)
  with torch.autocast('cuda',dtype=torch.bfloat16):
   logits=net(x);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),y)
  loss.backward();torch.cuda.synchronize()
  if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):raise ValueError('Nonfinite preflight gradients')
  peak=torch.cuda.max_memory_reserved()/2**30
  row=dict(corpus=corpus,probe=name,batch=len(idx),seconds=time.perf_counter()-t0,peak_reserved_GiB=peak,loss=float(loss.detach()),status='passed');records.append(row);print(json.dumps(row),flush=True)
  if peak>32:raise RuntimeError('Preflight exceeded 32 GiB reserved; lower RAW_BATCH_SIZE before full training')
  del net,x,y,logits,loss;gc.collect();torch.cuda.empty_cache()
atomic_json(root/'results'/('raw_probe_gpu_preflight.json' if args.raw else 'pooled_probe_gpu_preflight.json'),records)
