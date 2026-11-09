#!/usr/bin/env python3
"""Synthetic end-to-end smoke; never a scientific result or human label."""
import json,sys,tempfile,subprocess,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from portability.storage import ShardWriter,ShardDataset,atomic_json
from portability.probe_training import Corpus,train,make_probe,warm_parameters,depth_map,budget_split
root=Path(__file__).resolve().parents[1]
torch.set_num_threads(8)
with tempfile.TemporaryDirectory(prefix='probe_smoke_',dir=root/'artifacts/tmp') as td:
 base=Path(td)
 for name,layers in [('ministral3_8b',3),('qwen3_8b',4)]:
  folder=base/'artifacts'/f'stage1_{name}';writer=ShardWriter(folder/'hidden_states',dict(model=name,synthetic=True),(layers,4,16))
  rng=np.random.default_rng(42)
  for i in range(40):
   y=i%2;x=rng.normal(size=(layers,4,16)).astype(np.float16);x[:,:,0]+=y*2-1
   writer.append(x,dict(dataset='SYNTHETIC_ONLY',original_id=str(i),split='train' if i<24 else 'dev' if i<32 else 'test',generated_text='synthetic',label=y,n_gen_tokens=4,mean_token_logprob=-1-y,min_token_logprob=-2-y,mean_token_entropy=1+y))
  writer.flush();ds=ShardDataset(folder/'hidden_states');pooled=folder/'pooled';pooled.mkdir()
  np.save(pooled/'layer_mean.npy',np.stack([ds[i][0].mean(axis=1,dtype=np.float32) for i in range(len(ds))]))
  records=[dict(r,judge_labels={j:r['label'] for j in ['mistral','qwen','phi']}) for r in ds.records]
  atomic_json(pooled/'manifest.json',dict(complete=True,source_shape=[layers,4,16],records=records))
 subprocess.run([sys.executable,str(root/'scripts/run_probe_suite.py'),'--root',str(base),'--device','cpu','--seeds','42','--budgets','10','full','--probes','mean_linear','last_linear','last_mlp','M0','M1','--epochs','2','--patience','2','--bootstrap','5','--batch','8'],check=True)
 subprocess.run([sys.executable,str(root/'scripts/run_alignment.py'),'--root',str(base),'--device','cpu','--seeds','42','--pairs','10'],check=True)
 # Real architectural code, tiny tensor shape. Tests native-buffer warm-start and explicit depth mapping.
 for name in ['M2','M3','M4']:
  source=Corpus(base,'ministral3_8b',raw=True);target=Corpus(base,'qwen3_8b',raw=True)
  out=base/'artifacts/probes'/f'raw_{name}'
  net=train(source,name,'full',42,out,torch.device('cpu'),epochs=1,patience=1,batch=4)
  newer=make_probe(name,target.shape);warm_parameters(newer,net)
  x,_=target.batch(target.indices('test')[:2],torch.device('cpu'))
  assert newer(x.float()).shape==(2,)
  assert net(depth_map(x.float(),source.shape[0])).shape==(2,)
  assert torch.isfinite(newer(x.float())).all()
 # Label budget and resume invariants.
 data=Corpus(base,'qwen3_8b');tr,va=budget_split(data,'10',42)
 assert len(tr)+len(va)==10 and not set(tr)&set(va)
 assert not (set(tr)|set(va))&set(data.indices('test'))
 folder=base/'artifacts/probes/qwen3_8b__M0__s42__scratch_full'
 before=(folder/'best.pt').stat().st_mtime_ns
 train(data,'M0','full',42,folder,torch.device('cpu'),epochs=2,patience=2,batch=8)
 assert before==(folder/'best.pt').stat().st_mtime_ns
 print('PASS: synthetic pooled suite, warm-start, calibration, alignment, M2-M4 CPU, budget separation and completed-run resume.',flush=True)
