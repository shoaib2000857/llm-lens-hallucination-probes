#!/usr/bin/env python3
"""Bounded real-tensor forward checks of all eight probes; no training or claims."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from portability.storage import ShardDataset,atomic_json
from portability.probe_training import make_probe,POOLED,RAW
root=Path(__file__).resolve().parents[1];torch.set_num_threads(8);reports=[]
for base in sorted((root/'artifacts').glob('stage1_distribution_v1_*')):
    generator=next(g for g in ['ministral3_8b','qwen3_8b'] if base.name.endswith('_'+g))
    ds=ShardDataset(base/'hidden_states');n=min(8,len(ds));x=torch.from_numpy(ds.read_batch(list(range(n)))).to('cuda')
    for name in POOLED+RAW:
        ck=torch.load(root/'artifacts/probes'/f'{generator}__{name}__s42__scratch_full'/'best.pt',map_location='cpu',weights_only=True)
        assert list(ck['shape'])==ds.spec['shape']
        net=make_probe(name,ck['shape']).to('cuda');net.load_state_dict(ck['model']);net.eval();torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            logits=net(x.mean(dim=2,dtype=torch.float32) if name in POOLED else x)
        assert logits.shape==(n,) and torch.isfinite(logits).all()
        reports.append(dict(corpus=base.name,probe=name,n=n,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
        del net;torch.cuda.empty_cache()
    del x
out=root/'results/distribution_v1';out.mkdir(exist_ok=True);atomic_json(out/'probe_preflight.json',dict(status='pass',checks=reports,scientific_results=False))
print(f'PASS: {len(reports)} real-checkpoint/model-dimension forward checks, each at up to eight real target rows.')
