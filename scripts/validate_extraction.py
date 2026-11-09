#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
from portability.storage import ShardDataset,atomic_json

ap=argparse.ArgumentParser()
ap.add_argument('root',type=Path)
ap.add_argument('--minimum',type=int,default=10)
args=ap.parse_args()
ds=ShardDataset(args.root/'hidden_states',verify=True)
if len(ds)<args.minimum:raise ValueError('Not enough committed rows')
seen=set();contexts={};empty=0
for rec in ds.records:
    key=(rec['split'],rec['original_id'])
    if key in seen:raise ValueError('Duplicate original example')
    seen.add(key)
    if rec['n_gen_tokens'] != len(rec['generated_token_ids']):raise ValueError('Token count mismatch')
    if len(rec['token_logprobs'])!=rec['n_gen_tokens']:raise ValueError('Confidence alignment mismatch')
    contexts.setdefault(rec['split'],set()).add(rec['context'])
    empty+=not bool(rec['generated_text'].strip())
if contexts.get('dev',set()) & contexts.get('test',set()):raise ValueError('Development/test context overlap')
for i in sorted(set([0,len(ds)//2,len(ds)-1])):
    x,r=ds[i]
    if not np.isfinite(x).all():raise ValueError('Nonfinite tensor')
    if np.any(x[:,r['n_gen_tokens']:]):raise ValueError('Padding is not zero')
report=dict(committed_rows=len(ds),empty_responses=empty,shape=ds.spec['shape'],dtype=ds.spec['dtype'],
            checksum_validation='all committed files',tensor_spot_checks=3,passed=True)
atomic_json(args.root/'validation.json',report)
print(json.dumps(report),flush=True)
