#!/usr/bin/env python3
"""Temporary synthetic recovery suite; not research results or model-update evidence."""
import json,sys,tempfile,unittest.mock
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from portability.storage import ShardWriter,ShardDataset,atomic_json
from portability.probe_training import POOLED,RAW,make_probe,atomic_torch
import run_update_probes as suite
root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(dir=root/'artifacts/tmp',prefix='update_probes_smoke_') as td:
    base=Path(td);name='updates_v1_qwen3_8b_step000050';folder=base/'artifacts'/f'stage1_{name}';writer=ShardWriter(folder/'hidden_states',{'synthetic':True},(3,4,16));rng=np.random.default_rng(42)
    for i in range(40):
        writer.append(rng.normal(size=(3,4,16)).astype('float16'),dict(dataset='SYNTHETIC_ONLY',original_id=str(i),split='train' if i<24 else 'dev' if i<32 else 'test',generated_text='fixture',label=i%2,n_gen_tokens=4,mean_token_logprob=-1-i%2,min_token_logprob=-2-i%2,mean_token_entropy=1+i%2))
    writer.flush();ds=ShardDataset(folder/'hidden_states');pool=folder/'pooled';pool.mkdir();np.save(pool/'layer_mean.npy',np.stack([ds[i][0].mean(1,dtype=np.float32) for i in range(40)]));atomic_json(pool/'manifest.json',dict(complete=True,source_shape=[3,4,16],records=[dict(r,judge_labels={j:r['label'] for j in ['mistral','qwen','phi']}) for r in ds.records]))
    (base/'configs').mkdir();cfg=dict(model='SYNTHETIC_ONLY',model_revision='fixture');atomic_json(base/'configs/stage1_qwen3_8b.json',cfg);atomic_json(folder/'config.json',dict(**cfg,adapter='fixture',adapter_sha256='synthetic'))
    for probe in POOLED+RAW:
        source=base/'artifacts/probes'/f'qwen3_8b__{probe}__s42__scratch_full';source.mkdir(parents=True);net=make_probe(probe,[3,4,16]);atomic_torch(source/'best.pt',dict(shape=[3,4,16],model=net.state_dict()))
    suite.ROOT=base
    argv=['suite','--corpora',name,'--seeds','42','--budgets','10','full','--epochs','1','--bootstrap','5','--device','cpu']
    with unittest.mock.patch.object(sys,'argv',argv):suite.main()
    fits=list((base/'artifacts/probes').glob('updates_v1*/complete.json'));assert len(fits)==20
    evals=list((base/'results/model_updates_v1').glob('*/evaluation.json'));assert len(evals)==8
    timestamps={p:p.stat().st_mtime_ns for p in (base/'artifacts/probes').glob('updates_v1*/best.pt')}
    with unittest.mock.patch.object(sys,'argv',argv):suite.main()
    assert all(p.stat().st_mtime_ns==stamp for p,stamp in timestamps.items())
    print('PASS: eight frozen probes, 20 scratch/warm fits, target calibration, ID validation and completed-checkpoint resume. Temporary synthetic fixtures removed.')
