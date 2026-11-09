#!/usr/bin/env python3
"""Synthetic controls/evaluation test; all temporary fixtures deleted, not research results."""
import importlib.util,json,sys,tempfile,unittest.mock
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from portability.storage import ShardWriter,ShardDataset,atomic_json
from portability.probe_training import make_probe,atomic_torch
import run_distribution_controls as controls
import evaluate_distribution_round as frozen
ROOT=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/tmp',prefix='distribution_smoke_') as td:
    root=Path(td);corpus='distribution_v1_pubmedqa_qwen3_8b';base=root/'artifacts'/f'stage1_{corpus}'
    writer=ShardWriter(base/'hidden_states',{'synthetic':True},(3,4,16));rng=np.random.default_rng(42)
    for i in range(40):
        x=rng.normal(size=(3,4,16)).astype('float16')
        writer.append(x,dict(dataset='SYNTHETIC_ONLY',split='test',original_id=str(i),context=f'context_{i//2}',generated_text='synthetic fixture',label=i%2,n_gen_tokens=4,mean_token_logprob=-1.,min_token_logprob=-2.,mean_token_entropy=1.))
    writer.flush();ds=ShardDataset(base/'hidden_states');pooled=base/'pooled';pooled.mkdir()
    records=[dict(r,judge_labels={j:r['label'] for j in ['mistral','qwen','phi']}) for r in ds.records]
    np.save(pooled/'layer_mean.npy',np.stack([ds[i][0].mean(1,dtype=np.float32) for i in range(40)]))
    atomic_json(pooled/'manifest.json',dict(complete=True,source_shape=[3,4,16],records=records))
    source_pool=root/'artifacts/stage1_qwen3_8b/pooled';source_pool.mkdir(parents=True)
    np.save(source_pool/'layer_mean.npy',np.load(pooled/'layer_mean.npy'))
    source_records=[dict(r,split='train' if i<32 else 'dev' if i<36 else 'test') for i,r in enumerate(records)]
    atomic_json(source_pool/'manifest.json',dict(complete=True,source_shape=[3,4,16],records=source_records))
    source=root/'artifacts/probes/qwen3_8b__mean_linear__s42__scratch_full';source.mkdir(parents=True)
    net=make_probe('mean_linear',[3,4,16]);atomic_torch(source/'best.pt',dict(shape=[3,4,16],model=net.state_dict()))
    (root/'configs').mkdir();cfg=dict(model='SYNTHETIC_ONLY',model_revision='fixture')
    atomic_json(root/'configs/stage1_qwen3_8b.json',cfg);atomic_json(base/'config.json',cfg)
    controls.ROOT=root;frozen.ROOT=root
    common=['--corpora',corpus,'--probes','mean_linear','--seeds','42','--bootstrap','5','--device','cpu']
    with unittest.mock.patch.object(sys,'argv',['controls',*common,'--epochs','1']):controls.main()
    with unittest.mock.patch.object(sys,'argv',['frozen',*common]):frozen.main()
    for mode in ['scratch','warm']:
        p=root/'results/distribution_v1'/f'{corpus}__mean_linear__s42__cv_{mode}'/'predictions.json'
        rows=json.loads(p.read_text());assert len(rows)==40 and len({r['sample_id'] for r in rows})==40
    print('PASS: grouped five-fold scratch/warm controls, full OOF coverage and frozen distribution evaluation. Synthetic fixture removed.')
