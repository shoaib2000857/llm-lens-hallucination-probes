#!/usr/bin/env python3
"""Temporary synthetic next-round integration checks; never research evidence."""
import json, sys, tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from portability.storage import ShardWriter, ShardDataset, atomic_json
from portability.probe_training import make_probe, atomic_torch
import run_raw_update_recovery as recovery
import simulate_probe_updates as schedules
import compare_update_recovery as comparisons
root=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
with tempfile.TemporaryDirectory(dir=root/'artifacts/tmp',prefix='recovery_smoke_') as td:
    base=Path(td);(base/'configs').mkdir();cfg=dict(model='SYNTHETIC_ONLY',model_revision='fixture')
    atomic_json(base/'configs/stage1_qwen3_8b.json',cfg)
    rng=np.random.default_rng(42)
    for step in [0,50,150,300]:
        name='qwen3_8b' if not step else f'updates_v1_qwen3_8b_step{step:06d}'
        folder=base/'artifacts'/f'stage1_{name}';writer=ShardWriter(folder/'hidden_states',dict(synthetic=step),(3,4,16))
        for i in range(48):
            writer.append(rng.normal(size=(3,4,16)).astype('float16'),dict(dataset='SYNTHETIC_ONLY',original_id=str(i),split='train' if i<32 else 'dev' if i<40 else 'test',generated_text='fixture',label=i%2))
        writer.flush();ds=ShardDataset(folder/'hidden_states');pool=folder/'pooled';pool.mkdir()
        np.save(pool/'layer_mean.npy',np.stack([ds[i][0].mean(1,dtype=np.float32) for i in range(48)]))
        rows=[dict(r,judge_labels={j:r['label'] for j in ['mistral','qwen','phi']}) for r in ds.records]
        atomic_json(pool/'manifest.json',dict(complete=True,source_shape=[3,4,16],records=rows))
        atomic_json(folder/'config.json',dict(**cfg,adapter='fixture',adapter_sha256='fixture'))
        for probe in ['M2','M3','M4','M1']:
            ckfolder=base/'artifacts/probes'/(f'qwen3_8b__{probe}__s42__scratch_full' if not step else f'{name}__{probe}__s42__warm_10')
            ckfolder.mkdir(parents=True);net=make_probe(probe,[3,4,16]);atomic_torch(ckfolder/'best.pt',dict(shape=[3,4,16],model=net.state_dict()))
            if step:
                atomic_json(ckfolder/'config.json',dict(train_ids=[r['sample_id'] for r in rows[:8]],validation_ids=[r['sample_id'] for r in rows[8:10]]))
                atomic_json(ckfolder/'complete.json',dict(seconds_this_invocation=1.))
    replay=base/'artifacts/model_updates_v1/replay';replay.mkdir(parents=True)
    for step in [0,50,150,300]:np.save(replay/f'step_{step:06d}.npy',rng.normal(size=(8,3,16)).astype('float32'))
    argv=['recovery','--root',str(base),'--corpora','updates_v1_qwen3_8b_step000050','--seeds','42','--budgets','16','--epochs','1','--batch','4','--bootstrap','5','--device','cpu','--threads','2']
    with patch.object(sys,'argv',argv):recovery.main()
    fits=list((base/'artifacts/probes').glob('*__M[234]__s42__*_16/complete.json'));assert len(fits)==6
    stamps={p:p.stat().st_mtime_ns for p in (base/'artifacts/probes').glob('*__M[234]__s42__*_16/best.pt')}
    with patch.object(sys,'argv',argv):recovery.main()
    assert all(p.stat().st_mtime_ns==stamp for p,stamp in stamps.items())
    argv=['schedules','--root',str(base),'--probes','M1','--seeds','42','--recovery-budget','10','--monitor-samples','4','--calibration-labels','4','--device','cpu','--threads','2']
    with patch.object(sys,'argv',argv):schedules.main()
    report=json.loads((base/'results/model_update_schedules_v1/summary.json').read_text());policies={r['policy']:r for r in report['results']}
    assert len(policies)==6 and policies['every_checkpoint']['update_count']==3 and policies['every_checkpoint']['total_target_labels']==30
    assert policies['every_2_checkpoints']['update_count']==1 and policies['never']['total_target_labels']==0 and policies['platt_only']['total_target_labels']==12
    assert policies['validation_drop']['total_target_labels']>=12
    with patch.object(sys,'argv',argv):schedules.main()
    y=np.array([0,1]*8);a=np.linspace(.1,.9,16);paired=comparisons.paired(y,a,a,10,42)
    assert all(v['observed']==0 and v['ci95']==[0.,0.] for v in paired['differences'].values())
print('PASS: six raw scratch/warm fits and exact completed-fit resume; six policies and label costs; schedule cache resume; paired identity intervals. Synthetic fixtures removed.')
