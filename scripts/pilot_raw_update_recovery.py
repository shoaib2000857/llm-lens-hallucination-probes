#!/usr/bin/env python3
"""Bounded real full-size recovery pilot; temporary fits never enter headline results."""
import json, shutil, sys, tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from portability.storage import atomic_json
from portability.probe_training import Corpus
import run_raw_update_recovery as suite
ROOT=Path(__file__).resolve().parents[1]
print('RESOURCE PLAN: six one-epoch probe fits on 32 training, 8 development and 8 test cases at real36x96x4096 shape. No LLM. Batch8, eight CPU threads, one GPU process; possible M2 attention memory up to35GiB must be measured. Temporary checkpoints below1GiB, host memory below16GiB. Pilot results only.',flush=True)
class BoundedCorpus(Corpus):
    def indices(self,split):
        eligible=super().indices(split);count=16 if split=='train' else 4
        chosen=np.concatenate([eligible[self.y[eligible]==c][:count] for c in [0,1]])
        if len(chosen)!=2*count:raise ValueError('Pilot needs both classes')
        return chosen
with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/tmp',prefix='raw_recovery_real_pilot_') as td:
    root=Path(td);(root/'configs').mkdir();shutil.copyfile(ROOT/'configs/stage1_qwen3_8b.json',root/'configs/stage1_qwen3_8b.json')
    (root/'artifacts/probes').mkdir(parents=True);name='updates_v1_qwen3_8b_step000050'
    (root/'artifacts'/f'stage1_{name}').symlink_to(ROOT/'artifacts'/f'stage1_{name}',target_is_directory=True)
    for probe in ['M2','M3','M4']:
        n=f'qwen3_8b__{probe}__s42__scratch_full';(root/'artifacts/probes'/n).symlink_to(ROOT/'artifacts/probes'/n,target_is_directory=True)
    torch.cuda.reset_peak_memory_stats()
    with patch.object(suite,'Corpus',BoundedCorpus),patch.object(sys,'argv',['pilot','--root',str(root),'--corpora',name,'--seeds','42','--budgets','full','--epochs','1','--batch','8','--bootstrap','0']):suite.main()
    fits=list((root/'artifacts/probes').glob('updates*/complete.json'));assert len(fits)==6
    report=dict(status='pass',completed_fits=6,training_cases=32,development_cases=8,test_cases=8,shape=[36,96,4096],batch=8,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,caveat='One epoch per fit on bounded samples; not full-run performance evidence.')
    out=ROOT/'results/recovery_pilot';out.mkdir(parents=True,exist_ok=True);atomic_json(out/'real_tensor_smoke.json',report);print(json.dumps(report),flush=True)
