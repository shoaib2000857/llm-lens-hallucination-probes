#!/usr/bin/env python3
"""Matched source splits/budgets; conditional positional and capacity controls."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from portability.ablations import NAMES
from portability.probe_training import Corpus,train,make_probe
from portability.storage import atomic_json
from run_probe_suite import evaluate

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');ap.add_argument('--probes',nargs='+',default=list(NAMES));a=ap.parse_args();torch.set_num_threads(8)
    print('RESOURCE PLAN:30 source fits=2 generators x5 variants x3 seeds, existing representations only; batch8 raw or64 pooled, estimated<8GiB VRAM,<16GiBRAM,reserve8GiB checkpoint/results,8 CPU threads. Runtime provisional8-20hours, early stopping dependent; checkpoints resume. Positional variants preserve trainable parameter counts; uniform weighting removes257 attention parameters; global MLP within1percent ofM2 capacity.',flush=True)
    reports=[]
    for source in ['ministral3_8b','qwen3_8b']:
        for name in a.probes:
            raw=name.startswith(('M2_','M3_','M4_'));data=Corpus(ROOT,source,raw=raw)
            if a.pilot:data.indices=lambda split,original=data.indices:original(split)[:16] if split=='test' else original(split)
            for seed in ([42] if a.pilot else [42,43,44]):
                # Keep pilot outputs separate from full scientific results.
                folder=ROOT/'artifacts/probes'/f'{source}__{name}__s{seed}__{ "ablation_pilot" if a.pilot else "ablation_full"}'
                net=train(data,name,'100' if a.pilot else 'full',seed,folder,torch.device('cuda'),epochs=1 if a.pilot else 25,patience=4,batch=8 if raw else 64)
                if net is None:raise ValueError('Ablation training unavailable')
                if not (folder/'evaluation.json').exists():evaluate(net,data,torch.device('cuda'),8 if raw else 64,folder,'controlled_source_ablation',seed,0 if a.pilot else 500)
                reports.append(dict(source=source,probe=name,seed=seed,parameters=sum(p.numel() for p in net.parameters()),metrics=json.loads((folder/'evaluation.json').read_text())['metrics']));del net;torch.cuda.empty_cache()
    # Aggregate every completed variant, including separate queue invocations.
    reports=[]
    suffix='ablation_pilot' if a.pilot else 'ablation_full'
    for folder in sorted((ROOT/'artifacts/probes').glob(f'*__{suffix}')):
        if not (folder/'evaluation.json').exists() or not (folder/'complete.json').exists():continue
        source,name,seed,_=folder.name.split('__')
        completion=json.loads((folder/'complete.json').read_text())
        reports.append(dict(source=source,probe=name,seed=int(seed[1:]),parameters=completion['parameters'],metrics=json.loads((folder/'evaluation.json').read_text())['metrics']))
    out=ROOT/'results'/('architecture_ablation_pilot' if a.pilot else 'architecture_ablation_v1');out.mkdir(exist_ok=True);atomic_json(out/'summary.json',dict(rows=reports,caveat='Same source splits,full training budget and optimizer protocol; positional ablations match parameter counts, global capacity control approximately matchesM2. Uniform layer weighting removes257 parameters. Pilot100-label one-epoch metrics are smoke checks only. Do not infer broad architectural causation from two model families.'))
if __name__=='__main__':main()
