#!/usr/bin/env python3
"""Version extraction configurations only after actual adapter snapshots exist."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from portability.storage import atomic_json,sha256
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--adapters',type=Path,default=ROOT/'artifacts/model_updates_v1/qwen_hotpot');ap.add_argument('--steps',nargs='+',type=int,default=[50,150,300]);ap.add_argument('--prefix',default='updates_v1');args=ap.parse_args()
args.adapters=args.adapters.resolve()
for step in args.steps:
    adapter=args.adapters/f'step_{step:06d}/lora.pt'
    if not adapter.exists():raise ValueError(f'Missing adapter snapshot {step}')
    cfg=json.loads((ROOT/'configs/stage1_qwen3_8b.json').read_text());examples=ROOT/'artifacts/stage1_qwen3_8b/examples.json'
    cfg.update(experiment_id=f'{args.prefix}_qwen3_8b_step{step:06d}',examples_file=str(examples.relative_to(ROOT)),examples_sha256=sha256(examples),adapter=str(adapter.relative_to(ROOT)),adapter_sha256=sha256(adapter),record_prompt_truncation=True)
    path=ROOT/'configs'/f'{cfg["experiment_id"]}.json'
    if path.exists() and json.loads(path.read_text())!=cfg:raise ValueError('Immutable update extraction config changed')
    atomic_json(path,cfg);print(path,flush=True)
