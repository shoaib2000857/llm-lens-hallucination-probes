#!/usr/bin/env python3
"""Frozen modern same-family parameter transfer; exact compatible shape required."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from portability.probe_training import Corpus,make_probe,POOLED,RAW
from portability.storage import atomic_json,sha256
from run_probe_suite import evaluate,calibrate

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--corpus',required=True);a=ap.parse_args();torch.set_num_threads(8)
    cfg=json.loads((ROOT/'artifacts'/f'stage1_{a.corpus}/config.json').read_text());source=cfg['source_corpus']
    for probe in POOLED+RAW:
        data=Corpus(ROOT,a.corpus,raw=probe in RAW)
        for seed in [42,43,44]:
            file=ROOT/'artifacts/probes'/f'{source}__{probe}__s{seed}__scratch_full/best.pt';ck=torch.load(file,map_location='cpu',weights_only=True)
            if list(ck['shape'])!=list(data.shape):raise ValueError('Same-family geometry differs; explicit adapter required')
            out=ROOT/'results/samefamily_v1'/f'{a.corpus}__{probe}__s{seed}';out.mkdir(parents=True,exist_ok=True)
            config=dict(source_corpus=source,target_corpus=a.corpus,source_checkpoint_sha256=sha256(file),target_manifest_sha256=sha256(ROOT/'artifacts'/f'stage1_{a.corpus}/pooled/manifest.json'),target_model=cfg['model'],target_revision=cfg['model_revision'],mode='zero_shot_same_family_parameter_transfer',target_training_labels=0)
            if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config:raise ValueError('Transfer provenance changed')
            atomic_json(out/'config.json',config)
            if (out/'complete.json').exists() and (not len(data.indices('train')) or (out/'calibration'/ 'calibration.json').exists()):continue
            net=make_probe(probe,data.shape).cuda().eval();net.load_state_dict(ck['model']);began=time.time()
            p=evaluate(net,data,torch.device('cuda'),8 if probe in RAW else 64,out,config['mode'],seed,500)
            if len(data.indices('train')):calibrate(net,data,torch.device('cuda'),8 if probe in RAW else 64,out/'calibration',[25,50,100,250,500],seed,ptest=p)
            atomic_json(out/'complete.json',dict(seconds=time.time()-began));del net;torch.cuda.empty_cache()
    print('Same-family frozen transfer complete:'+a.corpus,flush=True)
if __name__=='__main__':main()
