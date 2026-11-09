#!/usr/bin/env python3
"""Full-token update recovery: matched scratch and deployment warm-start budgets."""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from portability.storage import atomic_json, sha256
from portability.probe_training import Corpus, train, RAW
from run_probe_suite import evaluate
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--corpora',nargs='+',default=[f'updates_v1_qwen3_8b_step{s:06d}' for s in [50,150,300]])
    ap.add_argument('--probes',nargs='+',choices=RAW,default=list(RAW)); ap.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    ap.add_argument('--adaptations',nargs='+',choices=['scratch','warm'],default=['scratch','warm']); ap.add_argument('--budgets',nargs='+',default=['100','500','full']); ap.add_argument('--epochs',type=int,default=25)
    ap.add_argument('--patience',type=int,default=4); ap.add_argument('--batch',type=int,default=8); ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--bootstrap',type=int,default=500); ap.add_argument('--device',default='cuda'); args=ap.parse_args()
    torch.set_num_threads(args.threads); device=torch.device(args.device); root=args.root
    parent=json.loads((root/'configs/stage1_qwen3_8b.json').read_text())
    for name in args.corpora:
        data=Corpus(root,name,raw=True); cfg=json.loads((root/'artifacts'/f'stage1_{name}/config.json').read_text())
        if cfg['model']!=parent['model'] or cfg['model_revision']!=parent['model_revision'] or not cfg.get('adapter'): raise ValueError('Expected same-parent updated model')
        for probe in args.probes:
            for seed in args.seeds:
                source=root/'artifacts/probes'/f'qwen3_8b__{probe}__s{seed}__scratch_full'
                ck=torch.load(source/'best.pt',map_location='cpu',weights_only=True)
                if list(ck['shape'])!=list(data.shape): raise ValueError('Full-token warm-start shape mismatch')
                del ck
                for budget in args.budgets:
                    for adaptation in args.adaptations:
                        out=root/'artifacts/probes'/f'{name}__{probe}__s{seed}__{adaptation}_{budget}'
                        out.mkdir(parents=True,exist_ok=True)
                        provenance=dict(adapter_sha256=cfg['adapter_sha256'],manifest_sha256=sha256(root/'artifacts'/f'stage1_{name}/pooled/manifest.json'),source_checkpoint_sha256=sha256(source/'best.pt') if adaptation=='warm' else None)
                        if (out/'recovery_provenance.json').exists() and json.loads((out/'recovery_provenance.json').read_text())!=provenance: raise ValueError('Recovery provenance changed')
                        atomic_json(out/'recovery_provenance.json',provenance)
                        started=time.time(); net=train(data,probe,budget,seed,out,device,args.epochs,args.patience,args.batch,source if adaptation=='warm' else None)
                        if net is None: continue
                        if not (out/'evaluation.json').exists():
                            evaluate(net,data,device,args.batch,out,'raw_warm_start_model_update' if adaptation=='warm' else 'raw_target_trained_model_update',seed,args.bootstrap)
                        else:
                            saved=json.loads((out/'predictions.json').read_text())
                            expected=[(data.rows[i]['sample_id'],int(data.y[i])) for i in data.indices('test')]
                            if [(r['sample_id'],r['label']) for r in saved]!=expected: raise ValueError('Saved recovery test alignment changed')
                        print(json.dumps(dict(experiment=out.name,seconds_this_call=time.time()-started)),flush=True)
                        del net
                        if device.type=='cuda': torch.cuda.empty_cache()
    print('Full-token recovery complete. All label budgets include early-stopping labels.',flush=True)
if __name__=='__main__': main()
