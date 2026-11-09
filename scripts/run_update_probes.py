#!/usr/bin/env python3
"""Post-deployment probe staleness, calibration and matched recovery curves."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from portability.storage import atomic_json,sha256
from portability.probe_training import Corpus,make_probe,train,POOLED,RAW,registry
from run_probe_suite import evaluate,calibrate,confidence_baselines
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--corpora',nargs='+',required=True);ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);ap.add_argument('--budgets',nargs='+',default=['25','50','100','250','500','1000','full']);ap.add_argument('--epochs',type=int,default=25);ap.add_argument('--bootstrap',type=int,default=500);ap.add_argument('--raw-batch',type=int,default=8);ap.add_argument('--device',default='cuda');args=ap.parse_args()
    torch.set_num_threads(8);device=torch.device(args.device)
    source_cfg=json.loads((ROOT/'configs/stage1_qwen3_8b.json').read_text())
    for corpus in args.corpora:
        cfg=json.loads((ROOT/'artifacts'/f'stage1_{corpus}'/'config.json').read_text())
        if cfg['model']!=source_cfg['model'] or cfg['model_revision']!=source_cfg['model_revision'] or not cfg.get('adapter'):raise ValueError('Not a controlled update of the deployment parent')
        for name in POOLED+RAW:
            target=Corpus(ROOT,corpus,name in RAW)
            for seed in args.seeds:
                source=ROOT/'artifacts/probes'/f'qwen3_8b__{name}__s{seed}__scratch_full';ck=torch.load(source/'best.pt',map_location='cpu',weights_only=True)
                if list(ck['shape'])!=list(target.shape):raise ValueError('Controlled update changed representation shape')
                net=make_probe(name,ck['shape']).to(device);net.load_state_dict(ck['model'])
                out=ROOT/'results/model_updates_v1'/f'{corpus}__{name}__s{seed}__frozen';out.mkdir(parents=True,exist_ok=True)
                provenance=dict(corpus=corpus,probe=name,seed=seed,source_checkpoint_sha256=sha256(source/'best.pt'),adapter_sha256=cfg['adapter_sha256'],feature_manifest_sha256=sha256(ROOT/'artifacts'/f'stage1_{corpus}/pooled/manifest.json'),bootstrap=args.bootstrap)
                if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=provenance:raise ValueError('Update evaluation provenance changed')
                atomic_json(out/'config.json',provenance)
                if (out/'evaluation.json').exists() and (out/'predictions.json').exists():
                    import numpy as np
                    saved=json.loads((out/'predictions.json').read_text());expected=[target.rows[i]['sample_id'] for i in target.indices('test')]
                    if [r['sample_id'] for r in saved]!=expected:raise ValueError('Saved frozen evaluation IDs differ')
                    p=np.asarray([r['probability'] for r in saved])
                else:
                    t0=time.time();record=dict(source=str(source),probe=name,seed=seed,total_target_labels=0);registry(target,out,record,'running')
                    p=evaluate(net,target,device,args.raw_batch if name in RAW else 64,out,'zero_shot_parameter_transfer',seed,args.bootstrap)
                    registry(target,out,record,'complete',time.time()-t0)
                if name in POOLED:
                    if not (out/'calibration/calibration.json').exists():calibrate(net,target,device,64,out/'calibration',[10,25,50,100,250,500],seed,ptest=p)
                    if seed==args.seeds[0] and name==POOLED[0]:confidence_baselines(target,out)
                del net;torch.cuda.empty_cache()
                # Frozen raw probes are included; repeated full-token recovery is a separate costly stage.
                if name in RAW:continue
                for budget in args.budgets:
                    for adaptation in ['scratch','warm']:
                        folder=ROOT/'artifacts/probes'/f'{corpus}__{name}__s{seed}__{adaptation}_{budget}'
                        net=train(target,name,budget,seed,folder,device,args.epochs,4,64,source if adaptation=='warm' else None)
                        if net is not None:evaluate(net,target,device,64,folder,'warm_start_model_update' if adaptation=='warm' else 'target_trained_model_update',seed,args.bootstrap if budget=='full' else 0)
                        del net;torch.cuda.empty_cache()
    print('Update probe round complete; raw-token retraining and replicated trajectories remain separate.')
if __name__=='__main__':main()
