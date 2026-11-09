#!/usr/bin/env python3
"""Same-text replay on a fixed development monitor; separates geometry from changed outputs."""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from portability.lora import attach,restore
from portability.storage import ShardDataset,atomic_json,sha256
from portability.probe_training import Corpus,make_probe,POOLED
ROOT=Path(__file__).resolve().parents[1]

def similarity(a,b):
    a=a.astype(np.float64);b=b.astype(np.float64)
    cosine=np.sum(a*b,axis=1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-12)
    a-=a.mean(0);b-=b.mean(0);ka=a@a.T;kb=b@b.T
    cka=float(np.sum(ka*kb)/max(np.linalg.norm(ka)*np.linalg.norm(kb),1e-12))
    return dict(linear_cka=cka,mean_cosine=float(cosine.mean()),relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-12)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--adapters',type=Path,default=ROOT/'artifacts/model_updates_v1/qwen_hotpot');ap.add_argument('--steps',nargs='+',type=int,default=[50,150,300]);ap.add_argument('--samples',type=int,default=256);ap.add_argument('--batch',type=int,default=4);ap.add_argument('--output',type=Path);ap.add_argument('--result-dir',type=Path);args=ap.parse_args()
    torch.set_num_threads(8);base=ROOT/'artifacts/stage1_qwen3_8b';source=Corpus(ROOT,'qwen3_8b');raw=ShardDataset(base/'hidden_states')
    eligible=source.indices('dev');idx=np.random.default_rng(42).permutation(eligible)[:args.samples]
    if len(idx)!=args.samples:raise ValueError('Insufficient valid monitor cases')
    if [r['sample_id'] for r in raw.records]!=[r['sample_id'] for r in source.rows]:raise ValueError('Replay/source order mismatch')
    records=[raw.records[i] for i in idx];shape=source.shape
    pilot=args.samples!=256 or args.steps!=[50,150,300]
    out=args.output or ROOT/'artifacts/model_updates_v1'/('replay_pilot' if pilot else 'replay');out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/stage1_qwen3_8b.json').read_text());tok=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['model_revision']);pad=tok.eos_token_id
    monitor=dict(sample_ids=[r['sample_id'] for r in records],source_manifest_sha256=sha256(base/'pooled/manifest.json'),split='dev',seed=42,protocol='teacher_forced_parent_response_prediction_positions_fixed96_mean_v1')
    if (out/'monitor.json').exists() and json.loads((out/'monitor.json').read_text())!=monitor:raise ValueError('Monitor cohort changed')
    atomic_json(out/'monitor.json',monitor)
    print('RESOURCE PLAN: one frozen BF16 parent on GPU, replay batch4, approximately 18–30 GiB VRAM, under32 GiB RAM, four ~144 MiB pooled replay arrays, eight CPU threads. No generator gradients.',flush=True)
    model=None;attached=False;features={}
    for step in [0,*args.steps]:
        adapter=args.adapters/f'step_{step:06d}/lora.pt';fingerprint=None if step==0 else sha256(adapter)
        dest=out/f'step_{step:06d}.npy';meta=out/f'step_{step:06d}.json';spec=dict(monitor=monitor,step=step,adapter_sha256=fingerprint,shape=[len(idx),shape[0],shape[2]])
        if meta.exists():
            saved=json.loads(meta.read_text())
            if saved['spec']!=spec or sha256(dest)!=saved['sha256']:raise ValueError('Replay artifact changed or corrupted')
            features[step]=np.load(dest);continue
        if model is None:
            model=AutoModelForCausalLM.from_pretrained(cfg['model'],revision=cfg['model_revision'],dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa');model.eval();model.requires_grad_(False)
        if step:
            ck=torch.load(adapter,map_location='cpu',weights_only=True)
            if ck['base_model']!=cfg['model'] or ck['base_revision']!=cfg['model_revision']:raise ValueError('Replay parent mismatch')
            if not attached:attach(model,ck['rank'],ck['alpha']);attached=True
            restore(model,ck['modules']);model.requires_grad_(False)
        result=np.empty(spec['shape'],dtype=np.float32);started=time.time();torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for start in range(0,len(records),args.batch):
                batch=records[start:start+args.batch];seq=[r['prompt_token_ids']+r['generated_token_ids'][:r['n_gen_tokens']-1] for r in batch];width=max(map(len,seq))
                ids=torch.full((len(seq),width),pad,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
                for j,t in enumerate(seq):ids[j,-len(t):]=torch.tensor(t,device='cuda');mask[j,-len(t):]=1
                positions=(mask.cumsum(1)-1).masked_fill(mask==0,0)
                states=model.model(input_ids=ids,attention_mask=mask,position_ids=positions,output_hidden_states=True,use_cache=False,return_dict=True).hidden_states[1:]
                if len(states)!=shape[0]:raise ValueError('Replay depth changed')
                for j,r in enumerate(batch):
                    offset=width-len(seq[j])+len(r['prompt_token_ids'])-1;length=r['n_gen_tokens']
                    result[start+j]=torch.stack([h[j,offset:offset+length].sum(0,dtype=torch.float32)/shape[1] for h in states]).cpu().numpy()
                del states,ids,mask,positions
                print(json.dumps(dict(step=step,replayed=min(start+len(batch),len(records)),total=len(records))),flush=True)
        with dest.with_suffix('.npy.tmp').open('wb') as f:np.save(f,result);f.flush();os.fsync(f.fileno())
        os.replace(dest.with_suffix('.npy.tmp'),dest);atomic_json(meta,dict(spec=spec,sha256=sha256(dest),seconds=time.time()-started,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30));features[step]=result
    if model is not None:del model;torch.cuda.empty_cache()
    reference=features[0];report=dict(monitor=monitor,n=len(idx),replay_numerical_control=[similarity(np.asarray(source.array[idx,l]),reference[:,l]) for l in range(shape[0])],checkpoints=[],caveat='Fixed parent responses and development cases, not fresh-generation test outcomes. Pooled similarity can miss token-level drift. One trajectory with three updates provides exploratory associations only.')
    projection=np.random.default_rng(42).normal(size=(shape[2],64)).astype(np.float32)/8
    for step in args.steps:
        target=features[step];layers=[]
        for l in range(shape[0]):
            m=similarity(reference[:,l],target[:,l]);a=reference[:,l]@projection;b=target[:,l]@projection;a-=a.mean(0);b-=b.mean(0);u,_,vt=np.linalg.svd(b.T@a,full_matrices=False);m['projected_procrustes_residual']=float(np.linalg.norm(b@(u@vt)-a)/max(np.linalg.norm(a),1e-12));layers.append(m)
        probes=[]
        for name in POOLED:
            for seed in [42,43,44]:
                ck=torch.load(ROOT/'artifacts/probes'/f'qwen3_8b__{name}__s{seed}__scratch_full/best.pt',map_location='cpu',weights_only=True);net=make_probe(name,shape).to('cuda');net.load_state_dict(ck['model']);net.eval();preds=[]
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    for x in [reference,target]:preds.append(net(torch.from_numpy(x).to('cuda')).float().sigmoid().cpu().numpy())
                probes.append(dict(probe=name,seed=seed,mean_absolute_probability_change=float(np.mean(np.abs(preds[1]-preds[0]))),mean_probability_shift=float(np.mean(preds[1]-preds[0]))));del net;torch.cuda.empty_cache()
        report['checkpoints'].append(dict(step=step,layers=layers,mean_linear_cka=float(np.mean([r['linear_cka'] for r in layers])),probe_relevant_score_drift=probes))
    result_dir=args.result_dir or ROOT/'results'/('model_updates_pilot' if pilot else 'model_updates_v1');result_dir.mkdir(parents=True,exist_ok=True);atomic_json(result_dir/'representation_drift.json',report)
    print('Same-text replay and representation drift complete',flush=True)
if __name__=='__main__':main()
