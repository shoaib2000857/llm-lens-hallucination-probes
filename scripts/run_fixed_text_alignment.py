#!/usr/bin/env python3
"""Unlabeled identical-text anchors; evaluate maps on natural target generations."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM,AutoModelForImageTextToText
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
from portability.storage import atomic_json,sha256,ShardDataset
from portability.probe_training import Corpus,make_probe
from reproduce_saved_scores import metrics,bootstrap

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--anchors',type=int,default=500);ap.add_argument('--batch',type=int,default=4);ap.add_argument('--pilot',action='store_true');a=ap.parse_args();torch.set_num_threads(8)
    names=['ministral3_8b','qwen3_8b'];out=ROOT/'results'/('fixed_text_alignment_pilot' if a.pilot else 'fixed_text_alignment_v1');out.mkdir(exist_ok=True)
    print('RESOURCE PLAN:1000 foreign-model teacher-forced responses for500 anchors per direction; one frozen BF16 model, batch4, estimated18-30GiB VRAM, RAM<16GiB, reserve3GiB features/maps/results,8 CPU threads. No new generations or target-test anchors. Models loaded sequentially. ExactD4096 maps; no hidden-state reshaping.',flush=True)
    data={n:Corpus(ROOT,n) for n in names}
    for source,target in [(names[0],names[1]),(names[1],names[0])]:
        src,tgt=data[source],data[target];folder=out/f'{source}_to_{target}';folder.mkdir(exist_ok=True)
        ds=ShardDataset(ROOT/'artifacts'/f'stage1_{source}/hidden_states')
        eligible=sorted([i for i,r in enumerate(ds.records) if r['split']=='train' and not r.get('prompt_was_truncated',False)],key=lambda i:hashlib.sha256(ds.records[i]['sample_id'].encode()).hexdigest())[:a.anchors]
        if len(eligible)!=a.anchors:raise ValueError('Insufficient unlabeled training anchors')
        rows=[ds.records[i] for i in eligible]
        cfg=json.loads((ROOT/'configs'/f'stage1_{target}.json').read_text());scfg=json.loads((ROOT/'configs'/f'stage1_{source}.json').read_text())
        spec=dict(source=source,target=target,anchor_ids=[r['sample_id'] for r in rows],source_manifest_sha256=sha256(ROOT/'artifacts'/f'stage1_{source}/pooled/manifest.json'),target_manifest_sha256=sha256(ROOT/'artifacts'/f'stage1_{target}/pooled/manifest.json'),target_model_revision=cfg['model_revision'],protocol='source training prompts and exact saved answer text replayed by target tokenizer; full response retained, EOS added only when original endedEOS; predictive response states summed over all tokens then divided by canonical96 and averaged over all layers; no target labels used in mapping')
        cp=folder/'config.json'
        if cp.exists() and json.loads(cp.read_text())!=spec:raise ValueError('Alignment provenance changed')
        atomic_json(cp,spec)
        source_features=np.asarray(src.array[eligible],dtype=np.float32).mean(1)
        cache=folder/'target_anchor_features.npy';marker=folder/'features_complete.json'
        if marker.exists():
            if sha256(cache)!=json.loads(marker.read_text())['sha256']:raise ValueError('Alignment cache corrupt')
        else:
            tok=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['model_revision'],local_files_only=True);stok=AutoTokenizer.from_pretrained(scfg['model'],revision=scfg['model_revision'],local_files_only=True)
            kind=AutoConfig.from_pretrained(cfg['model'],revision=cfg['model_revision'],local_files_only=True).model_type
            cls=AutoModelForImageTextToText if kind=='mistral3' else AutoModelForCausalLM
            model=cls.from_pretrained(cfg['model'],revision=cfg['model_revision'],local_files_only=True,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval();model.requires_grad_(False)
            for start in range(0,len(rows),a.batch):
                shard=folder/f'anchor_{start:05d}.npy';meta=shard.with_suffix('.json')
                if meta.exists():
                    if sha256(shard)!=json.loads(meta.read_text())['sha256']:raise ValueError('Anchor shard corrupt')
                    continue
                batch=rows[start:start+a.batch];sequences=[];sizes=[];prompt_sizes=[]
                for r in batch:
                    p=tok.encode(r['prompt']);ans=tok.encode(r['generated_text'],add_special_tokens=False)
                    if r['generated_token_ids'][-1]==stok.eos_token_id:ans.append(tok.eos_token_id)
                    if not ans:raise ValueError('Empty retokenized response')
                    if len(p)+len(ans)>4096:raise ValueError('Anchor exceeds4096; do not silently truncate')
                    sequences.append(p+ans[:-1]);sizes.append(len(ans));prompt_sizes.append(len(p))
                width=max(map(len,sequences));ids=torch.full((len(batch),width),tok.pad_token_id or tok.eos_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
                for b,seq in enumerate(sequences):ids[b,-len(seq):]=torch.tensor(seq,device='cuda');mask[b,-len(seq):]=1
                with torch.inference_mode():
                    states=model(input_ids=ids,attention_mask=mask,position_ids=mask.cumsum(1).sub(1).clamp_min(0),output_hidden_states=True,use_cache=False,logits_to_keep=1).hidden_states[1:]
                    pooled=[]
                    for b in range(len(batch)):
                        pos=width-len(sequences[b])+prompt_sizes[b]-1
                        pooled.append(torch.stack([h[b,pos:pos+sizes[b]].sum(0,dtype=torch.float32)/96 for h in states]).mean(0).cpu().numpy())
                with shard.with_suffix('.npy.tmp').open('wb') as f:np.save(f,np.stack(pooled))
                os.replace(shard.with_suffix('.npy.tmp'),shard);atomic_json(meta,dict(ids=[r['sample_id'] for r in batch],sha256=sha256(shard),retokenized_answer_lengths=sizes));del states,ids,mask
                print(json.dumps(dict(source=source,target=target,replayed=start+len(batch))),flush=True)
            with cache.with_suffix('.npy.tmp').open('wb') as f:np.save(f,np.concatenate([np.load(p) for p in sorted(folder.glob('anchor_*.npy'))]))
            os.replace(cache.with_suffix('.npy.tmp'),cache);atomic_json(marker,dict(sha256=sha256(cache)));del model;torch.cuda.empty_cache()
        x=np.load(cache);y=source_features
        if x.shape!=y.shape:raise ValueError('Anchor geometry mismatch')
        te=tgt.indices('test');native=np.asarray(tgt.array[te],dtype=np.float32).mean(1);cal=np.random.default_rng(42).permutation(tgt.indices('train'))[:100];native_cal=np.asarray(tgt.array[cal],dtype=np.float32).mean(1)
        if a.pilot:
            atomic_json(folder/'pilot_complete.json',dict(shape=list(x.shape),finite=bool(np.isfinite(x).all())));continue
        for count in [25,50,100,250,500]:
            if count>len(x):continue
            xx=torch.tensor(x[:count],device='cuda');yy=torch.tensor(y[:count],device='cuda');mx=xx.mean(0);my=yy.mean(0);u=xx-mx;v=yy-my
            gram=u@u.T;ridge=.01*gram.diag().mean().clamp_min(1e-6);dual=torch.linalg.solve(gram+ridge*torch.eye(count,device='cuda'),v)
            maps={'ridge':lambda z:(z-mx)@u.T@dual+my}
            if count==500:
                left,_,right=torch.linalg.svd(u.T@v,full_matrices=False);rotation=left@right;maps['orthogonal_procrustes']=lambda z:(z-mx)@rotation+my
                atomic_json(folder/'procrustes_check.json',dict(orthogonality_relative_error=float(torch.linalg.norm(rotation.T@rotation-torch.eye(rotation.shape[0],device='cuda'))/rotation.shape[0])))
            z=torch.tensor(native,device='cuda');zc=torch.tensor(native_cal,device='cuda')
            for method,mapper in maps.items():
                with torch.inference_mode():mapped=mapper(z);mapped_cal=mapper(zc)
                for probe in ['mean_linear','M0']:
                    for seed in [42,43,44]:
                        dest=folder/f'{method}_{count}_{probe}_s{seed}.json'
                        if dest.exists():continue
                        ck=torch.load(ROOT/'artifacts/probes'/f'{source}__{probe}__s{seed}__scratch_full/best.pt',map_location='cpu',weights_only=True);net=make_probe(probe,src.shape).cuda().eval();net.load_state_dict(ck['model'])
                        with torch.inference_mode():p=net(mapped[:,None,:]).sigmoid().cpu().numpy();raw=net(z[:,None,:]).sigmoid().cpu().numpy();pc=net(mapped_cal[:,None,:]).sigmoid().cpu().numpy()
                        fit=LogisticRegression().fit(logit(np.clip(pc,1e-6,1-1e-6))[:,None],tgt.y[cal]);pp=fit.predict_proba(logit(np.clip(p,1e-6,1-1e-6))[:,None])[:,1]
                        atomic_json(dest,dict(source=source,target=target,probe=probe,seed=seed,method=method,anchors=count,alignment_target_labels=0,calibration_target_labels=len(cal),direct=metrics(tgt.y[te],raw),aligned=metrics(tgt.y[te],p),aligned_plus_platt=metrics(tgt.y[te],pp),aligned_ci95=bootstrap(tgt.y[te],p,500,seed),predictions=[dict(sample_id=tgt.rows[i]['sample_id'],label=int(tgt.y[i]),raw=float(r),aligned=float(q),calibrated=float(c)) for i,r,q,c in zip(te,raw,p,pp)],caveat='Same source training answers form unlabeled anchors; evaluation uses natural target held-out responses. Global mean probes only; cross-token/per-layer transfer not claimed. Tokenizers differ; no response truncation.'))
                        del net
            del xx,yy,z,zc;torch.cuda.empty_cache()
    atomic_json(out/'complete.json',dict(status='complete',pilot=a.pilot))
if __name__=='__main__':main()
