#!/usr/bin/env python3
"""Frozen teacher-forced attention replay; immutable sample IDs and token sequences."""
import argparse,json,sys,time,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM,AutoModelForImageTextToText
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from portability.lookback import RatioCapture
from portability.storage import ShardDataset,atomic_json,sha256
ROOT=Path(__file__).resolve().parents[1]
def make_batch(rows,tok,device,max_input):
    sequences=[];bounds=[];lengths=[];prompts=[]
    for row in rows:
        encoded=tok(row['prompt'],return_offsets_mapping=True,truncation=True,max_length=max_input)
        if encoded['input_ids']!=row['prompt_token_ids']:raise ValueError('Replay prompt token IDs differ')
        marker=row['prompt'].rfind('\n\nAnswer:\n')
        if marker<0:raise ValueError('Unknown response marker')
        prefix=row.get('_lookback_prefix',sum(end<=marker for start,end in encoded['offset_mapping']))
        n=len(row['prompt_token_ids']);t=len(row['generated_token_ids'])
        if not 0<prefix<n or not t:raise ValueError('Invalid prompt partition or empty response')
        sequences.append(row['prompt_token_ids']+row['generated_token_ids'][:-1]);bounds.append(prefix);lengths.append(t);prompts.append(n)
    width=max(map(len,sequences));steps=max(lengths);pad=tok.pad_token_id
    ids=torch.full((len(rows),width),pad,dtype=torch.long,device=device);mask=torch.zeros_like(ids);positions=[];ends=[]
    for b,seq in enumerate(sequences):
        offset=width-len(seq);ids[b,offset:]=torch.tensor(seq,device=device);mask[b,offset:]=1
        positions.append([offset+prompts[b]-1+min(t,lengths[b]-1) for t in range(steps)]);ends.append([offset,offset+bounds[b]])
    return dict(input_ids=ids,attention_mask=mask,position_ids=mask.cumsum(1).sub(1).clamp_min(0)),dict(positions=torch.tensor(positions,device=device),prefix_ends=torch.tensor(ends,device=device),lengths=torch.tensor(lengths,device=device))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--corpora',nargs='+',required=True);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--batch',type=int,default=4);ap.add_argument('--output-prefix',default='lookback_v1');args=ap.parse_args()
    torch.set_num_threads(8);device=torch.device('cuda');model=None;model_key=None;tok=None;capture=RatioCapture();original=ALL_ATTENTION_FUNCTIONS['sdpa'];ALL_ATTENTION_FUNCTIONS.register('sdpa',capture.wrap(original))
    try:
      for corpus in args.corpora:
        cfg=json.loads((ROOT/'artifacts'/f'stage1_{corpus}/config.json').read_text());key=(cfg['model'],cfg['model_revision'],cfg.get('adapter_sha256'))
        ds=ShardDataset(ROOT/'artifacts'/f'stage1_{corpus}/hidden_states');rows=ds.records[:args.limit] if args.limit else ds.records
        out=ROOT/'artifacts'/args.output_prefix/corpus;out.mkdir(parents=True,exist_ok=True)
        protocol=dict(partition_invalid_policy='explicitly exclude cases whose saved right-truncated prompt lacks response marker; compare all methods on same supported IDs',corpus=corpus,model=cfg['model'],revision=cfg['model_revision'],adapter_sha256=cfg.get('adapter_sha256'),sample_count=len(rows),batch=args.batch,source_metadata_sha256=__import__('hashlib').sha256(json.dumps(ds.records,sort_keys=True).encode()).hexdigest(),feature='Lookback Lens per-head ratio of mean attention to input prefix versus answer marker plus previous generated tokens; average actual response prediction positions including EOS, no padded positions',response_label='our existing response-level labels; not original span-level benchmark',precision='BF16 model; selected post-RoPE QK weights and ratios computed FP32',reference_repo_commit=__import__('subprocess').check_output(['git','-C',str(ROOT/'artifacts/prior_methods/Lookback-Lens'),'rev-parse','HEAD'],text=True).strip())
        cp=out/'config.json'
        if cp.exists() and json.loads(cp.read_text())!=protocol:raise ValueError('Lookback protocol changed on resume')
        atomic_json(cp,protocol)
        completed={}
        for commit in sorted(out.glob('shard_*.json')):
            info=json.loads(commit.read_text());path=out/info['array']
            if sha256(path)!=info['sha256']:raise ValueError('Corrupt feature shard')
            for r in info['rows']:
                if r['sample_id'] in completed:raise ValueError('Duplicate committed feature ID')
                completed[r['sample_id']]=r
        unsupported_path=out/'unsupported.json'
        unsupported=json.loads(unsupported_path.read_text()) if unsupported_path.exists() else []
        unsupported_ids={r['sample_id'] for r in unsupported}
        pending=[]
        # Verify exact saved tokens and record unsupported partitions without altering prompts.
        tokenizer=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['model_revision'],local_files_only=True)
        for row in rows:
            if row['sample_id'] in completed or row['sample_id'] in unsupported_ids:continue
            enc=tokenizer(row['prompt'],return_offsets_mapping=True,truncation=True,max_length=cfg['max_input_tokens'])
            if enc['input_ids']!=row['prompt_token_ids']:raise ValueError('Replay prompt IDs changed')
            marker=row['prompt'].rfind('\n\nAnswer:\n');prefix=sum(end<=marker for start,end in enc['offset_mapping'])
            if marker<0 or not 0<prefix<len(row['prompt_token_ids']):
                unsupported.append(dict(sample_id=row['sample_id'],split=row['split'],reason='response marker unavailable in saved truncated input'));unsupported_ids.add(row['sample_id'])
            else:pending.append(dict(row,_lookback_prefix=prefix))
        atomic_json(unsupported_path,unsupported)
        if not pending:print('Lookback already complete',corpus,flush=True);continue
        if key!=model_key:
            if model is not None:del model;torch.cuda.empty_cache()
            mc=AutoConfig.from_pretrained(cfg['model'],revision=cfg['model_revision'],local_files_only=True);tok=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['model_revision'],local_files_only=True)
            if tok.pad_token_id is None:tok.pad_token=tok.eos_token
            factory=AutoModelForImageTextToText if mc.model_type=='mistral3' else AutoModelForCausalLM
            model=factory.from_pretrained(cfg['model'],revision=cfg['model_revision'],local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa')
            if cfg.get('adapter'):
                from portability.lora import load_adapter
                adapter=ROOT/cfg['adapter']
                if sha256(adapter)!=cfg['adapter_sha256']:raise ValueError('Adapter changed')
                load_adapter(model,adapter,cfg['model'],cfg['model_revision'])
            model.eval();model.requires_grad_(False);model_key=key
        begin=time.time();buffer=[];meta=[];shard=len(list(out.glob('shard_*.json')));peak=0
        def flush():
            nonlocal buffer,meta,shard
            if not buffer:return
            filename=f'shard_{shard:05d}.npy';path=out/filename;temp=out/(filename+'.tmp')
            with temp.open('wb') as f:np.save(f,np.stack(buffer));f.flush();os.fsync(f.fileno())
            os.replace(temp,path);atomic_json(out/f'shard_{shard:05d}.json',dict(array=filename,sha256=sha256(path),rows=meta));shard+=1;buffer=[];meta=[]
        for start in range(0,len(pending),args.batch):
            batch=pending[start:start+args.batch];inputs,context=make_batch(batch,tok,device,cfg['max_input_tokens']);capture.context=context;capture.layers={};torch.cuda.reset_peak_memory_stats();t0=time.time()
            with torch.inference_mode():model(**inputs,use_cache=False,logits_to_keep=1)
            features=torch.stack([capture.layers[i] for i in range(len(capture.layers))],1).float().cpu().numpy();capture.context=None;capture.layers={}
            if features.shape[0]!=len(batch) or not np.isfinite(features).all():raise ValueError('Invalid captured features')
            peak=max(peak,torch.cuda.max_memory_reserved()/2**30)
            for r,x in zip(batch,features):buffer.append(x);meta.append(dict(sample_id=r['sample_id'],original_id=r['original_id'],split=r['split'],n_gen_tokens=r['n_gen_tokens']))
            if len(buffer)>=128:flush()
            print(json.dumps(dict(corpus=corpus,processed=start+len(batch),pending=len(pending),batch_seconds=time.time()-t0,peak_reserved_GiB=peak,feature_shape=list(features.shape[1:]))),flush=True)
        flush();all_info=[json.loads(p.read_text()) for p in sorted(out.glob('shard_*.json'))];ids=[r['sample_id'] for c in all_info for r in c['rows']]
        if len(ids)+len(unsupported_ids)!=len(rows) or set(ids)|unsupported_ids!={r['sample_id'] for r in rows}:raise ValueError('Incomplete/extra feature IDs')
        atomic_json(out/'complete.json',dict(samples=len(ids),unsupported=len(unsupported_ids),seconds_this_call=time.time()-begin,peak_reserved_GiB=peak,shards=len(all_info)))
    finally:ALL_ATTENTION_FUNCTIONS.register('sdpa',original)
if __name__=='__main__':main()
