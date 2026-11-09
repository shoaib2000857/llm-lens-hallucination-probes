#!/usr/bin/env python3
"""Explicit response-level SpikeScore+NLL adaptation on a fixed label-blind subset."""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'artifacts/prior_methods/SpikeScore'))
from portability.storage import atomic_json, ShardDataset
from strategy_library import StrategyDefinitions, StrategySelector
from transformers import AutoTokenizer, AutoModelForCausalLM

def spike(scores):
    a=np.asarray(scores,dtype=float)
    if len(a)<3 or not np.isfinite(a).all():raise ValueError('Spike requires at least three finite scores')
    return float(np.abs(np.diff(a,n=2)).max())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--samples',type=int,default=200);ap.add_argument('--turns',type=int,default=20);ap.add_argument('--batch',type=int,default=4);ap.add_argument('--max-new',type=int,default=256);ap.add_argument('--output',default='spikescore_ppl_v1');a=ap.parse_args()
    if a.turns<3:ap.error('Need at least three continuations')
    torch.set_num_threads(8)
    cfg=json.loads((ROOT/'configs/stage1_qwen3_8b.json').read_text())
    out=ROOT/'results'/a.output;out.mkdir(parents=True,exist_ok=True)
    protocol=dict(model=cfg['model'],revision=cfg['model_revision'],samples_per_corpus=a.samples,turns=a.turns,batch=a.batch,max_new_tokens=a.max_new,max_context=4096,seed=42,temperature=.6,top_p=.95,schedule='official repository progressive',repo_commit=subprocess.check_output(['git','-C',str(ROOT/'artifacts/prior_methods/SpikeScore'),'rev-parse','HEAD'],text=True).strip(),caveat='Adapted context-grounded response-label protocol; saved original answer retained,20 continuation turns scored by mean negative token log probability, max absolute second difference. Qwen reasoning enabled. Finite256-token continuation cap and4096-token left context truncation differ from original study; cap/truncation tracked. Not SAPLMA/SEP or original benchmark reproduction. Selection by sample-ID hash, without labels. No target-label score inversion.')
    protocol['scoring_distribution']='Unwarped model softmax, teacher-forced replay; mean response negative log probability including EOS'
    protocol['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cp=out/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=protocol:raise ValueError('Immutable run configuration changed')
    atomic_json(cp,protocol)
    tok=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['model_revision'],local_files_only=True,padding_side='left');tok.pad_token=tok.eos_token;tok.truncation_side='left'
    model=AutoModelForCausalLM.from_pretrained(cfg['model'],revision=cfg['model_revision'],local_files_only=True,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval();model.requires_grad_(False)
    selector=StrategySelector(StrategyDefinitions())
    for corpus in ['qwen3_8b','distribution_v1_hotpotqa_qwen3_8b','distribution_v1_pubmedqa_qwen3_8b']:
        metadata=ShardDataset(ROOT/'artifacts'/f'stage1_{corpus}/hidden_states').records
        rows=sorted([r for r in metadata if r['split']=='test'],key=lambda r:hashlib.sha256(r['sample_id'].encode()).hexdigest())[:a.samples]
        folder=out/corpus;folder.mkdir(exist_ok=True)
        manifest=dict(ids=[r['sample_id'] for r in rows])
        if (folder/'selection.json').exists() and json.loads((folder/'selection.json').read_text())!=manifest:raise ValueError('Selection changed')
        atomic_json(folder/'selection.json',manifest)
        for start in range(0,len(rows),a.batch):
            batch=rows[start:start+a.batch];path=folder/f'batch_{start:05d}.json'
            if path.exists():
                saved=json.loads(path.read_text())
                if [r['sample_id'] for r in saved]!=[r['sample_id'] for r in batch]:raise ValueError('Resume ID mismatch')
                continue
            histories=[[dict(role='user',content=r['prompt']),dict(role='assistant',content=r['generated_text'])] for r in batch]
            records=[dict(sample_id=r['sample_id'],turns=[],scores=[]) for r in batch];began=time.time()
            for turn in range(1,a.turns+1):
                follow=selector.select_progressive(turn).text
                for history in histories:history.append(dict(role='user',content=follow))
                prompts=[tok.apply_chat_template(h,tokenize=False,add_generation_prompt=True,enable_thinking=True) for h in histories]
                full_lengths=[len(tok.encode(p,add_special_tokens=False)) for p in prompts]
                inputs=tok(prompts,return_tensors='pt',padding=True,truncation=True,max_length=4096,add_special_tokens=False).to('cuda')
                torch.manual_seed(42+start*100+turn)
                with torch.inference_mode():
                    result=model.generate(**inputs,max_new_tokens=a.max_new,do_sample=True,temperature=.6,top_p=.95,pad_token_id=tok.pad_token_id,return_dict_in_generate=True,output_scores=False)
                    tokens=result.sequences[:,inputs['input_ids'].shape[1]:]
                    # PPL backbone scores the unwarped model distribution, not
                    # temperature/top-p filtered sampling probabilities.
                    replay=result.sequences[:,:-1]
                    replay_mask=torch.cat([inputs['attention_mask'],torch.ones_like(tokens[:,:-1])],dim=1)
                    raw=model(input_ids=replay,attention_mask=replay_mask,position_ids=replay_mask.cumsum(1).sub(1).clamp_min(0),use_cache=False,logits_to_keep=tokens.shape[1]).logits
                    transition=raw.float().log_softmax(-1).gather(-1,tokens.unsqueeze(-1)).squeeze(-1)
                    del raw
                tokens=tokens.cpu();transition=transition.float().cpu()
                for b in range(len(batch)):
                    ids=tokens[b].tolist();n=next((i+1 for i,x in enumerate(ids) if x==tok.eos_token_id),len(ids));text=tok.decode(ids[:n],skip_special_tokens=True);score=float(-transition[b,:n].mean())
                    records[b]['scores'].append(score);records[b]['turns'].append(dict(turn=turn,followup=follow,text=text,token_count=n,hit_cap=n==a.max_new,prompt_truncated=full_lengths[b]>4096));histories[b].append(dict(role='assistant',content=text))
                del result,transition,tokens
            for r in records:r['spikescore']=spike(r['scores'])
            atomic_json(path,records)
            print(json.dumps(dict(corpus=corpus,completed=start+len(batch),seconds=time.time()-began,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)),flush=True)
        atomic_json(folder/'complete.json',dict(samples=len(rows),turns=a.turns))
    atomic_json(out/'complete.json',dict(status='complete'))

if __name__=='__main__':main()
