#!/usr/bin/env python3
"""Label-blind repeated-answer disagreement; same saved prompts and evaluation IDs."""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch
from transformers import AutoTokenizer,AutoConfig,AutoModelForCausalLM,AutoModelForImageTextToText
from portability.storage import ShardDataset,atomic_json,sha256
from portability.probe_training import Corpus
from reproduce_saved_scores import metrics,bootstrap

def normalized(s):return ' '.join(''.join(c.lower() if c.isalnum() else ' ' for c in s).split())
def disagreement(original,answers):
 a=set(normalized(original).split());return float(np.mean([1-len(a&set(normalized(b).split()))/max(1,len(a|set(normalized(b).split()))) for b in answers]))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--samples',type=int,default=200);ap.add_argument('--draws',type=int,default=5);ap.add_argument('--pilot',action='store_true');a=ap.parse_args();torch.set_num_threads(8)
 out=ROOT/'results'/('sampling_disagreement_pilot' if a.pilot else 'sampling_disagreement_v1');out.mkdir(exist_ok=True)
 print('RESOURCE PLAN: one BF16 generator sequentially, batch1, five96-token independent samples per saved response;2models x3settings x200cases=6000 sampled answers. VRAM approximately18-26GiB,RAM<12GiB,8threads,0workers,output<1GiB,provisional1-4hours. No hidden-state dumps. Original saved prompts retained.',flush=True)
 for parent in ['qwen3_8b','ministral3_8b']:
  cfg=json.loads((ROOT/'configs'/f'stage1_{parent}.json').read_text());tok=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['model_revision'],local_files_only=True);tok.pad_token=tok.eos_token
  cls=AutoModelForImageTextToText if AutoConfig.from_pretrained(cfg['model'],revision=cfg['model_revision'],local_files_only=True).model_type=='mistral3' else AutoModelForCausalLM
  model=cls.from_pretrained(cfg['model'],revision=cfg['model_revision'],local_files_only=True,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval();model.requires_grad_(False)
  for corpus in [parent,f'distribution_v1_hotpotqa_{parent}',f'distribution_v1_pubmedqa_{parent}']:
   folder=out/corpus;folder.mkdir(exist_ok=True);data=Corpus(ROOT,corpus);ds=ShardDataset(ROOT/'artifacts'/f'stage1_{corpus}/hidden_states')
   rows=sorted([r for r in ds.records if r['split']=='test'],key=lambda r:hashlib.sha256(r['sample_id'].encode()).hexdigest())[:a.samples]
   spec=dict(model=cfg['model'],revision=cfg['model_revision'],ids=[r['sample_id'] for r in rows],draws=a.draws,max_new_tokens=96,temperature=.7,top_p=.95,seed=42,score='mean token-set Jaccard disagreement with original answer; not semantic entropy or factuality proof',selection='test ID hash without labels',script_sha256=sha256(Path(__file__)))
   if (folder/'config.json').exists():
    old=json.loads((folder/'config.json').read_text())
    if {k:v for k,v in old.items() if k!='script_sha256'}!={k:v for k,v in spec.items() if k!='script_sha256'}:raise ValueError('Changed resume protocol')
    if old.get('script_sha256')!=spec['script_sha256']:atomic_json(folder/'reporting_code_revision.json',dict(previous=old.get('script_sha256'),current=spec['script_sha256'],reason='Handle zero-valid-label pilot without fabricating metrics; sampling unchanged'))
   atomic_json(folder/'config.json',spec);records=[]
   for i,r in enumerate(rows):
    dest=folder/f'row_{i:04d}.json'
    if dest.exists():
     saved=json.loads(dest.read_text())
     if saved['sample_id']!=r['sample_id']:raise ValueError('ID mismatch')
    else:
     inputs=tok(r['prompt'],return_tensors='pt').to('cuda');answers=[]
     for draw in range(a.draws):
      torch.manual_seed(42+i*100+draw)
      with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=96,do_sample=True,temperature=.7,top_p=.95,pad_token_id=tok.eos_token_id,use_cache=True)
      answers.append(tok.decode(generated[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True))
     saved=dict(sample_id=r['sample_id'],answers=answers,score=disagreement(r['generated_text'],answers));atomic_json(dest,saved)
    records.append(saved)
    if (i+1)%10==0:print(json.dumps(dict(corpus=corpus,completed=i+1,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)),flush=True)
   label={r['sample_id']:int(data.y[i]) for i,r in enumerate(data.rows) if data.y[i] in (0,1)}
   valid=[dict(r,label=label[r['sample_id']]) for r in records if r['sample_id'] in label];y=np.array([r['label'] for r in valid]);p=np.array([r['score'] for r in valid])
   atomic_json(folder/'evaluation.json',dict(metrics=metrics(y,p) if len(y) else None,ci95=bootstrap(y,p,500,42) if len(y) else None,selected=len(records),valid=len(valid),predictions=valid,caveat='Score is disagreement, not a calibrated probability; Brier/ECE are descriptive only. Label-blind subset, strict existing consensus, fixed96-token cap. No target-label inversion.'))
  del model;torch.cuda.empty_cache()
 atomic_json(out/'complete.json',dict(status='complete'))
if __name__=='__main__':main()
