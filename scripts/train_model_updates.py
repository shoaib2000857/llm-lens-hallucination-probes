#!/usr/bin/env python3
"""Frozen BF16 parent + FP32 query/value LoRA; restartable controlled updates."""
import argparse,json,os,sys,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from portability.lora import attach,adapter_state,restore
from portability.storage import atomic_json,sha256
from portability.probe_training import atomic_torch
ROOT=Path(__file__).resolve().parents[1]

def encode(row,tok,maxseq):
    prompt=f"Context:\n{row['context']}\n\nQuestion:\n{row['question']}\n\nAnswer:\n"
    answer=tok.encode(row['answer'],add_special_tokens=False)+[tok.eos_token_id]
    p=tok.encode(prompt,add_special_tokens=True);budget=maxseq-len(answer)
    if budget<16:raise ValueError('Answer exceeds controlled training token budget')
    truncated=len(p)>budget;p=p[-budget:]
    return dict(input_ids=p+answer,labels=[-100]*len(p)+answer,truncated=truncated)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=ROOT/'configs/model_updates_v1.json');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--limit-steps',type=int,default=0);args=ap.parse_args()
    cfg=json.loads(args.config.read_text());out=args.output;out.mkdir(parents=True,exist_ok=True)
    path=ROOT/'artifacts/model_updates_v1/adaptation_examples.json';data=json.loads(path.read_text());provenance=dict(training_config=cfg,examples_sha256=sha256(path),protocol='bf16_parent_fp32_lora_answer_only_loss_v2_deterministic',deterministic_algorithms=True,cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'])
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=provenance:raise ValueError('Training resume provenance changed')
    atomic_json(out/'config.json',provenance)
    target=min(args.limit_steps,cfg['steps']) if args.limit_steps else cfg['steps']
    if target<1:raise ValueError('Invalid step limit')
    if (out/'completion.json').exists():
        completed=json.loads((out/'completion.json').read_text())['completed_steps']
        required=[s for s in cfg['checkpoints'] if s<=target] if not args.limit_steps else [target]
        if completed>=target and all((out/f'step_{s:06d}/metadata.json').exists() for s in required):
            for s in required:
                folder=out/f'step_{s:06d}';metadata=json.loads((folder/'metadata.json').read_text())
                if sha256(folder/'lora.pt')!=metadata['adapter_sha256']:raise ValueError('Corrupted committed adapter snapshot')
            print('Requested model updates already complete; snapshots verified.',flush=True);return
    torch.set_num_threads(8);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    print('RESOURCE PLAN '+json.dumps(dict(gpu_processes=1,cpu_threads=8,cpu_workers=0,VRAM='estimated 24–34 GiB; measured 34 GiB allocation guard',RAM='under 32 GiB',storage='adapter snapshots and optimizer only, under 1 GiB; parent checkpoint reused',steps=target,sequence_tokens=cfg['max_sequence_tokens'],microbatch=1,gradient_accumulation=cfg['gradient_accumulation'])),flush=True)
    tok=AutoTokenizer.from_pretrained(cfg['base_model'],revision=cfg['base_revision']);tok.pad_token=tok.eos_token
    rows=[encode(r,tok,cfg['max_sequence_tokens']) for r in data['examples'] if r['split']=='train'];dev=[encode(r,tok,cfg['max_sequence_tokens']) for r in data['examples'] if r['split']=='dev'][:32]
    atomic_json(out/'tokenization.json',dict(train=len(rows),validation_monitor=len(dev),training_prompt_truncated=sum(r['truncated'] for r in rows),truncation='left truncate training prompt only; preserve complete gold answer and EOS'))
    model=AutoModelForCausalLM.from_pretrained(cfg['base_model'],revision=cfg['base_revision'],dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa');model.requires_grad_(False)
    modules=attach(model,cfg['rank'],cfg['alpha']);model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.config.use_cache=False
    params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=0.,fused=True)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:min((step+1)/cfg['warmup_steps'],max(.1,1-step/cfg['steps'])))
    start=0;history=[]
    if (out/'last.pt').exists():
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=True);restore(model,state['modules']);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler']);start=state['step'];history=state['history'];torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    atomic_json(out/'parameters.json',dict(adapter_modules=modules,trainable_parameters=sum(p.numel() for p in params),frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad)))
    def save(step,snapshot=False):
        adapter=dict(format='explicit_qv_lora_v1',base_model=cfg['base_model'],base_revision=cfg['base_revision'],rank=cfg['rank'],alpha=cfg['alpha'],step=step,seed=cfg['seed'],training_manifest_sha256=provenance['examples_sha256'],modules=adapter_state(model))
        atomic_torch(out/'last.pt',dict(**adapter,optimizer=opt.state_dict(),scheduler=sched.state_dict(),history=history,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        if snapshot:
            folder=out/f'step_{step:06d}';folder.mkdir(exist_ok=True);atomic_torch(folder/'lora.pt',adapter);atomic_json(folder/'metadata.json',{k:v for k,v in adapter.items() if k!='modules'}|dict(adapter_sha256=sha256(folder/'lora.pt')))
        atomic_json(out/'progress.json',dict(completed_steps=step,target_steps=cfg['steps'],history=history))
    if start and start in cfg['checkpoints'] and not (out/f'step_{start:06d}/metadata.json').exists():save(start,True)
    for step in range(start,target):
        model.train();opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();t0=time.time();loss_total=0.
        chosen=np.random.default_rng(cfg['seed']+step).choice(len(rows),cfg['gradient_accumulation'],replace=False)
        for i in chosen:
            r=rows[int(i)];ids=torch.tensor([r['input_ids']],device='cuda');labels=torch.tensor([r['labels']],device='cuda')
            loss=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels,use_cache=False).loss
            if not torch.isfinite(loss):raise ValueError('Nonfinite LoRA loss')
            (loss/cfg['gradient_accumulation']).backward();loss_total+=float(loss.detach())/cfg['gradient_accumulation']
            del loss,ids,labels
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(norm):raise ValueError('Nonfinite adapter gradients')
        opt.step();sched.step()
        rec=dict(step=step+1,train_loss=loss_total,seconds=time.time()-t0,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
        snapshot=step+1 in cfg['checkpoints'] or (args.limit_steps and step+1==target)
        if snapshot:
            model.eval();values=[]
            with torch.inference_mode():
                for r in dev:
                    ids=torch.tensor([r['input_ids']],device='cuda');labels=torch.tensor([r['labels']],device='cuda');values.append(float(model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels,use_cache=False).loss))
            rec['validation_monitor_loss']=float(np.mean(values));rec['validation_monitor_n']=len(dev)
        history.append(rec);print(json.dumps(rec),flush=True)
        if snapshot or (step+1)%10==0:save(step+1,snapshot)
        if rec['peak_allocated_GiB']>34:
            save(step+1,snapshot);raise RuntimeError('Training exceeded 34 GiB allocation guard; resume after reviewing resource plan')
    if start<target:save(target,True)
    atomic_json(out/'completion.json',dict(completed_steps=max(start,target),full_trajectory_complete=max(start,target)>=cfg['steps'],checkpoints=cfg['checkpoints'],history=history))
    print('Controlled model update training complete for requested steps',flush=True)
if __name__=='__main__':main()
