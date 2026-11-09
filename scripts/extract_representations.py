#!/usr/bin/env python3
"""Stage 1: immutable new-model representations. No judging or probe training."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from portability.storage import ShardWriter, atomic_json, canonical, sha256


def peak_rss_gib():
    # getrusage is blocked by this container's syscall policy; /proc is available.
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmHWM:'):
            return int(line.split()[1])/2**20
    return None


def prepare_examples(cfg, root):
    if cfg.get('examples_file'):
        source=Path(cfg['examples_file'])
        if not source.is_absolute():source=Path(__file__).resolve().parents[1]/source
        if sha256(source)!=cfg['examples_sha256']:raise ValueError('External example manifest checksum mismatch')
        data=json.loads(source.read_text())
        examples=data['examples']
        keys=[(e['dataset'],e['split'],e['original_id']) for e in examples]
        if len(set(keys))!=len(keys):raise ValueError('Duplicate external examples')
        for split in ['train','dev','test']:
            if sum(e['split']==split for e in examples)!=cfg[split+'_samples']:raise ValueError('External split counts mismatch')
        for e in examples:
            if e['dataset']!=cfg['dataset'] or not e['question'] or not e['context']:raise ValueError('Invalid external example')
        path=root/'examples.json'
        stored=dict(selection={'external_sha256':cfg['examples_sha256']},examples=examples)
        if path.exists() and json.loads(path.read_text())!=stored:raise ValueError('External selection changed')
        atomic_json(path,stored)
        return examples
    from datasets import load_dataset
    path=root/'examples.json'
    spec={k:cfg[k] for k in ['dataset','dataset_revision','seed','train_samples','dev_samples','test_samples']}
    if path.exists():
        d=json.loads(path.read_text())
        if d['selection'] != spec: raise ValueError('Existing example selection differs')
        return d['examples']
    train=load_dataset(cfg['dataset'], split='train', revision=cfg['dataset_revision'])
    val=load_dataset(cfg['dataset'], split='validation', revision=cfg['dataset_revision'])
    rng=random.Random(cfg['seed'])
    # Native train/test separation; dev/test partitioned by paragraph context to avoid shared passages.
    train_order=list(range(len(train))); rng.shuffle(train_order)
    grouped={}
    for i, row in enumerate(val):
        key=hashlib.sha256(row['context'].encode()).hexdigest(); grouped.setdefault(key,[]).append(i)
    keys=sorted(grouped);rng.shuffle(keys)
    dev_idx=[];test_idx=[]
    for key in keys:
        if len(dev_idx)<cfg['dev_samples']:dev_idx.extend(grouped[key])
        else:test_idx.extend(grouped[key])
    selections=[('train','train',train,train_order[:cfg['train_samples']]),
                ('dev','validation',val,dev_idx[:cfg['dev_samples']]),
                ('test','validation',val,test_idx[:cfg['test_samples']])]
    examples=[]
    for split,native,ds,idx in selections:
        for i in idx:
            row=ds[i]
            examples.append(dict(dataset=cfg['dataset'], split=split, original_split=native,
                original_id=str(row['id']), question=row['question'], context=row['context'],
                gold_answers=row['answers']['text'], is_unanswerable=not bool(row['answers']['text'])))
    if len(examples)!=sum(cfg[k] for k in ['train_samples','dev_samples','test_samples']):
        raise ValueError('Insufficient examples for requested split sizes')
    atomic_json(path,dict(selection=spec,examples=examples))
    return examples


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--limit',type=int,default=100,help='Total examples this invocation, 0=full user-launched run')
    ap.add_argument('--batch-size',type=int,default=2)
    ap.add_argument('--autotune',action='store_true',help='Measure batches up to 32 on up to 32 pending examples, then use fastest safe setting')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.limit<0 or args.batch_size<1:ap.error('Invalid limit/batch')
    cfg=json.loads(args.config.read_text());args.output.mkdir(parents=True,exist_ok=True)
    adapter_path=None
    if cfg.get('adapter'):
        adapter_path=Path(cfg['adapter'])
        if not adapter_path.is_absolute():adapter_path=Path(__file__).resolve().parents[1]/adapter_path
        if sha256(adapter_path)!=cfg['adapter_sha256']:raise ValueError('Adapter checkpoint checksum mismatch')
    lock=(args.output/'.job.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch
    from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM,AutoModelForImageTextToText
    from portability.extraction import extract_batch
    threads=int(os.environ.get('EXTRACTION_THREADS','4'))
    if not 1<=threads<=16:raise ValueError('Extraction threads must be between 1 and 16')
    torch.set_num_threads(threads);torch.manual_seed(cfg['seed'])
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for real extraction stage')
    mc=AutoConfig.from_pretrained(cfg['model'],revision=cfg['model_revision'])
    text_config=getattr(mc,'text_config',mc)
    shape=[text_config.num_hidden_layers,cfg['max_new_tokens'],text_config.hidden_size]
    rows=sum(cfg[k] for k in ['train_samples','dev_samples','test_samples'])
    target=min(rows,args.limit) if args.limit else rows
    bytes_needed=target*__import__('math').prod(shape)*2
    free=shutil.disk_usage(args.output).free
    plan=dict(experiment_id=cfg['experiment_id'],VRAM='18–30 GiB planning envelope; runtime hard stop below',
              RAM='under 16 GiB expected, mmap/metadata only',tensor_output_GiB=bytes_needed/2**30,
              runtime='pilot-measured; no unmeasured completion-time promise',cpu_threads=threads,cpu_workers=0,gpu_processes=1,
              batch_size=args.batch_size,free_disk_GiB=free/2**30)
    print('RESOURCE PLAN '+json.dumps(plan),flush=True)
    if free<bytes_needed+20*2**30:raise RuntimeError('Insufficient disk budget plus 20 GiB reserve')
    if (args.output/'config.json').exists() and json.loads((args.output/'config.json').read_text()) != cfg:
        raise ValueError('Output already belongs to a different configuration')
    atomic_json(args.output/'config.json',cfg)
    examples=prepare_examples(cfg,args.output)
    provenance=dict(config=cfg,examples_sha256=sha256(args.output/'examples.json'),
                    extraction='legacy_predictor_positions_v1_left_pad_eos_mask',storage_dtype='float16')
    writer=ShardWriter(args.output/'hidden_states',provenance,shape,'float16',16)
    finished=set()
    for path in sorted((args.output/'hidden_states').glob('shard_*.commit.json')):
        c=json.loads(path.read_text());data=json.loads((args.output/'hidden_states'/c['metadata']).read_text())
        finished.update((r['split'],r['original_id']) for r in data)
    selected=examples[:target]
    pending=[e for e in selected if (e['split'],e['original_id']) not in finished]
    if not pending:
        print('Requested samples already committed; nothing to do.',flush=True);return
    tok=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['model_revision'])
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    dtype=getattr(torch,cfg['compute_dtype'])
    # Explicit device: never silently offload a slow job to shared CPU.
    model_factory=AutoModelForImageTextToText if mc.model_type=='mistral3' else AutoModelForCausalLM
    model=model_factory.from_pretrained(cfg['model'],revision=cfg['model_revision'],
            dtype=dtype,device_map={'':'cuda:0'},attn_implementation='sdpa')
    if adapter_path:
        from portability.lora import load_adapter
        load_adapter(model,adapter_path,cfg['model'],cfg['model_revision'])
    model.eval();model.requires_grad_(False)
    start=time.time();done=0;bs=args.batch_size
    benchmark_path=args.output/'batch_benchmark.json'
    previous=json.loads(benchmark_path.read_text()) if benchmark_path.exists() else None
    if args.autotune and previous and previous.get('anchor_count',0)>=min(32,len(pending)):
        bs=previous['selected_batch_size']
        print(f'Reusing measured batch size {bs}; OOM backoff remains enabled',flush=True)
    elif args.autotune:
        anchors=pending[:32]
        prompts=[f"Context:\n{e['context']}\n\nQuestion:\n{e['question']}\n\nAnswer:\n" for e in anchors]
        measurements=[]
        for candidate in [1,2,4,8,16,32]:
            if candidate>len(anchors):continue
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t0=time.time()
            try:
                for offset in range(0,len(prompts),candidate):
                    trial=extract_batch(model,tok,prompts[offset:offset+candidate],cfg['max_input_tokens'],cfg['max_new_tokens'])
                    del trial
                torch.cuda.synchronize()
                peak=torch.cuda.max_memory_allocated()/2**30
                record=dict(batch_size=candidate,seconds=time.time()-t0,peak_GiB=peak)
                print('BATCH BENCHMARK '+json.dumps(record),flush=True)
                if peak<=32:measurements.append(record)
                else:break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache();print(f'Batch {candidate} exceeds VRAM; retaining previous candidate',flush=True);break
        if not measurements:raise RuntimeError('No batch passed the 32 GiB tuning envelope')
        bs=min(measurements,key=lambda r:r['seconds'])['batch_size']
        atomic_json(args.output/'batch_benchmark.json',dict(measurements=measurements,selected_batch_size=bs,anchor_count=len(anchors)))
    while pending:
        batch=pending[:bs]
        prompts=[f"Context:\n{e['context']}\n\nQuestion:\n{e['question']}\n\nAnswer:\n" for e in batch]
        torch.cuda.reset_peak_memory_stats();t0=time.time()
        try:
            result=extract_batch(model,tok,prompts,cfg['max_input_tokens'],cfg['max_new_tokens'])
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs==1:raise
            bs=max(1,bs//2);print(f'OOM: reducing batch size to {bs}; no rows skipped',flush=True);continue
        peak=torch.cuda.max_memory_allocated()/2**30
        for e,r,prompt in zip(batch,result,prompts):
            if cfg.get('record_prompt_truncation'):
                n=len(tok.encode(prompt,add_special_tokens=True))
                r.update(prompt_tokens_before_truncation=n,prompt_was_truncated=n>cfg['max_input_tokens'])
            hidden=r.pop('hidden');writer.append(hidden,dict(e,**r,prompt=prompt,prompt_version='legacy_context_question_answer_v1',generation_batch_size=len(batch)))
        pending=pending[len(batch):];done+=len(batch)
        telemetry=dict(completed_this_invocation=done,committed_total=len(writer.completed),remaining=len(pending),
            batch_size=bs,batch_seconds=time.time()-t0,elapsed_seconds=time.time()-start,
            samples_per_second=done/(time.time()-start),peak_allocated_GiB=peak,
            peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
            peak_RSS_GiB=peak_rss_gib())
        atomic_json(args.output/'progress.json',telemetry);print(json.dumps(telemetry),flush=True)
        if peak>34:
            writer.flush()
            if bs==1:raise RuntimeError('Peak VRAM exceeds 34 GiB envelope at batch 1')
            bs=max(1,bs//2)
    writer.flush()
    atomic_json(args.output/'completion.json',dict(experiment_id=cfg['experiment_id'],requested=target,committed=len(writer.completed),
        elapsed_seconds=time.time()-start,full_dataset_complete=len(writer.completed)==rows,
        torch=torch.__version__,transformers=__import__('transformers').__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),git_commit=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()))
    print('Extraction stage completed. Labels are pending separate judge stage.',flush=True)


if __name__=='__main__':main()
