#!/usr/bin/env python3
"""Pinned context-supported target datasets. No generation and no hidden tensors."""
import argparse, hashlib, json, random, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from portability.storage import atomic_json, sha256
ROOT=Path(__file__).resolve().parents[1]
SPECS={'hotpotqa':('hotpotqa/hotpot_qa','distractor','validation','1908d6afbbead072334abe2965f91bd2709910ab'),
       'pubmedqa':('qiaojin/PubMedQA','pqa_labeled','train','9001f2853fb87cab8d220904e0de81ac6973b318')}

def convert(name,row,native):
    if name=='hotpotqa':
        context=row['context']
        text='\n\n'.join(title+'\n'+''.join(sentences) for title,sentences in zip(context['title'],context['sentences']))
        original=str(row['id']);answers=[row['answer']]
    else:
        text='\n\n'.join(row['context']['contexts']);original=str(row['pubid']);answers=[row['final_decision']]
    return dict(dataset=SPECS[name][0],split='test',original_split=native,original_id=original,
                question=row['question'],context=text,gold_answers=answers,is_unanswerable=False,
                evaluation_domain=name,context_sha256=hashlib.sha256(text.encode()).hexdigest())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--samples',type=int,default=1000);ap.add_argument('--seed',type=int,default=42);args=ap.parse_args()
    if not 1<=args.samples<=1000:ap.error('Use 1–1000 examples per dataset; PubMedQA labeled has 1000')
    from datasets import load_dataset
    for name,(repo,config,split,revision) in SPECS.items():
        path=ROOT/'artifacts/distribution_v1'/f'{name}_examples.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        selection=dict(dataset=repo,config=config,revision=revision,native_split=split,seed=args.seed,samples=args.samples,protocol='all_distractor_contexts_or_abstract_v1')
        if path.exists():
            data=json.loads(path.read_text())
            if data['selection']!=selection:raise ValueError('Selection changed; use a separate versioned round')
        else:
            # Streaming bounds download/RAM; reservoir sampling is uniform without loading the dataset.
            rng=random.Random(args.seed);reservoir=[];count=0
            for count,row in enumerate(load_dataset(repo,config,split=split,revision=revision,streaming=True),1):
                if count<=args.samples:reservoir.append(row)
                else:
                    j=rng.randrange(count)
                    if j<args.samples:reservoir[j]=row
            if count<args.samples:raise ValueError('Dataset smaller than requested count')
            examples=sorted([convert(name,r,split) for r in reservoir],key=lambda r:r['original_id'])
            if len({r['original_id'] for r in examples})!=len(examples):raise ValueError('Duplicate native IDs')
            data=dict(selection=selection,native_population=count,examples=examples)
            atomic_json(path,data)
        for generator in ['ministral3_8b','qwen3_8b']:
            cfg=json.loads((ROOT/'configs'/f'stage1_{generator}.json').read_text())
            corpus=f'distribution_v1_{name}_{generator}'
            cfg.update(experiment_id=corpus,dataset=repo,dataset_revision=revision,train_samples=0,dev_samples=0,test_samples=args.samples,
                       examples_file=str(path.relative_to(ROOT)),examples_sha256=sha256(path),record_prompt_truncation=True)
            out=ROOT/'configs'/f'{corpus}.json'
            if out.exists() and json.loads(out.read_text())!=cfg:raise ValueError('Extraction config changed')
            atomic_json(out,cfg)
        print(json.dumps(dict(dataset=name,selected=len(data['examples']),native_population=data['native_population'],manifest=str(path),sha256=sha256(path))),flush=True)
if __name__=='__main__':main()
