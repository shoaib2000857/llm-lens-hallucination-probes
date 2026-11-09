#!/usr/bin/env python3
"""Pinned modern same-family variants; same saved QA examples and fixed prompts."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from portability.storage import atomic_json,sha256
VARIANTS={'qwen3_8b':('Qwen/Qwen3-8B-Base','49e3418fbbbca6ecbdf9608b4d22e5a407081db4'),'ministral3_8b':('mistralai/Ministral-3-8B-Instruct-2512','5b26027e7b19eeb4b7352e1fed3926375dd2cb4d')}
def main():
    folder=ROOT/'artifacts/samefamily_v2';folder.mkdir(exist_ok=True);corpora=[];selections=[]
    for generator,(model,revision) in VARIANTS.items():
        for domain in ['squad','hotpotqa','pubmedqa','nq_retrieval','triviaqa']:
            if domain=='squad':path=ROOT/'artifacts'/f'stage1_{generator}/examples.json'
            elif domain in ['hotpotqa','pubmedqa']:path=ROOT/'artifacts/distribution_v1'/f'{domain}_examples.json'
            else:path=ROOT/'artifacts/distribution_v2'/f'{domain}_examples.json'
            original=json.loads(path.read_text());examples=original['examples']
            if sum(r['split']=='test' for r in examples)!=1000:raise ValueError('Expected1000 held-out examples per setting')
            selection=dict(source_examples_sha256=sha256(path),source_corpus=generator,domain=domain,prompt_protocol='unchanged plain source prompt, greedy96-token decoding; no native-chat format adaptation',nq_note='NQ10-document retrieval, not context-free NQ-Open')
            # Reuse original manifests by reference, avoiding dataset duplication.
            selection.update(source_path=str(path.relative_to(ROOT)),sample_count=len(examples));selections.append(selection)
            cfg=json.loads((ROOT/'configs'/f'stage1_{generator}.json').read_text());corpus=f'samefamily_v2_{domain}_{generator}'
            cfg.update(experiment_id=corpus,model=model,model_revision=revision,source_corpus=generator,dataset=examples[0]['dataset'],dataset_revision=original.get('selection',{}).get('revision',cfg['dataset_revision']),train_samples=sum(r['split']=='train' for r in examples),dev_samples=sum(r['split']=='dev' for r in examples),test_samples=1000,examples_file=str(path.relative_to(ROOT)),examples_sha256=sha256(path),record_prompt_truncation=True,portability_axis='same-family frozen parameter transfer')
            dest=ROOT/'configs'/f'{corpus}.json'
            if dest.exists() and json.loads(dest.read_text())!=cfg:raise ValueError('Variant config changed')
            atomic_json(dest,cfg);corpora.append(corpus)
    atomic_json(folder/'plan.json',dict(corpora=corpora,variants=VARIANTS,selections=selections,samples_total=22000,test_samples_total=10000))
    print(json.dumps(dict(corpora=corpora,total_samples=22000,heldout_cases=10000)),flush=True)
if __name__=='__main__':main()
