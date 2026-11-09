#!/usr/bin/env python3
"""Pinned modern NQ retrieval-QA and TriviaQA context-supported expansion."""
import gzip,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from portability.storage import atomic_json,sha256

def record(domain,repo,original,question,context,answers):
    if not context.strip():raise ValueError('Missing authentic context')
    return dict(dataset=repo,split='test',original_split='validation',original_id=str(original),question=question,context=context,gold_answers=answers,is_unanswerable=False,evaluation_domain=domain,context_sha256=hashlib.sha256(context.encode()).hexdigest())

def main():
    folder=ROOT/'artifacts/distribution_v2';folder.mkdir(parents=True,exist_ok=True)
    for domain in ['nq_retrieval','triviaqa']:
        file=folder/f'{domain}_examples.json'
        if not file.exists():
            if domain=='nq_retrieval':
                source=ROOT/'artifacts/prior_methods/Lookback-Lens/data/nq-open-10_total_documents_gold_at_4.jsonl.gz'
                with gzip.open(source,'rt') as f:raw=[json.loads(line) for line in f]
                raw=sorted(raw,key=lambda r:hashlib.sha256(r['question'].encode()).hexdigest())[:1000]
                examples=[record(domain,'Lookback-Lens/NQ-open-10-documents',hashlib.sha256(r['question'].encode()).hexdigest(),r['question'],'\n\n'.join(c['title']+'\n'+c['text'] for c in r['ctxs']),r['answers']) for r in raw]
                selection=dict(source_file_sha256=sha256(source),sampling='question-hash first1000 of supplied evaluation collection',protocol='all10 retrieved documents, original order; not context-free NQ-Open or legacy reproduction',revision=sha256(source))
            else:
                from datasets import load_dataset
                revision='0f7faf33a3908546c6fd5b73a660e0f8ff173c2f';examples=[];seen=set()
                for r in load_dataset('mandarjoshi/trivia_qa','rc.wikipedia',split='validation',revision=revision,streaming=True):
                    if r['question_id'] in seen:continue
                    pages=r['entity_pages'];texts=pages['wiki_context'];titles=pages['title']
                    context='\n\n'.join(title+'\n'+text for title,text in zip(titles,texts) if text.strip())
                    if not context:continue
                    examples.append(record(domain,'mandarjoshi/trivia_qa',r['question_id'],r['question'],context,[r['answer']['value']]+r['answer']['aliases']));seen.add(r['question_id'])
                    if len(examples)==1000:break
                selection=dict(dataset='mandarjoshi/trivia_qa',config='rc.wikipedia',revision=revision,sampling='first1000 unique validation rows with nonempty Wikipedia contexts; not uniform population sample',protocol='all supplied Wikipedia documents in native order; no reference-answer context fabrication')
            if len(examples)!=1000 or len({r['original_id'] for r in examples})!=1000:raise ValueError('Incomplete or duplicate selection')
            atomic_json(file,dict(selection=selection,examples=examples))
        data=json.loads(file.read_text())
        for generator in ['ministral3_8b','qwen3_8b']:
            cfg=json.loads((ROOT/'configs'/f'stage1_{generator}.json').read_text());corpus=f'distribution_v2_{domain}_{generator}'
            cfg.update(experiment_id=corpus,dataset=data['examples'][0]['dataset'],dataset_revision=data['selection']['revision'],train_samples=0,dev_samples=0,test_samples=1000,examples_file=str(file.relative_to(ROOT)),examples_sha256=sha256(file),record_prompt_truncation=True)
            path=ROOT/'configs'/f'{corpus}.json'
            if path.exists() and json.loads(path.read_text())!=cfg:raise ValueError('Config changed')
            atomic_json(path,cfg)
        print(json.dumps(dict(domain=domain,samples=len(data['examples']),protocol=data['selection'])),flush=True)
if __name__=='__main__':main()
