#!/usr/bin/env python3
"""Disjoint supervised adaptation examples and immutable source replay cohort."""
import argparse,hashlib,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from portability.storage import atomic_json,sha256
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=ROOT/'configs/model_updates_v1.json');args=ap.parse_args()
    cfg=json.loads(args.config.read_text());out=ROOT/'artifacts/model_updates_v1';out.mkdir(exist_ok=True)
    exclusion_paths=[ROOT/'artifacts/stage1_qwen3_8b/examples.json',ROOT/'artifacts/distribution_v1/hotpotqa_examples.json',ROOT/'artifacts/distribution_v1/pubmedqa_examples.json']
    excluded_questions=set();excluded_contexts=set()
    for path in exclusion_paths:
        for r in json.loads(path.read_text())['examples']:
            excluded_questions.add(' '.join(r['question'].lower().split()));excluded_contexts.add(hashlib.sha256(r['context'].encode()).hexdigest())
    provenance=dict(config=cfg,exclusion_manifests={str(p.relative_to(ROOT)):sha256(p) for p in exclusion_paths},protocol='hotpot_gold_support_training_only_answer_loss_v1')
    path=out/'adaptation_examples.json'
    if path.exists():
        if json.loads(path.read_text())['provenance']!=provenance:raise ValueError('Adaptation manifest changed')
        print('Adaptation manifest already complete',flush=True);return
    from datasets import load_dataset
    rng=random.Random(cfg['seed']);reservoir=[];eligible=0;seen=set();excluded=0;want=cfg['train_samples']+cfg['dev_samples']
    for row in load_dataset(cfg['dataset'],cfg['dataset_config'],split='train',revision=cfg['dataset_revision'],streaming=True):
        question=' '.join(row['question'].lower().split());full='\n\n'.join(t+'\n'+''.join(s) for t,s in zip(row['context']['title'],row['context']['sentences']))
        support=set(row['supporting_facts']['title']);context='\n\n'.join(t+'\n'+''.join(s) for t,s in zip(row['context']['title'],row['context']['sentences']) if t in support)
        group=hashlib.sha256(context.encode()).hexdigest()
        if question in excluded_questions or hashlib.sha256(full.encode()).hexdigest() in excluded_contexts or group in excluded_contexts or group in seen:
            excluded+=1;continue
        if not context or not row['answer']:raise ValueError('Empty supervised context/answer')
        seen.add(group);eligible+=1
        item=dict(original_id=row['id'],question=row['question'],context=context,answer=row['answer'],context_sha256=group)
        if len(reservoir)<want:reservoir.append(item)
        else:
            j=rng.randrange(eligible)
            if j<want:reservoir[j]=item
    if len(reservoir)!=want:raise ValueError('Insufficient independent adaptation examples')
    rng.shuffle(reservoir)
    for i,r in enumerate(reservoir):r['split']='train' if i<cfg['train_samples'] else 'dev'
    atomic_json(path,dict(provenance=provenance,examples=reservoir,eligible=eligible,excluded=excluded))
    print(json.dumps(dict(train=cfg['train_samples'],dev=cfg['dev_samples'],eligible=eligible,excluded=excluded)),flush=True)
if __name__=='__main__':main()
