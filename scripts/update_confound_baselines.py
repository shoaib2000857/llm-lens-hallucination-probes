#!/usr/bin/env python3
"""Visible-input/output confounds, trained on parent or updated training split only."""
import argparse,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.linear_model import LogisticRegression
from portability.storage import ShardDataset,atomic_json
from portability.probe_training import Corpus
from reproduce_saved_scores import metrics
ROOT=Path(__file__).resolve().parents[1]

def features(corpus):
    ds=ShardDataset(ROOT/'artifacts'/f'stage1_{corpus.name}/hidden_states')
    if [r['sample_id'] for r in ds.records]!=[r['sample_id'] for r in corpus.rows]:raise ValueError('Confound metadata IDs differ')
    values=[]
    for r in ds.records:
        answer=set(re.findall(r'\w+',r['generated_text'].lower()));context=set(re.findall(r'\w+',r['context'].lower()))
        refusal=bool(re.search(r"i (?:cannot|can't|do not know|don't know)|insufficient|not enough|unable to|not provided",r['generated_text'],re.I))
        values.append([len(r['prompt_token_ids']),r['n_gen_tokens'],len(answer&context)/max(len(answer),1),int(refusal),r['mean_token_logprob'],r['min_token_logprob'],r['mean_token_entropy']])
    return np.asarray(values,dtype=float)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--corpora',nargs='+',required=True);args=ap.parse_args()
    parent=Corpus(ROOT,'qwen3_8b');xp=features(parent);source_train=parent.indices('train')
    names=['input_tokens','output_tokens','context_response_overlap','refusal_pattern','mean_logprob','min_logprob','mean_entropy']
    for name in args.corpora:
        target=Corpus(ROOT,name);xt=features(target);test=target.indices('test');reports=[]
        for mode,train,data,y in [('source_fitted',source_train,xp,parent.y),('target_trained',target.indices('train'),xt,target.y)]:
            for signal,cols in [(n,[i]) for i,n in enumerate(names)]+[('combined_visible_confounds',[0,1,2,3])]:
                if len(np.unique(y[train]))<2:reports.append(dict(mode=mode,signal=signal,status='single_class_training'));continue
                mean=data[train][:,cols].mean(0);std=np.maximum(data[train][:,cols].std(0),1e-8)
                model=LogisticRegression(random_state=42,max_iter=500).fit((data[train][:,cols]-mean)/std,y[train]);p=model.predict_proba((xt[test][:,cols]-mean)/std)[:,1]
                reports.append(dict(mode=mode,signal=signal,training_labels=len(train),metrics=metrics(target.y[test],p)))
        out=ROOT/'results/model_updates_v1'/f'{name}__confounds';out.mkdir(parents=True,exist_ok=True);atomic_json(out/'baselines.json',reports)
        print('Saved confidence/confound controls',name,flush=True)
if __name__=='__main__':main()
