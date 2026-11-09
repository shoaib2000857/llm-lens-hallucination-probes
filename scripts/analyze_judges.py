#!/usr/bin/env python3
"""Metadata-only agreement and conservative three-judge consensus. No fabricated labels."""
import itertools
import argparse
import json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from portability.storage import atomic_json
from sklearn.metrics import cohen_kappa_score
ROOT=Path(__file__).resolve().parents[1]

def main(argv=None):
 ap=argparse.ArgumentParser();ap.add_argument('--corpora',nargs='+',default=['ministral3_8b','qwen3_8b']);args=ap.parse_args(argv)
 for corpus in args.corpora:
  base=ROOT/'artifacts'/f'stage1_{corpus}'
  judges={};status_counts={}
  for name in ['mistral','qwen','phi']:
   rows=[r for p in sorted((base/f'judge_{name}_v3').glob('batch_*.json')) for r in json.loads(p.read_text())]
   status_counts[name]=dict(Counter(r.get('status','legacy_unknown') for r in rows))
   judges[name]={r['sample_id']:r['label'] for r in rows}
   if len(judges[name])!=len(rows):raise ValueError('Duplicate sample IDs')
  ids=sorted(set().union(*(set(x) for x in judges.values())))
  consensus=[]
  for sid in ids:
   votes={j:rows.get(sid) for j,rows in judges.items()}
   complete=all(v in [0,1] for v in votes.values())
   consensus.append(dict(sample_id=sid,labels=votes,label=int(sum(votes.values())>=2) if complete else None,complete=complete,disagreement=len(set(votes.values()))>1 if complete else None))
  pairs={}
  for a,b in itertools.combinations(judges,2):
   both=[sid for sid in ids if judges[a].get(sid) in [0,1] and judges[b].get(sid) in [0,1]]
   x=[judges[a][i] for i in both];y=[judges[b][i] for i in both]
   k=cohen_kappa_score(x,y) if both and len(set(x+y))>1 else None
   pairs[f'{a}__{b}']=dict(n=len(both),agreement=sum(u==v for u,v in zip(x,y))/len(both) if both else None,kappa=float(k) if k is not None else None)
  valid=[r for r in consensus if r['complete']]
  n=len(valid);fleiss=None
  if n:
   counts=[sum(r['labels'].values()) for r in valid]
   observed=sum((k*(k-1)+(3-k)*(2-k))/6 for k in counts)/n
   p=sum(counts)/(3*n);expected=p*p+(1-p)**2
   if expected<1:fleiss=(observed-expected)/(1-expected)
  out=ROOT/'results/stage2'/corpus;out.mkdir(parents=True,exist_ok=True)
  atomic_json(out/'consensus.json',consensus)
  report=dict(corpus=corpus,n_samples=len(ids),complete_three_judge=n,consensus_positive=sum(r['label']==1 for r in valid),consensus_negative=sum(r['label']==0 for r in valid),invalid_or_missing=len(ids)-n,disagreement_rate=sum(r['disagreement'] for r in valid)/n if n else None,fleiss_kappa=fleiss,pairs=pairs,judges={j:dict(total=len(v),positive=sum(x==1 for x in v.values()),negative=sum(x==0 for x in v.values()),invalid=sum(x is None for x in v.values())) for j,v in judges.items()},caveat='Judge agreement is not human factuality validation. Qwen judge shares generator family. Only all-three-valid rows receive primary consensus.')
  report['judge_status_counts']=status_counts
  atomic_json(out/'agreement.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
