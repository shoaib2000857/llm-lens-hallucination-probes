#!/usr/bin/env python3
"""Descriptive label rates; does not infer bias without independent human correctness."""
import json,csv
from pathlib import Path
from collections import defaultdict
root=Path(__file__).resolve().parents[1];groups=defaultdict(list)
for corpus in ['ministral3_8b','qwen3_8b']:
 labels={r['sample_id']:r for r in json.loads((root/'results/stage2'/corpus/'consensus.json').read_text())}
 states=root/'artifacts'/f'stage1_{corpus}'/'hidden_states'
 for p in sorted(states.glob('*.commit.json')):
  c=json.loads(p.read_text())
  for r in json.loads((states/c['metadata']).read_text()):
   ll=labels[r['sample_id']];length='capped96' if r['n_gen_tokens']==96 else 'below96'
   for judge,v in ll['labels'].items():
    if v in [0,1]:groups[(corpus,r['split'],judge,bool(r['is_unanswerable']),length)].append(v)
rows=[dict(generator=k[0],split=k[1],judge=k[2],unanswerable=k[3],length_group=k[4],n=len(v),positive_rate=sum(v)/len(v)) for k,v in sorted(groups.items())]
out=root/'results/current_run'
with (out/'judge_family_strata.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Saved',len(rows),'judge/generator/split/answerability/length cells. Human-conditioned bias analysis pending.')
