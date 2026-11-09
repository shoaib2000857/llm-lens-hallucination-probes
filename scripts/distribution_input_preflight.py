#!/usr/bin/env python3
"""Tokenizer-only length check for pinned, immutable target manifests."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from portability.storage import atomic_json
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
report=[]
for model,revision in [('mistralai/Mistral-7B-Instruct-v0.2','63a8b081895390a26e140280378bc85ec8bce07a'),('Qwen/Qwen3-8B','b968826d9c46dd6066d109eabc6255188de91218'),('microsoft/Phi-4-mini-instruct','cfbefacb99257ffa30c83adab238a50856ac3083')]:
    tok=AutoTokenizer.from_pretrained(model,revision=revision)
    for name in ['hotpotqa','pubmedqa']:
        rows=json.loads((ROOT/'artifacts/distribution_v1'/f'{name}_examples.json').read_text())['examples']
        # Reserve 512 tokens for judge instructions, chat framing and the 96-token response.
        # This conservative proxy is followed by exact runtime checks of the rendered prompt.
        lengths=[len(tok.encode(r['context']+'\n'+r['question'])) for r in rows]
        rec=dict(model=model,dataset=name,n=len(rows),max_context_question_tokens=max(lengths),judge_budget_reserve=512,
                 conservative_over_8192=sum(n+512>8192 for n in lengths),generator_context_over_2048=sum(n+24>2048 for n in lengths))
        print(json.dumps(rec),flush=True);report.append(rec)
        if rec['conservative_over_8192']:raise ValueError('Full contexts require a larger explicitly recorded judge capacity')
out=ROOT/'results/distribution_v1';out.mkdir(parents=True,exist_ok=True);atomic_json(out/'input_preflight.json',report)
