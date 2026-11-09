#!/usr/bin/env python3
"""Full target-setting SQuAD references with original6k labeled train/dev budget."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from portability.probe_training import Corpus,train,POOLED,RAW
from run_probe_suite import evaluate
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--corpus',required=True);a=ap.parse_args();torch.set_num_threads(8)
    print('RESOURCE PLAN:120 maximum fits for this target corpus:8 probes x3 seeds x5 scratch/warm budget settings. Full scratch reference plus100/500-label scratch and warm comparisons; original representations reused. Raw batch8,pooled64,one GPU,8threads,RAM<16GiB,reserve5GiB checkpoints; atomic epoch resume. Small-budget fits add roughly1-3hours/corpus, early stopping dependent.',flush=True)
    source=json.loads((ROOT/'artifacts'/f'stage1_{a.corpus}/config.json').read_text())['source_corpus']
    for probe in POOLED+RAW:
        data=Corpus(ROOT,a.corpus,raw=probe in RAW)
        if not len(data.indices('train')):raise ValueError('Full-setting references require target training split')
        for seed in [42,43,44]:
            for budget,mode in [('full','scratch'),('100','scratch'),('500','scratch'),('100','warm'),('500','warm')]:
                out=ROOT/'artifacts/probes'/f'{a.corpus}__{probe}__s{seed}__{mode}_{budget}';parent=ROOT/'artifacts/probes'/f'{source}__{probe}__s{seed}__scratch_full' if mode=='warm' else None
                net=train(data,probe,budget,seed,out,torch.device('cuda'),epochs=25,patience=4,batch=8 if probe in RAW else 64,source=parent)
                if net is None:print('Single-class budget skipped:'+out.name,flush=True);continue
                if not (out/'evaluation.json').exists():evaluate(net,data,torch.device('cuda'),8 if probe in RAW else 64,out,'full_target_setting_reference' if budget=='full' else f'same_family_{mode}_recovery',seed,500)
                del net;torch.cuda.empty_cache()
if __name__=='__main__':main()
