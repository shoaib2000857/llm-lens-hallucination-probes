#!/usr/bin/env python3
"""One sequential pass over original shards; restartable small pooled cache."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from portability.storage import ShardDataset,atomic_json,sha256

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--corpus',required=True);ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
 torch.set_num_threads(int(os.environ.get('POOL_THREADS','16')))
 root=Path(__file__).resolve().parents[1];base=root/'artifacts'/f'stage1_{args.corpus}'
 ds=ShardDataset(base/'hidden_states');labels_path=root/'results/stage2'/args.corpus/'consensus.json'
 labels=json.loads(labels_path.read_text());byid={r['sample_id']:r for r in labels}
 if len(byid)!=len(labels):raise ValueError('Duplicate consensus IDs')
 n=min(args.limit,len(ds)) if args.limit else len(ds)
 if not args.limit:
  if set(byid)!={r['sample_id'] for r in ds.records}:raise ValueError('Consensus IDs do not exactly match representation IDs')
  report=json.loads((labels_path.parent/'agreement.json').read_text())
  if report['n_samples']!=len(ds) or any(j['total']!=len(ds) for j in report['judges'].values()):raise ValueError('Stage 2 is not complete. Do not train on the pilot labels.')
  if report['complete_three_judge']/len(ds)<.7:raise ValueError('Consensus coverage below 70%; inspect invalid judge responses first')
 out=base/('pooled_smoke' if args.limit else 'pooled');out.mkdir(exist_ok=True)
 records=[]
 for row in ds.records[:n]:
  label=byid.get(row['sample_id'])
  records.append({k:row[k] for k in ['sample_id','original_id','split','n_gen_tokens','mean_token_logprob','min_token_logprob','mean_token_entropy'] } | dict(label=label['label'] if label else None,judge_labels=label['labels'] if label else {}))
 cfg=dict(protocol='fixed_T_mean_fp32_v1',source_spec_sha256=sha256(base/'hidden_states/spec.json'),labels_sha256=sha256(labels_path),n=n,source_shape=ds.spec['shape'])
 if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=cfg:raise ValueError('Pooled cache provenance changed; use a new output directory')
 atomic_json(out/'config.json',cfg)
 if (out/'manifest.json').exists():
  manifest=json.loads((out/'manifest.json').read_text())
  if sha256(out/'layer_mean.npy')!=manifest['tensor_sha256']:raise ValueError('Pooled tensor checksum mismatch')
  print('Pooled cache already complete and checksum verified',out);return
 shape=(n,ds.spec['shape'][0],ds.spec['shape'][2]);path=out/'layer_mean.npy'
 start=json.loads((out/'progress.json').read_text())['rows'] if (out/'progress.json').exists() else 0
 a=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float32,shape=shape)
 if a.shape!=shape or a.dtype!=np.float32:raise ValueError('Pooled cache header mismatch')
 t0=time.time()
 for i in range(start,n):
  x,r=ds[i];a[i]=torch.from_numpy(x).mean(dim=1,dtype=torch.float32).numpy()
  if (i+1)%128==0 or i+1==n:
   a.flush();atomic_json(out/'progress.json',dict(rows=i+1,total=n));print(json.dumps(dict(corpus=args.corpus,pooled=i+1,total=n,elapsed=time.time()-t0)),flush=True)
 del a
 atomic_json(out/'manifest.json',dict(complete=True,source_shape=ds.spec['shape'],records=records,pooling=cfg['protocol'],tensor_sha256=sha256(path),label_file_sha256=cfg['labels_sha256']))
if __name__=='__main__':main()
