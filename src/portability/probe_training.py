"""New-round training; explicit pooled features, budgets and positional-buffer handling."""
import csv, json, os, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .legacy_probes import ProbeMLP, LayerWiseMLP, CrossLayerTransformer, HierarchicalTransformer, CrossLayerAttentionTransformerV2
from .storage import atomic_json, ShardDataset, sha256

POOLED=('mean_linear','last_linear','last_mlp','M0','M1')
RAW=('M2','M3','M4')

class PooledProbe(nn.Module):
    def __init__(self,name,dim,layers):
        super().__init__();self.name=name
        if name=='M0':self.net=ProbeMLP(hidden_dim=dim)
        elif name=='M1':self.net=LayerWiseMLP(n_layers=layers,hidden_dim=dim)
        elif name=='last_mlp':self.net=nn.Sequential(nn.Linear(dim,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.3),nn.Linear(256,1))
        else:self.net=nn.Linear(dim,1)
    def forward(self,x):
        if self.name in ('M0','M1'):return self.net(x.unsqueeze(2))
        return self.net(x[:,-1] if self.name.startswith('last_') else x.mean(1)).squeeze(-1)

def make_probe(name,shape):
    l,t,d=shape
    from .ablations import NAMES,build
    if name in NAMES:return build(name,shape)
    if name in POOLED:return PooledProbe(name,d,l)
    return {'M2':CrossLayerTransformer,'M3':HierarchicalTransformer,'M4':CrossLayerAttentionTransformerV2}[name](n_layers=l,t_fixed=t,hidden_dim=d)

def warm_parameters(target,source):
    """Copy learned parameters only. Target positional buffers keep native layer count."""
    a=dict(target.named_parameters());b=dict(source.named_parameters())
    if a.keys()!=b.keys():raise ValueError('Different architecture parameter keys')
    with torch.no_grad():
        for name,p in a.items():
            if p.shape!=b[name].shape:raise ValueError(f'Incompatible parameter {name}: {p.shape} vs {b[name].shape}')
            p.copy_(b[name])

def depth_map(x,layers):
    """Explicit linear interpolation along normalized layer depth; NOT direct transfer."""
    if x.shape[1]==layers:return x
    b,l,t,d=x.shape
    y=x.permute(0,2,3,1).reshape(b*t*d,1,l)
    y=torch.nn.functional.interpolate(y,size=layers,mode='linear',align_corners=True)
    return y.reshape(b,t,d,layers).permute(0,3,1,2).contiguous()

class Corpus:
    def __init__(self,root,name,raw=False):
        self.name=name;self.root=Path(root);base=self.root/'artifacts'/f'stage1_{name}'
        self.manifest=json.loads((base/'pooled/manifest.json').read_text())
        if not self.manifest['complete']:raise ValueError('Incomplete pooled cache')
        self.rows=self.manifest['records'];self.shape=self.manifest['source_shape']
        self.array=np.load(base/'pooled/layer_mean.npy',mmap_mode='r')
        if self.array.shape!=(len(self.rows),self.shape[0],self.shape[2]):raise ValueError('Pooled shape mismatch')
        if len({r['sample_id'] for r in self.rows})!=len(self.rows):raise ValueError('Duplicate feature IDs')
        if any(r['label'] not in (0,1,None) for r in self.rows):raise ValueError('Invalid consensus label')
        self.raw=ShardDataset(base/'hidden_states') if raw else None
        if self.raw and [r['sample_id'] for r in self.raw.records]!=[r['sample_id'] for r in self.rows]:raise ValueError('Raw/pooled ID mismatch')
        self.y=np.array([r['label'] if r['label'] is not None else -1 for r in self.rows],dtype=np.float32)
    def indices(self,split):return np.array([i for i,r in enumerate(self.rows) if r['split']==split and r['label'] is not None],dtype=int)
    def batch(self,idx,device):
        if self.raw:
            x=torch.empty((len(idx),*self.shape),dtype=torch.float16,pin_memory=device.type=='cuda')
            self.raw.read_batch(idx,out=x.numpy())
        else:
            x=torch.from_numpy(np.array(self.array[idx],copy=True))
            if device.type=='cuda':x=x.pin_memory()
        if device.type=='cpu':x=x.float()
        return x.to(device,non_blocking=True),torch.as_tensor(self.y[idx],device=device)

def budget_split(corpus,budget,seed):
    tr=corpus.indices('train');rng=np.random.default_rng(seed);tr=rng.permutation(tr)
    if budget=='full':return tr,corpus.indices('dev')
    n=int(budget)
    if n>len(tr):raise ValueError('Budget exceeds valid training population')
    chosen=tr[:n];nval=max(2,n//5)
    # Whole selected budget includes labels spent on early stopping.
    return chosen[nval:],chosen[:nval]

def epoch_order(idx,seed,raw):
    rng=np.random.default_rng(seed)
    if not raw:return rng.permutation(idx)
    # Shuffle contiguous 16-row blocks, then rows within blocks, for bounded I/O.
    groups={}
    for i in idx:groups.setdefault(int(i)//16,[]).append(i)
    keys=list(groups);rng.shuffle(keys)
    return np.concatenate([rng.permutation(groups[k]) for k in keys])

def atomic_torch(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def predict(net,corpus,idx,device,batch,source_layers=None):
    net.eval();out=[]
    with torch.inference_mode():
        for start in range(0,len(idx),batch):
            x,_=corpus.batch(idx[start:start+batch],device)
            if source_layers is not None:x=depth_map(x,source_layers)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                logits=net(x)
            out.extend(logits.float().sigmoid().cpu().tolist())
    return np.asarray(out)

def registry(corpus,out,config,status,seconds=0):
    path=corpus.root/'experiments/registry.csv';path.parent.mkdir(parents=True,exist_ok=True)
    fields='experiment_id research_question source_model target_model dataset probe training_mode adaptation seed sample_count status start_time end_time gpu_hours output_path notes'.split()
    rows=[]
    if path.exists():
        with path.open(newline='') as f:rows=list(csv.DictReader(f))
    old=next((r for r in rows if r['experiment_id']==out.name),{})
    now=datetime.now(timezone.utc).isoformat()
    row={k:'' for k in fields};row.update(old)
    dataset='hotpotqa' if 'hotpotqa' in corpus.name else 'pubmedqa' if 'pubmedqa' in corpus.name else 'squad_v2'
    target_config=corpus.root/'artifacts'/f'stage1_{corpus.name}/config.json'
    if target_config.exists():dataset=json.loads(target_config.read_text()).get('dataset',dataset)
    row.update(experiment_id=out.name,research_question='Architecture and parameter portability; target-label efficiency',source_model=config['source'] or corpus.name,target_model=corpus.name,dataset=dataset,probe=config['probe'],training_mode='full_tensor' if corpus.raw else 'pooled_cache',adaptation='warm' if config['source'] else 'scratch',seed=config['seed'],sample_count=config['total_target_labels'],status=status,start_time=old.get('start_time') or now,end_time=now if status in ('complete','skipped') else '',gpu_hours=seconds/3600,output_path=str(out.relative_to(corpus.root)),notes='Label budget includes early-stopping labels; test held out.')
    if 'distribution_v1' in corpus.name:
        row['research_question']='Distribution portability; matched target-trained and domain warm-start controls'
        if config['total_target_labels']==0:
            row.update(adaptation='none',sample_count=len(corpus.indices('test')),notes='Frozen same-generator distribution transfer; zero target training labels.')
        elif '__cv' in out.name:
            row['notes']='Context-grouped cross-validation fold; target budget includes development labels. OOF result, not independent final test.'
    if corpus.name.startswith('updates_v1'):
        row['research_question']='Post-deployment parameter portability and target-label recovery'
        if config['total_target_labels']==0:
            row.update(adaptation='none',sample_count=len(corpus.indices('test')),notes='Frozen deployment probe on same-parent LoRA update; new response labels; held-out original test questions.')
    rows=[r for r in rows if r['experiment_id']!=out.name]+[row]
    tmp=path.with_suffix('.csv.tmp')
    with tmp.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
    os.replace(tmp,path)

def train(corpus,name,budget,seed,out,device,epochs=25,patience=4,batch=64,source=None,lr=2e-4):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    tr,va=budget_split(corpus,budget,seed)
    config=dict(corpus=corpus.name,probe=name,shape=corpus.shape,budget=budget,seed=seed,epochs=epochs,patience=patience,batch=batch,lr=lr,
                source=str(source) if source else None,train_ids=[corpus.rows[i]['sample_id'] for i in tr],validation_ids=[corpus.rows[i]['sample_id'] for i in va],
                train_labels=len(tr),validation_labels=len(va),total_target_labels=len(tr)+len(va),loss='train-weighted BCE; dev unweighted Brier',pooling='fixed-T mean including zero padding' if not corpus.raw else 'original full tensors',warm_start='learned parameters only; native positional buffers' if source else None)
    cp=out/'config.json'
    if ('distribution_v1' in corpus.name or corpus.name.startswith(('updates_v1','samefamily_'))) and source:
        config['source_checkpoint_sha256']=sha256(Path(source)/'best.pt')
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError(f'Config mismatch: {out}')
    atomic_json(cp,config)
    if len(np.unique(corpus.y[tr]))<2:
        atomic_json(out/'skipped.json',dict(reason='single_class_training_budget'));registry(corpus,out,config,'skipped');return None
    torch.manual_seed(seed);net=make_probe(name,corpus.shape).to(device)
    if source:
        ck=torch.load(Path(source)/'best.pt',map_location='cpu',weights_only=True)
        src=make_probe(name,ck['shape']);src.load_state_dict(ck['model']);warm_parameters(net,src);del src
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    begin=0;best=float('inf');stale=0;history=[]
    if (out/'complete.json').exists():
        ck=torch.load(out/'best.pt',map_location='cpu',weights_only=True);net.load_state_dict(ck['model']);return net
    if (out/'last.pt').exists():
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
        net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
        begin=ck['epoch']+1;best=ck['best'];stale=ck['stale'];history=ck['history'];torch.set_rng_state(ck['rng'])
        if device.type=='cuda':torch.cuda.set_rng_state(ck['cuda_rng'])
    positives=float(corpus.y[tr].sum());weight=torch.tensor((len(tr)-positives)/positives,device=device)
    t0=time.time();registry(corpus,out,config,'running')
    for epoch in range(begin,epochs):
        if stale>=patience:break
        net.train();order=epoch_order(tr,seed+epoch,bool(corpus.raw));loss_sum=0
        for start in range(0,len(order),batch):
            idx=order[start:start+batch];x,y=corpus.batch(idx,device);opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                logits=net(x);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),y,pos_weight=weight)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step();loss_sum+=float(loss.detach())*len(idx)
        pv=predict(net,corpus,va,device,batch);score=float(np.mean((pv-corpus.y[va])**2));sched.step()
        if not np.isfinite(score):raise ValueError('Invalid validation score')
        improved=score<best
        if improved:best=score;stale=0
        else:stale+=1
        history.append(dict(epoch=epoch,train_loss=loss_sum/len(tr),validation_brier=score,elapsed_seconds=time.time()-t0))
        state={k:v.detach().cpu() for k,v in net.state_dict().items()}
        if improved:atomic_torch(out/'best.pt',dict(model=state,shape=corpus.shape,probe=name,epoch=epoch))
        atomic_torch(out/'last.pt',dict(model=state,shape=corpus.shape,optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=epoch,best=best,stale=stale,history=history,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if device.type=='cuda' else torch.get_rng_state()))
        atomic_json(out/'history.json',history);print(json.dumps(dict(experiment=out.name,**history[-1])),flush=True)
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=True);net.load_state_dict(ck['model'])
    atomic_json(out/'complete.json',dict(best_epoch=ck['epoch'],epochs=len(history),parameters=sum(p.numel() for p in net.parameters()),seconds_this_invocation=time.time()-t0))
    registry(corpus,out,config,'complete',time.time()-t0 if device.type=='cuda' else 0)
    return net
