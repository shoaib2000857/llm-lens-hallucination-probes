"""Small explicit LoRA implementation; no duplicated parent checkpoints."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class LoRALinear(nn.Module):
    def __init__(self,base,rank,alpha):
        super().__init__();self.base=base;self.rank=rank;self.alpha=alpha
        base.requires_grad_(False)
        self.A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
    def forward(self,x):
        delta=F.linear(F.linear(x.float(),self.A),self.B)*(self.alpha/self.rank)
        return self.base(x)+delta.to(x.dtype)

def attach(model,rank=8,alpha=16):
    selected=[(n,m) for n,m in model.named_modules() if isinstance(m,nn.Linear) and n.rsplit('.',1)[-1] in ['q_proj','v_proj']]
    if not selected:raise ValueError('No compatible query/value projections')
    for name,base in selected:
        parent,_,key=name.rpartition('.');setattr(model.get_submodule(parent) if parent else model,key,LoRALinear(base,rank,alpha))
    return len(selected)

def adapter_state(model):
    return {name:{'A':m.A.detach().cpu().clone(),'B':m.B.detach().cpu().clone()} for name,m in model.named_modules() if isinstance(m,LoRALinear)}

def restore(model,state):
    modules={n:m for n,m in model.named_modules() if isinstance(m,LoRALinear)}
    if modules.keys()!=state.keys():raise ValueError('Adapter module names incompatible')
    with torch.no_grad():
        for name,m in modules.items():
            for key in ['A','B']:
                p=getattr(m,key);v=state[name][key]
                if p.shape!=v.shape:raise ValueError('Adapter shape mismatch')
                p.copy_(v)

def load_adapter(model,path,expected_model,expected_revision):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['base_model']!=expected_model or payload['base_revision']!=expected_revision:raise ValueError('Adapter belongs to another parent checkpoint')
    attach(model,payload['rank'],payload['alpha']);restore(model,payload['modules'])
    return payload
