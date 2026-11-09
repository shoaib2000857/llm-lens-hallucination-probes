"""Explicit source-controlled ablations; original probe definitions remain intact."""
import math
import torch
from torch import nn
from .legacy_probes import LayerWiseMLP,CrossLayerTransformer,HierarchicalTransformer,CrossLayerAttentionTransformerV2

class UniformLayerProbe(nn.Module):
    def __init__(self,dim,layers):
        super().__init__();self.base=LayerWiseMLP(n_layers=layers,hidden_dim=dim);self.base.layer_attn=None
    def forward(self,x):
        b,l,d=x.shape
        features=self.base.layer_proj(x.reshape(b*l,d)).reshape(b,l,-1)
        return self.base.classifier(features.mean(1)).squeeze(-1)

class CapacityGlobalProbe(nn.Module):
    def __init__(self,dim,target_parameters):
        super().__init__();width=int((-(dim+3)+math.sqrt((dim+3)**2+4*(target_parameters-1)))/2)
        self.net=nn.Sequential(nn.Linear(dim,width),nn.GELU(),nn.Dropout(.3),nn.Linear(width,width),nn.GELU(),nn.Dropout(.3),nn.Linear(width,1))
    def forward(self,x):return self.net(x.mean(1)).squeeze(-1)

NAMES=('M1_uniform','M2_no_positions','M3_no_layer_positions','M4_no_token_positions','capacity_global')
def build(name,shape):
    layers,tokens,dim=shape
    if name=='M1_uniform':return UniformLayerProbe(dim,layers)
    if name=='capacity_global':
        reference=CrossLayerTransformer(n_layers=layers,t_fixed=tokens,hidden_dim=dim);count=sum(p.numel() for p in reference.parameters());del reference
        return CapacityGlobalProbe(dim,count)
    cls={'M2_no_positions':CrossLayerTransformer,'M3_no_layer_positions':HierarchicalTransformer,'M4_no_token_positions':CrossLayerAttentionTransformerV2}[name]
    net=cls(n_layers=layers,t_fixed=tokens,hidden_dim=dim)
    buffer='pos_enc_global' if name=='M3_no_layer_positions' else 'pos_enc'
    getattr(net,buffer).zero_()
    return net
