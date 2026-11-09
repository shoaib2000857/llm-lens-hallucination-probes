"""Lookback attention ratios on saved answer-prediction positions, without full maps."""
import torch

def selected_ratios(query,key,positions,prefix_ends,lengths,attention_mask=None,scaling=None,is_causal=True,sliding_window=None):
    b,h,s,d=query.shape
    if h%key.shape[1]:raise ValueError('Unsupported GQA head counts')
    if key.shape[1]!=h:key=key.repeat_interleave(h//key.shape[1],dim=1)
    q=query.gather(2,positions[:,None,:,None].expand(b,h,-1,d))
    scores=torch.matmul(q.float(),key.float().transpose(-2,-1))*(scaling if scaling is not None else d**-.5)
    keys=torch.arange(key.shape[2],device=query.device)[None,None,None,:]
    if is_causal:scores.masked_fill_(keys>positions[:,None,:,None],float('-inf'))
    if sliding_window:scores.masked_fill_(keys<=positions[:,None,:,None]-sliding_window,float('-inf'))
    if attention_mask is not None:
        mask=attention_mask
        if mask.ndim!=4:raise ValueError('Expected prepared4D SDPA mask')
        if mask.shape[-2]!=1:mask=mask.gather(2,positions[:,None,:,None].expand(b,mask.shape[1],-1,mask.shape[-1]))
        if mask.dtype==torch.bool:scores.masked_fill_(~mask,float('-inf'))
        else:scores+=mask.float()
    weights=scores.softmax(-1)
    # prefix_ends is the first response-header token (question remains in prefix).
    start=prefix_ends[:,0];end=prefix_ends[:,1]
    prefix=(keys>=start[:,None,None,None])&(keys<end[:,None,None,None])
    new=(keys>=end[:,None,None,None])&(keys<=positions[:,None,:,None])
    prefix_mean=(weights*prefix).sum(-1)/(end-start)[:,None,None]
    new_count=positions-end[:,None]+1
    if torch.any(new_count<=0):raise ValueError('Response marker must contain at least one token')
    new_mean=(weights*new).sum(-1)/new_count[:,None,:]
    ratio=prefix_mean/(prefix_mean+new_mean).clamp_min(1e-30)
    valid=torch.arange(positions.shape[1],device=query.device)[None,:]<lengths[:,None]
    if not torch.isfinite(ratio[valid[:,None,:].expand_as(ratio)]).all():raise ValueError('Nonfinite attention ratio')
    return (ratio*valid[:,None,:]).sum(-1)/lengths[:,None]

class RatioCapture:
    def __init__(self):self.context=None;self.layers={}
    def wrap(self,original):
        def forward(module,query,key,value,attention_mask,dropout=0.,scaling=None,is_causal=None,**kwargs):
            result=original(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,is_causal=is_causal,**kwargs)
            if self.context is not None:
                causal=getattr(module,'is_causal',True) if is_causal is None else is_causal
                self.layers[module.layer_idx]=selected_ratios(query,key,**self.context,attention_mask=attention_mask,scaling=scaling,is_causal=causal,sliding_window=kwargs.get('sliding_window'))
            return result
        return forward
