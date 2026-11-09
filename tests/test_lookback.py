import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from portability.lookback import selected_ratios
class LookbackTests(unittest.TestCase):
 def test_exact_causal_formula_and_gqa(self):
  torch.manual_seed(4);q=torch.randn(2,4,9,3);k=torch.randn(2,2,9,3);p=torch.tensor([[4,5,6],[5,6,6]]);bounds=torch.tensor([[0,3],[1,4]]);lengths=torch.tensor([3,2]);mask=torch.zeros(2,1,9,9);mask[1,:,:,0]=-float('inf')
  actual=selected_ratios(q,k,p,bounds,lengths,mask)
  full=q@k.repeat_interleave(2,1).transpose(-2,-1)/(3**.5);full+=mask;full.masked_fill_(torch.arange(9)[None,None,None,:]>torch.arange(9)[None,None,:,None],-float('inf'));a=full.softmax(-1);expected=[]
  for b in range(2):
   values=[];lo,hi=bounds[b]
   for pos in p[b,:lengths[b]]:
    c=a[b,:,pos,lo:hi].mean(-1);n=a[b,:,pos,hi:pos+1].mean(-1);values.append(c/(c+n))
   expected.append(torch.stack(values).mean(0))
  torch.testing.assert_close(actual,torch.stack(expected),atol=1e-6,rtol=1e-6)
 def test_invalid_empty_new_partition(self):
  with self.assertRaises(ValueError):selected_ratios(torch.randn(1,1,3,2),torch.randn(1,1,3,2),torch.tensor([[1]]),torch.tensor([[0,2]]),torch.tensor([1]))
