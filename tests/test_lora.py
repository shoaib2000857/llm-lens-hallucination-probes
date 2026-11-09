import unittest
import torch
from torch import nn
from portability.lora import attach,adapter_state,restore
class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.q_proj=nn.Linear(6,8);self.v_proj=nn.Linear(6,4)
    def forward(self,x):return self.q_proj(x).sum()+self.v_proj(x).sum()
class LoRATest(unittest.TestCase):
    def test_identity_gradients_and_restore(self):
        torch.manual_seed(42);model=Toy();model.requires_grad_(False);x=torch.randn(3,6);before=model(x).detach()
        self.assertEqual(attach(model,2,4),2);self.assertTrue(torch.equal(before,model(x).detach()))
        model(x).backward();self.assertIsNone(model.q_proj.base.weight.grad);self.assertGreater(model.q_proj.B.grad.abs().sum(),0)
        with torch.no_grad():model.q_proj.B.add_(.5)
        state=adapter_state(model);value=model(x).detach()
        with torch.no_grad():model.q_proj.B.zero_()
        restore(model,state);self.assertTrue(torch.equal(value,model(x).detach()))
        state['q_proj']['B']=torch.zeros(1,1)
        with self.assertRaises(ValueError):restore(model,state)
