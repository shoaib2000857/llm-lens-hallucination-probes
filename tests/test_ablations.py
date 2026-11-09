import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from portability.probe_training import make_probe

class AblationTests(unittest.TestCase):
    def test_positional_controls_preserve_capacity(self):
        torch.set_num_threads(2)
        for variant,base,buffer in [('M2_no_positions','M2','pos_enc'),('M3_no_layer_positions','M3','pos_enc_global'),('M4_no_token_positions','M4','pos_enc')]:
            torch.manual_seed(42);reference=make_probe(base,(4,6,32))
            torch.manual_seed(42);net=make_probe(variant,(4,6,32))
            self.assertEqual(sum(p.numel() for p in reference.parameters()),sum(p.numel() for p in net.parameters()))
            for (n,a),(m,b) in zip(reference.named_parameters(),net.named_parameters()):self.assertEqual(n,m);self.assertTrue(torch.equal(a,b))
            self.assertEqual(int(torch.count_nonzero(getattr(net,buffer))),0)
            with torch.inference_mode():self.assertTrue(torch.isfinite(net.eval()(torch.randn(2,4,6,32))).all())
    def test_capacity_and_uniform_controls(self):
        reference=make_probe('M2',(34,96,4096));net=make_probe('capacity_global',(34,96,4096))
        self.assertLess(abs(sum(p.numel() for p in net.parameters())/sum(p.numel() for p in reference.parameters())-1),.01)
        uniform=make_probe('M1_uniform',(4,6,32)).eval();x=torch.randn(2,4,32)
        with torch.inference_mode():self.assertTrue(torch.allclose(uniform(x),uniform(x.flip(1)),atol=1e-6))
if __name__=='__main__':unittest.main()
