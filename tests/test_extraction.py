import types
import unittest
import numpy as np
import torch
from portability.extraction import extract_batch


class FakeTokenizer:
    pad_token_id=0
    eos_token_id=2
    def __call__(self, prompts, **kwargs):
        return {'input_ids':torch.tensor([[1,4,5],[0,1,6]]), 'attention_mask':torch.tensor([[1,1,1],[0,1,1]])}
    def decode(self, ids, **kwargs):return str(ids)


class FakeModel:
    device='cpu'
    def generate(self, **kwargs):
        hs=[]
        for t in range(3):
            # First generation step includes entire prompt, later cache steps have one position.
            size=3 if t==0 else 1
            hs.append(tuple(torch.full((2,size,4),float(t*10+l)) for l in range(3)))
        return types.SimpleNamespace(sequences=torch.tensor([[1,4,5,7,2,0],[0,1,6,8,9,2]]),
            hidden_states=hs,logits=[torch.zeros(2,16) for _ in range(3)])


class ExtractionTest(unittest.TestCase):
    def test_prediction_positions_eos_padding(self):
        rows=extract_batch(FakeModel(),FakeTokenizer(),['a','b'],max_new=4)
        self.assertEqual(rows[0]['n_gen_tokens'],2)
        self.assertEqual(rows[1]['n_gen_tokens'],3)
        self.assertEqual(rows[0]['prompt_token_ids'],[1,4,5])
        self.assertEqual(rows[1]['prompt_token_ids'],[1,6])
        self.assertEqual(rows[0]['hidden'].shape,(2,4,4))
        self.assertTrue((rows[0]['hidden'][:,2:]==0).all())
        self.assertTrue((rows[1]['hidden'][:,3:]==0).all())
        self.assertEqual(rows[0]['hidden'][0,0,0],1) # excludes embedding layer
        self.assertEqual(rows[0]['hidden'][1,1,0],12)
        self.assertAlmostEqual(rows[0]['mean_token_logprob'],-np.log(16),places=6)


if __name__=='__main__':unittest.main()
