import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from portability.storage import ShardWriter, ShardDataset


class StorageTest(unittest.TestCase):
    def test_resume_alignment_and_lazy(self):
        with tempfile.TemporaryDirectory() as d:
            w = ShardWriter(d, {'model':'fixture'}, (2, 3, 4), rows_per_shard=2)
            for i in range(3):
                w.append(np.full((2,3,4), i, np.float16), dict(dataset='fixture', split='train', original_id=str(i), label=i%2))
            w.flush()
            w2 = ShardWriter(d, {'model':'fixture'}, (2,3,4), rows_per_shard=2)
            self.assertEqual(len(w2.completed),3)
            ds = ShardDataset(d, cache_shards=1, verify=True)
            for i in [2,0,1]:
                x,m = ds[i]; self.assertTrue((x==i).all()); self.assertEqual(m['label'],i%2)
                self.assertLessEqual(len(ds.cache),1)
            with self.assertRaises(ValueError):
                w2.append(np.zeros((2,3,4),np.float16), dict(dataset='fixture',split='train',original_id='0'))
            with self.assertRaises(ValueError): ShardWriter(d, {'model':'other'}, (2,3,4))

    def test_orphan_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            w=ShardWriter(d,{},(1,1,1),rows_per_shard=1)
            Path(d,'shard_000000.npy.tmp').write_bytes(b'incomplete')
            w.append(np.zeros((1,1,1),np.float16),dict(dataset='fixture',split='test',original_id='a'))
            p=next(Path(d).glob('*.npy'))
            with p.open('ab') as f:f.write(b'corrupt')
            with self.assertRaises(ValueError): ShardDataset(d,verify=True)

    def test_shape_dtype(self):
        with tempfile.TemporaryDirectory() as d:
            w=ShardWriter(d,{},(2,3,4))
            with self.assertRaises(ValueError): w.append(np.zeros((2,3,4),np.float32),{})


if __name__=='__main__': unittest.main()
