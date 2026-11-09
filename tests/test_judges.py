import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('judge',Path(__file__).resolve().parents[1]/'scripts/judge_representations.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ParseTest(unittest.TestCase):
 def test_valid(self):
  self.assertEqual(m.parse_verdict('{"verdict":0,"reason":"ok"}')[0],0)
  self.assertEqual(m.parse_verdict('```json\n{"verdict":1}\n```')[0],1)
 def test_invalid_is_not_label(self):
  for raw in ['','maybe','{"verdict":true}','{"verdict":"1"}','{"verdict":2}','{"verdict":','{"reason":"wrong"}']:
   self.assertIsNone(m.parse_verdict(raw)[0])

class AgreementTest(unittest.TestCase):
 def test_consensus_requires_three_valid_and_uses_ids(self):
  import tempfile,json,contextlib,io
  spec=importlib.util.spec_from_file_location('agreement',Path(__file__).resolve().parents[1]/'scripts/analyze_judges.py')
  a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
  with tempfile.TemporaryDirectory() as tmp:
   a.ROOT=Path(tmp)
   for corpus in ['ministral3_8b','qwen3_8b']:
    for judge,labels in [('mistral',[0,1,0]),('qwen',[0,1,None]),('phi',[1,1,1])]:
     folder=a.ROOT/'artifacts'/f'stage1_{corpus}'/f'judge_{judge}_v3';folder.mkdir(parents=True)
     rows=[dict(sample_id=str(i),label=v) for i,v in enumerate(labels)]
     (folder/'batch_000000.json').write_text(json.dumps(list(reversed(rows)) if judge=='phi' else rows))
   with contextlib.redirect_stdout(io.StringIO()):a.main([])
   rows=json.loads((a.ROOT/'results/stage2/qwen3_8b/consensus.json').read_text())
   self.assertEqual([r['label'] for r in rows],[0,1,None])
   report=json.loads((a.ROOT/'results/stage2/qwen3_8b/agreement.json').read_text())
   self.assertAlmostEqual(report['fleiss_kappa'],.25)

if __name__=='__main__':unittest.main()
