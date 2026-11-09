import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('operating_report',Path(__file__).resolve().parents[1]/'scripts/portability_reference_report.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class OperatingTests(unittest.TestCase):
 def test_deployment_confusion_counts(self):
  r=module.operating([0,0,1,1],[.1,.8,.4,.9],.5)
  self.assertEqual([r[k] for k in ['tp','fp','fn','tn']],[1,1,1,1])
  for key in ['fpr','tpr','fnr','precision','accuracy']:self.assertEqual(r[key],.5)
 def test_ties_and_empty_positive_predictions(self):
  self.assertEqual(module.operating([0,1],[.5,.5],.5)['tp'],1)
  r=module.operating([0,1],[.1,.9],1.01)
  self.assertIsNone(r['precision']);self.assertEqual(r['fnr'],1.)
