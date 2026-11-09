"""Annotation ownership/locking/adjudication checks in isolated temporary storage."""
import importlib.util,json,shutil,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('annotation_app',ROOT/'human_annotation/web/app.py');app=importlib.util.module_from_spec(spec);spec.loader.exec_module(app)
class AnnotationTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();base=Path(self.tmp.name);package=base/'package';package.mkdir()
  template=ROOT/'human_annotation/revision_v2/annotation_template.csv'
  for name in ['annotation_template.csv','annotation.csv']:shutil.copyfile(template,package/name)
  self.store=app.Store(base/'web',package)
 def tearDown(self):self.tmp.cleanup()
 def test_balanced_independent_and_hidden_labels(self):
  s=self.store;self.assertTrue(all(len(v)==120 for v in s.assignments.values()))
  self.assertEqual(len(s.pairs),480);self.assertTrue(all(a!=b for a,b in s.pairs.values()))
  sid=s.assignments[0][0];other=next(i for i in range(8) if i not in s.pairs[sid])
  with self.assertRaises(PermissionError):s.save(other,dict(annotation_id=sid,label='0',name='test'))
  with self.assertRaises(ValueError):s.save(0,dict(annotation_id=sid,label='unsure',name='test',note=''))
  s.save(0,dict(annotation_id=sid,label='0',name='test'))
  b=s.pairs[sid][1];case=next(r for r in s.view(b)['cases'] if r['annotation_id']==sid)
  self.assertIsNone(case['saved']);self.assertNotIn('reviews',case)
  with self.assertRaises(ValueError):s.submit(0)
 def test_third_review_only_after_lock_and_not_original_reviewer(self):
  s=self.store;sid=s.assignments[0][0];a,b=s.pairs[sid];third=s.third[sid]
  with s.connect() as c:
   for person in [a,b]:
    for case in s.assignments[person]:c.execute('INSERT INTO labels VALUES(?,?,?,?,?)',(case,person,'0','',1.))
   c.execute('UPDATE labels SET label="1" WHERE case_id=? AND person=?',(sid,b))
  self.assertFalse(s.adjudication_view(third)['cases']);s.submit(a);s.submit(b)
  self.assertIn(sid,[r['annotation_id'] for r in s.adjudication_view(third)['cases']])
  with self.assertRaises(ValueError):s.save(a,dict(annotation_id=sid,label='1',name='test'))
  with self.assertRaises(PermissionError):s.adjudicate(a,dict(annotation_id=sid,label='1',note='test reason'))
  s.adjudicate(third,dict(annotation_id=sid,label='1',note='test reason'))
  self.assertEqual(s.progress()['adjudicated'],1)
