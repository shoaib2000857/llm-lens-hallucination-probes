import hashlib,json,zipfile
from pathlib import Path
root=Path('/home/jovyan/hallucination-portability');archive=root/'exports/paper_evidence_20261012.zip';extra=[]
with zipfile.ZipFile(archive,'a',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
 seen=set(z.namelist())
 for parent in sorted((root/'artifacts').glob('stage1_*')):
  for p in sorted(parent.rglob('*.json')):
   if 'pooled' in p.parts and p.name!='manifest.json':continue
   if 'hidden_states' in p.parts and p.name.endswith('.commit.json'):continue
   rel=str(p.relative_to(root))
   if rel in seen:continue
   blob=p.read_bytes();extra.append(dict(path=rel,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()));z.write(p,rel)
 z.writestr('RAW_METADATA_MANIFEST.json',json.dumps(extra,indent=2))
with zipfile.ZipFile(archive) as z:
 bad=z.testzip()
 if bad:raise ValueError(bad)
h=hashlib.sha256()
with archive.open('rb') as f:
 for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
archive.with_suffix('.zip.sha256').write_text(h.hexdigest()+'  '+archive.name+'\n')
print(json.dumps(dict(extra_raw_files=len(extra),zip_bytes=archive.stat().st_size,checksum=h.hexdigest())))
