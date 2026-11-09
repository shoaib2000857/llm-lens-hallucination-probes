#!/usr/bin/env python3
"""CPU-only complete paper evidence bundle; no base weights or hidden tensors."""
import csv,hashlib,json,re,statistics,subprocess,time,zipfile
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exports/paper_evidence_20261012';OUT.mkdir(parents=True,exist_ok=True)
def write(name,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for row in rows for k in row))
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def walk(obj,context='root'):
 if isinstance(obj,dict):
  if 'auroc' in obj and isinstance(obj.get('auroc'),(int,float,type(None))):yield context,obj
  for k,v in obj.items():
   if k not in ('predictions','records','turns','answers'):yield from walk(v,context+'.'+k)
 elif isinstance(obj,list):
  for i,v in enumerate(obj):yield from walk(v,context+f'[{i}]')
def main():
 files=list((ROOT/'results').rglob('*'));probe=[]
 for folder in (ROOT/'artifacts/probes').iterdir():
  if not folder.is_dir():continue
  probe.extend(p for p in folder.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.jsonl','.md') and 'fixture' not in str(p))
 rows=[];issues=[];primary=[]
 for p in sorted([p for p in files+probe if p.suffix=='.json' and p.is_file()]):
  try:d=json.loads(p.read_text())
  except Exception as e:issues.append(dict(file=str(p.relative_to(ROOT)),error=str(e)));continue
  for context,m in walk(d):
   row=dict(file=str(p.relative_to(ROOT)),context=context,experiment=p.parent.name,**{k:v for k,v in m.items() if isinstance(v,(str,int,float,type(None),bool)) and k not in ('file','context','experiment')});rows.append(row)
  if p.name=='evaluation.json' and isinstance(d,dict) and isinstance(d.get('metrics'),dict):
   parts=p.parent.name.split('__');probe_name=next((part for part in parts if part in ['M0','M1','M2','M3','M4','mean_linear','last_linear','last_mlp','best_layer_linear','M1_uniform','M2_no_positions','M3_no_layer_positions','M4_no_token_positions','capacity_global']), 'baseline')
   seed=next((int(v[1:]) for v in parts if re.fullmatch(r's\d+',v)),None)
   primary.append(dict(file=str(p.relative_to(ROOT)),experiment=p.parent.name,setting=parts[0],probe=probe_name,seed=seed,adaptation='__'.join(parts[3:]),mode=d.get('mode','baseline'),**d['metrics']))
 write('all_metric_occurrences.csv',rows);write('primary_evaluations.csv',primary);write('parse_issues.csv',issues)
 grouped=defaultdict(list)
 for row in primary:grouped[(row['setting'],row['probe'],row['adaptation'],row['mode'])].append(row)
 aggregates=[]
 for key,rs in sorted(grouped.items()):
  row=dict(zip(['setting','probe','adaptation','mode'],key));row['evaluation_count']=len(rs);row['seeds']=','.join(str(s) for s in sorted(set(r['seed'] for r in rs if r['seed'] is not None)))
  for metric in ['auroc','average_precision','f1_05','ece','brier','mean_probability']:
   vals=[r[metric] for r in rs if isinstance(r.get(metric),(int,float))];row[metric+'_mean']=statistics.mean(vals) if vals else None;row[metric+'_std']=statistics.stdev(vals) if len(vals)>1 else None
  aggregates.append(row)
 write('seed_summary.csv',aggregates)
 (OUT/'seed_summary.md').write_text('|Setting|Probe|Adaptation|Raw AUROC mean|Seed SD|AP mean|F1@0.5 mean|ECE mean|Brier mean|\n|---|---|---|---|---|---|---|---|---|\n'+'\n'.join('|'+ '|'.join(str(r.get(k,'')) for k in ['setting','probe','adaptation','auroc_mean','auroc_std','average_precision_mean','f1_05_mean','ece_mean','brier_mean'])+'|' for r in aggregates)+'\n')
 # Matched latest family references: unchanged IDs and labels required.
 gaps=[]
 for p in sorted((ROOT/'results/samefamily_v1').glob('*/predictions.json')):
  parts=p.parent.name.split('__')
  if len(parts)!=3:continue
  target,probe_name,seed=parts
  ref=ROOT/'artifacts/probes'/f'{target}__{probe_name}__{seed}__scratch_full'
  scope='held_out_target_test'
  if not (ref/'evaluation.json').exists():ref=ROOT/'results/distribution_v1'/f'{target}__{probe_name}__{seed}__cv_scratch';scope='context_grouped_oof_reference'
  if not (ref/'evaluation.json').exists():issues.append(dict(file=str(p.relative_to(ROOT)),error='Missing fresh target reference'));continue
  a=json.loads((p.parent/'evaluation.json').read_text())['metrics'];b=json.loads((ref/'evaluation.json').read_text())['metrics']
  predfile=ref/'predictions.json'
  if predfile.exists():
   x={r['sample_id']:r['label'] for r in json.loads(p.read_text())};y={r['sample_id']:r['label'] for r in json.loads(predfile.read_text())}
   if x!=y:raise ValueError('Family reference label/ID mismatch:'+str(p))
  au=a['auroc'];bu=b['auroc'];gaps.append(dict(target=target,probe=probe_name,seed=int(seed[1:]),reference_scope=scope,transfer_auroc=au,reference_auroc=bu,gap=bu-au if au is not None and bu is not None else None,retention=(au-.5)/(bu-.5) if au is not None and bu is not None and bu>.52 else None,transfer_file=str(p.parent.relative_to(ROOT)),reference_file=str(ref.relative_to(ROOT))))
 write('samefamily_matched_gaps.csv',gaps);write('parse_issues.csv',issues)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 for family,prefix in [('Same-family frozen transfer','samefamily_v2'),('Architecture ablations','ablation')]:
  selected=[r for r in aggregates if (r['mode']=='zero_shot_same_family_parameter_transfer' if prefix=='samefamily_v2' else 'ablation' in r['adaptation']) and r['auroc_mean'] is not None]
  if not selected:continue
  fig,ax=plt.subplots(figsize=(11,max(4,len(selected)*.16)))
  ax.barh(range(len(selected)),[r['auroc_mean'] for r in selected],xerr=[r['auroc_std'] or 0 for r in selected]);ax.set_yticks(range(len(selected)),[r['setting'].replace('samefamily_v2_','')+' / '+r['probe'] for r in selected],fontsize=7);ax.axvline(.5,color='gray',linestyle='--');ax.set_xlim(0,1);ax.set_xlabel('Raw AUROC; error bars = sample SD across saved probe seeds');ax.set_title(family);fig.tight_layout()
  for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{prefix}_auroc.{ext}',dpi=160)
  plt.close(fig)
 readme='''# Paper evidence bundle\n\nStart with primary_evaluations.csv, seed_summary.csv and samefamily_matched_gaps.csv. all_metric_occurrences.csv exhaustively indexes nested JSON metric records; it includes duplicated summaries, pilots, diagnostic inversion, per-judge and calibration metrics. It is an INDEX, not a set of independent experiments.\n\nOriginal results/ contains raw prediction scores, sampled text, per-judge metrics, calibration outputs, confidence intervals, processed summaries and existing vector/raster figures. artifacts/probes/ includes saved evaluation/config/history/completion/prediction metadata, not weights. Protocol/selection manifests and scripts are included for traceability. No hidden tensors/base checkpoints, credentials, participant links or private human records are included.\n\nRaw AUROC is primary; inversion is diagnostic and target-label-dependent. Average precision is AP, not trapezoidal PR area. Error bars in the new figures are seed SD, not confidence intervals. Individual CIs are in evaluation.json; matched bootstrap outputs from prior analyses remain in results/. OOF target references are not independent held-out tests. Retention is undefined when reference AUROC<=.52. Global MLP/ridge/Procrustes alignment is not per-layer alignment. Sampling disagreement is lexical token-set disagreement, not semantic entropy; ECE/Brier on this signal are not calibrated factuality claims. Adapted SpikeScore and Lookback use our response-label protocols; do not call them exact original benchmark reproduction. NQ uses10retrieved documents, not context-free NQ-Open. Three update trajectories share parent/domain/data; optimizer seeds do not imply independent domains. Judges are not human truth. Human validation remains pending.\n\nHistorical docs/results can contain superseded status notes; TASKS.md latest dated entries govern completion. Pilot/smoke/medium outputs and historical failed-run artifacts are preserved for transparency; exclude them from headline tables. Missing scope items are documented, not silently completed. Manuscript claims need author review.\n\nFigures: results/model_updates_v1/, results/publication_cpu_v1/, results/architecture_audit_v1/ and this directory. Coverage and computation notes: TASKS.md, docs/REVIEWER_REQUIREMENTS_MATRIX.md, docs/PROJECT_AUDIT.md, docs/RESOURCE_PLAN.md.\n'''
 (OUT/'README.md').write_text(readme)
 (OUT/'build_info.json').write_text(json.dumps(dict(timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),metric_occurrences=len(rows),primary_evaluations=len(primary),samefamily_gaps=len(gaps),issues=issues),indent=2))
 include=[p for p in files if p.is_file()]+probe
 for dirname in ['docs','configs','scripts','src','tests','experiments']:
  include.extend(p for p in (ROOT/dirname).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pt','.npy'))
 include.extend(p for p in (ROOT/'logs').iterdir() if p.is_file() and p.suffix in ('.log','.csv','.exit') and not any(v in p.name.lower() for v in ['annotation','ngrok','cloudflare','private','hf_sync']));include.append(ROOT/'TASKS.md');include.extend(p for p in OUT.iterdir() if p.is_file())
 # Compact original response/judge metadata: no representations or private human material.
 for parent in (ROOT/'artifacts').glob('stage1_*'):
  include.extend(p for p in parent.glob('*.json') if p.is_file())
  include.extend(p for p in parent.rglob('*.jsonl') if 'hidden_states' not in p.parts and 'pooled' not in p.parts)
  include.extend(p for p in parent.rglob('*.json') if not p.name.endswith('.commit.json'))
 manifest=[];archive=ROOT/'exports/paper_evidence_20261012.zip';tmp=archive.with_suffix('.zip.tmp')
 with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
  for p in sorted(set(include)):
   rel=str(p.relative_to(ROOT));h=hashlib.sha256()
   with p.open('rb') as f:
    for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
   manifest.append(dict(path=rel,bytes=p.stat().st_size,sha256=h.hexdigest()));z.write(p,rel)
  z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
 tmp.replace(archive)
 with zipfile.ZipFile(archive) as z:
  bad=z.testzip()
  if bad:raise ValueError('Corrupt ZIP entry:'+bad)
 digest=hashlib.sha256()
 with archive.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):digest.update(c)
 archive.with_suffix('.zip.sha256').write_text(digest.hexdigest()+'  '+archive.name+'\n')
 print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,files=len(manifest),evaluations=len(primary),gaps=len(gaps),issues=len(issues))),flush=True)
if __name__=='__main__':main()
