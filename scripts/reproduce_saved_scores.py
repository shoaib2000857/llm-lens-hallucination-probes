#!/usr/bin/env python3
"""Recalculate legacy scores without LLM inference. Never tunes on test labels.

AP is average precision, not trapezoidal PR area. ECE includes p=1.
Legacy invalid probabilities/labels are reported, never silently imputed.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def metrics(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if not len(y) or len(y) != len(p) or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid labels or lengths')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid probabilities')
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    bins = np.minimum((p * 15).astype(int), 14)
    ece = sum(np.mean(bins == b) * abs(float(p[bins == b].mean() - y[bins == b].mean()))
              for b in range(15) if np.any(bins == b))
    return dict(n=len(y), positives=int(y.sum()), auroc=auc,
                average_precision=float(average_precision_score(y, p)) if y.sum() else None,
                f1_05=float(f1_score(y, p >= .5, zero_division=0)),
                ece=float(ece), brier=float(np.mean((p-y)**2)),
                mean_probability=float(p.mean()),
                diagnostic_inverted_auroc=None if auc is None else 1-auc)


def bootstrap(y, p, repeats, seed):
    y, p = np.asarray(y), np.asarray(p)
    strata = [np.flatnonzero(y == c) for c in (0, 1)]
    if any(len(s) == 0 for s in strata):
        return {'ci_status': 'undefined_single_class'}
    rng = np.random.default_rng(seed)
    keys = ['auroc', 'average_precision', 'f1_05', 'ece', 'brier', 'mean_probability']
    samples = {k: [] for k in keys}
    for _ in range(repeats):
        idx = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        m = metrics(y[idx], p[idx])
        for k in keys:
            samples[k].append(m[k])
    return {k+'_ci95': np.quantile(v, [.025, .975]).tolist() for k, v in samples.items()}


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def load_groups(path, limit):
    groups, seen, issues, examples = {}, set(), [], set()
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            sample = (row['dataset'], row['id'])
            if limit and sample not in examples and len(examples) >= limit:
                continue
            examples.add(sample)
            probe = row.get('probe', 'ensemble')
            key = (*sample, probe)
            if key in seen:
                raise ValueError(f'Duplicate sample/probe key: {path}: {key}')
            seen.add(key)
            try:
                label = float(row['judge_label'])
                if label not in (0, 1):
                    raise ValueError('nonbinary label')
                p = float(row.get('prob_hallucination', row.get('ensemble_prob', 'nan')))
                if not np.isfinite(p) or not 0 <= p <= 1:
                    raise ValueError('nonfinite/out-of-range probability')
                groups.setdefault((row['dataset'], probe), []).append((row['id'], int(label), p))
            except (ValueError, KeyError) as exc:
                issues.append(dict(sample_id=row['id'], dataset=row['dataset'], probe=probe, reason=str(exc)))
    return groups, issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--legacy-root', type=Path, default=Path('/home/jovyan/Previous-files/Transformer cross testing'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--bootstrap', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0, help='Max distinct samples per file; 0=all')
    args = ap.parse_args()
    if args.bootstrap < 0 or args.limit < 0:
        ap.error('Counts must be nonnegative')
    args.output.mkdir(parents=True, exist_ok=True)
    patterns = ['*with_mistral_judge.csv', '*with_similarity_and_judge.csv', 'ensemble_raw_against_mistral_judge.csv']
    paths = sorted({p for pattern in patterns for p in args.legacy_root.rglob(pattern)})
    if not paths:
        raise ValueError('No saved score files found')
    start = time.time()
    rows, provenance, failures = [], [], []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        groups, issues = load_groups(path, args.limit)
        rel = str(path.relative_to(args.legacy_root))
        provenance.append(dict(path=rel, sha256=digest))
        failures.extend(dict(file=rel, **i) for i in issues)
        for (dataset, probe), values in sorted(groups.items()):
            ids, y, p = zip(*sorted(values))
            row = dict(file=rel, dataset=dataset, probe=probe, mode='zero_shot_transfer', **metrics(y, p))
            if args.bootstrap:
                row.update(bootstrap(y, p, args.bootstrap, args.seed))
            rows.append(row)
        print(f'{rel}: {sum(len(v) for v in groups.values())} valid rows; {len(issues)} invalid', flush=True)
    atomic_json(args.output/'metrics.json', rows)
    atomic_json(args.output/'invalid_rows.json', failures)
    fields = sorted({k for r in rows for k in r})
    with (args.output/'metrics.csv.tmp').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (args.output/'metrics.csv.tmp').replace(args.output/'metrics.csv')
    md = '| Run file | Dataset | Probe | n | Raw AUROC | F1@.5 | Diagnostic inverted AUROC |\n|---|---|---|---:|---:|---:|---:|\n'
    for r in rows:
        fmt = lambda k: 'undefined' if r[k] is None else f'{r[k]:.6f}'
        md += f"| {r['file']} | {r['dataset']} | {r['probe']} | {r['n']} | {fmt('auroc')} | {fmt('f1_05')} | {fmt('diagnostic_inverted_auroc')} |\n"
    (args.output/'metrics.md').write_text(md)
    def latex(s):
        return str(s).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')
    lines = [r'\begin{tabular}{llrrr}', r'Dataset & Probe & N & AUROC & F1 \\']
    for r in rows:
        lines.append(f"{latex(r['dataset'])} & {latex(r['probe'])} & {r['n']} & {r['auroc']} & {r['f1_05']:.6f}" + r' \\')
    lines.append(r'\end{tabular}')
    (args.output/'metrics.tex').write_text('\n'.join(lines))
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    atomic_json(args.output/'manifest.json', dict(experiment_id=args.output.name,
        command=__import__('sys').argv, seed=args.seed, bootstrap=args.bootstrap,
        source_files=provenance, python=platform.python_version(), numpy=np.__version__,
        sklearn=__import__('sklearn').__version__, git_commit=commit,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        elapsed_seconds=time.time()-start, gpu_hours=0, invalid_row_count=len(failures),
        note='Artifact arithmetic reproduction; no generator/probe inference. Invalid rows explicitly excluded. Inversion diagnostic requires target labels.'))


if __name__ == '__main__':
    main()
