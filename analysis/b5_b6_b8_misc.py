"""B4c per-fold OOF check, B5 judge-human error structure, B6 legacy table, B8 CPU probe latency."""
import sys, time, re
import numpy as np, pandas as pd
import lib

DIST = lib.EV / 'results' / 'distribution_v1'


def perfold_oof():
    """Pooled OOF AUROC mixes 5 separately-fitted models' score scales; report mean within-fold AUROC too."""
    rows = []
    for src in lib.SOURCES:
        for tgt in lib.TARGETS:
            for probe in lib.CORE:
                pooled, within = [], []
                for s in lib.SEEDS:
                    f = DIST / f'{tgt}_{src}__{probe}__s{s}__cv_scratch' / 'predictions.json'
                    if not f.exists():
                        continue
                    d = pd.DataFrame(lib.jload(f))
                    pooled.append(lib.auroc(d.label, d.probability))
                    within.append(np.nanmean([lib.auroc(g.label, g.probability) for _, g in d.groupby('fold')]))
                if pooled:
                    rows.append(dict(source=src, target=tgt, probe=probe, oof_pooled_auroc=np.mean(pooled), oof_within_fold_auroc=np.mean(within)))
    d = pd.DataFrame(rows)
    fz = pd.read_csv(lib.OUT / 'B4a_domain_frozen_vs_oof.csv')[['source', 'target', 'probe', 'frozen_auroc_mean_seed']]
    d = d.merge(fz, on=['source', 'target', 'probe'])
    d['gap_within_minus_frozen'] = d.oof_within_fold_auroc - d.frozen_auroc_mean_seed
    lib.save_csv(d, 'B4c_oof_pooled_vs_within_fold.csv'); return d


def judge_human():
    c = pd.read_csv(lib.HUMAN / 'case_ledger_deidentified.csv')
    rows = []
    for keys, g in [(('all',), c)] + [((k,), g) for k, g in c.groupby('dataset')] + [((a, b), g) for (a, b), g in c.groupby(['dataset', 'generator'])]:
        g = g[g.consensus.notna() & g.human.notna()]
        h = g.human.astype(int); j = g.consensus.astype(int)
        tp = int(((j == 1) & (h == 1)).sum()); fp = int(((j == 1) & (h == 0)).sum()); fn = int(((j == 0) & (h == 1)).sum()); tn = int(((j == 0) & (h == 0)).sum())
        rows.append(dict(group='/'.join(keys), n=len(g), human_pos=int(h.sum()), judge_pos=int(j.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                         agreement=(tp + tn) / len(g), judge_precision=tp / max(tp + fp, 1), judge_recall=tp / max(tp + fn, 1)))
    # per-judge agreement with humans using manifest judge labels
    per = []
    for r in c.itertuples():
        m = lib.manifest(r.corpus).get(r.sample_id)
        if m:
            for k, v in (m.get('judge_labels') or {}).items():
                if v is not None:
                    per.append(dict(dataset=r.dataset, generator=r.generator, judge=k, agree=int(v == r.human), self_judge=int(k == 'qwen' and r.generator == 'qwen3_8b')))
    P = pd.DataFrame(per)
    lib.save_csv(pd.DataFrame(rows), 'B5b_consensus_vs_human_confusion.csv')
    pj = P.groupby(['judge', 'dataset', 'generator']).agree.agg(['mean', 'count']).reset_index()
    lib.save_csv(pj, 'B5c_per_judge_human_agreement.csv')
    return pd.DataFrame(rows), P.groupby(['judge', 'generator']).agree.mean().unstack()


def legacy():
    """Extract the legacy tables verbatim from local LaTeX sources for manual provenance review."""
    out = []
    for name, path in [('V1_arxiv_2604.06277', lib.WS / 'OLD-References' / 'V1' / 'main.tex'),
                       ('ARR2721', lib.WS / 'OLD-References' / 'llm-lens-temp' / 'EMNLP' / 'main_revised.tex')]:
        t = path.read_text(encoding='utf-8', errors='ignore')
        for m in re.finditer(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', t, re.S):
            cap = re.search(r'\\caption\{(.*?)\}\s*\\label', m.group(0), re.S)
            out.append(f'## {name}\n\nCaption: {cap.group(1).strip() if cap else "?"}\n\n```latex\n{m.group(0)}\n```\n')
    p = lib.OUT / 'B6_legacy_tables_extracted.md'
    p.write_text('# Legacy tables extracted verbatim (for provenance review; not re-verified numbers)\n\n' + '\n'.join(out), encoding='utf-8')
    print('wrote', p.relative_to(lib.WS), len(out), 'tables')


def latency():
    sys.path.insert(0, str(lib.EV / 'src'))
    import torch
    from portability.probe_training import make_probe
    torch.set_num_threads(4); rows = []
    for name, L in [('qwen3_8b', 36), ('ministral3_8b', 34)]:
        for probe in lib.CORE:
            net = make_probe(probe, (L, 96, 4096)).eval()
            params = sum(p.numel() for p in net.parameters())
            for bs in [1, 32]:
                x = torch.randn(bs, L, 4096) if probe in ('M0', 'M1', 'mean_linear', 'last_linear', 'last_mlp') else torch.randn(bs, L, 96, 4096)
                with torch.inference_mode():
                    for _ in range(2): net(x)
                    t = time.perf_counter(); reps = 5
                    for _ in range(reps): net(x)
                    dt = (time.perf_counter() - t) / reps
                rows.append(dict(generator=name, probe=probe, params=params, batch=bs, ms_per_batch=dt * 1000, ms_per_sample=dt * 1000 / bs,
                                 input='pooled L×D' if x.dim() == 3 else 'full L×T×D', device='CPU, 4 threads, synthetic fp32 input'))
    lib.save_csv(pd.DataFrame(rows), 'B8_cpu_latency_synthetic.csv')


if __name__ == '__main__':
    d = perfold_oof(); print(d.groupby(['source', 'target'])[['oof_pooled_auroc', 'oof_within_fold_auroc', 'frozen_auroc_mean_seed']].mean().round(3))
    print('frozen above within-fold OOF in', int((d.gap_within_minus_frozen < 0).sum()), 'of', len(d))
    conf, pj = judge_human(); print(conf.round(3).to_string()); print(pj.round(3))
    legacy(); latency(); print(pd.read_csv(lib.OUT / 'B8_cpu_latency_synthetic.csv').round(2).to_string())
