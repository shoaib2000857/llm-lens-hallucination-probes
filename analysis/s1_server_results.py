"""Analyse server results (layer scan, latency, answer-only, same-family strata). Read-only."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import lib

R = lib.WS / 'side_papers' / '02_journal_probing' / 'server_jobs' / 'results_20261108' / 'results'


def layer_scan():
    s = pd.DataFrame(json.loads((R / 'layer_scan_v1' / 'layer_scan_summary.json').read_text()))
    s['depth'] = (s.layer + 1) / s.n_layers
    lib.save_csv(s, 'S1_layer_scan_long.csv')
    out = []
    for (src, corpus), g in s.groupby(['source', 'eval_corpus']):
        g = g.sort_values('layer')
        best = g.loc[g.consensus_auroc.idxmax()]; devbest = g.loc[g.dev_auroc.idxmax()]
        out.append(dict(source=src, corpus=corpus, n_layers=int(g.n_layers.iloc[0]), best_test_layer=int(best.layer) + 1,
                        best_test_auroc=best.consensus_auroc, dev_selected_layer=int(devbest.layer) + 1, dev_selected_test_auroc=devbest.consensus_auroc,
                        first_layer_auroc=g.consensus_auroc.iloc[0], last_layer_auroc=g.consensus_auroc.iloc[-1],
                        mid_band_auroc=g[(g.depth > .4) & (g.depth <= .7)].consensus_auroc.mean()))
    o = pd.DataFrame(out); lib.save_csv(o, 'S1_layer_scan_summary.csv'); return s, o


def human_by_layer():
    c = pd.read_csv(lib.HUMAN / 'case_ledger_deidentified.csv')
    rows = []
    for src in lib.SOURCES:
        for corpus, kind in [(src, 'test')] + [(f'{t}_{src}', 'all') for t in ['distribution_v1_hotpotqa', 'distribution_v1_pubmedqa']]:
            sc = np.load(R / 'layer_scan_v1' / f'scores_{corpus}_{kind}.npy'); ids = json.loads((R / 'layer_scan_v1' / f'ids_{corpus}_{kind}.json').read_text())
            pos = {i: k for k, i in enumerate(ids)}
            h = c[(c.corpus == corpus) & c.human.notna()]
            h = h[h.sample_id.isin(pos)]
            idx = [pos[i] for i in h.sample_id]
            for l in range(sc.shape[1]):
                rows.append(dict(source=src, corpus=corpus, layer=l + 1, n_layers=sc.shape[1], n=len(h),
                                 human_auroc=lib.auroc(h.human.astype(int), sc[idx, l]), consensus_auroc=lib.auroc(h.consensus.astype(int), sc[idx, l])))
    d = pd.DataFrame(rows); lib.save_csv(d, 'S1_layer_scan_human.csv'); return d


def latency():
    d = pd.DataFrame(json.loads((R / 'latency_v1' / 'latency.json').read_text()))
    lib.save_csv(d, 'S2_latency.csv'); return d


def answer_only():
    a = pd.read_csv(R / 'answer_only_v1' / 'answer_only_metrics.csv')
    ref = pd.read_csv(lib.OUT / 'B9_probe_auroc_by_setting.csv').set_index('setting')
    s = pd.read_csv(lib.OUT / 'S1_layer_scan_summary.csv').set_index(['source', 'corpus'])
    rows = []
    for r in a.itertuples():
        key = f'source:{r.source}' if r.eval_corpus == r.source else f'frozen:{r.eval_corpus}'
        rows.append(dict(source=r.source, corpus=r.eval_corpus, head=r.head, answer_only_auroc=r.consensus_auroc,
                         full_mean_linear=ref.loc[key, 'mean_linear'], full_dev_best_layer_scan=s.loc[(r.source, r.eval_corpus), 'dev_selected_test_auroc']))
    d = pd.DataFrame(rows); lib.save_csv(d, 'S3_answer_only_vs_full.csv'); return d


if __name__ == '__main__':
    s, o = layer_scan(); print(o.round(3).to_string())
    h = human_by_layer(); print(h.groupby(['corpus']).apply(lambda g: g.loc[g.human_auroc.idxmax(), ['layer', 'human_auroc']], include_groups=False))
    print(latency().round(3).to_string())
    print(answer_only().round(3).to_string())
