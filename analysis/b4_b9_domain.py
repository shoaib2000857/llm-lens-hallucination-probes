"""Cross-domain paired gaps, training-free confidence cues, strata (shortcut checks) and ranking stability."""
import numpy as np, pandas as pd
from scipy.stats import kendalltau
import lib

DIST = lib.EV / 'results' / 'distribution_v1'


def seed_mean(dirs):
    ps = [lib.preds(d) for d in dirs]; ps = [p for p in ps if p]
    if not ps:
        return None
    ids = sorted(set.intersection(*[set(p) for p in ps]))
    return {i: (ps[0][i][0], float(np.mean([p[i][1] for p in ps]))) for i in ids}, len(ps)


def frozen(src, tgt, probe):
    return seed_mean([DIST / f'{tgt}_{src}__{probe}__s{s}' for s in lib.SEEDS])


def oof(src, tgt, probe):
    return seed_mean([DIST / f'{tgt}_{src}__{probe}__s{s}__cv_scratch' for s in lib.SEEDS])


def source_test(src, probe):
    return seed_mean([lib.PROBES / f'{src}__{probe}__s{s}__scratch_full' for s in lib.SEEDS])


def mean_seed_auroc(dirs):
    v = []
    for d in dirs:
        f = d / 'evaluation.json'
        if f.exists():
            v.append(lib.jload(f)['metrics']['auroc'])
    return float(np.mean(v)) if v else np.nan


def domain_gaps():
    rows = []
    for src in lib.SOURCES:
        for tgt in lib.TARGETS:
            corpus = f'{tgt}_{src}'; meta = lib.meta(corpus)
            for probe in lib.CORE:
                F, O = frozen(src, tgt, probe), oof(src, tgt, probe)
                if not F or not O:
                    continue
                (F, nf), (O, no) = F, O
                ids = sorted(set(F) & set(O)); y = np.array([F[i][0] for i in ids])
                pf = np.array([F[i][1] for i in ids]); po = np.array([O[i][1] for i in ids]); g = [meta[i]['context'] for i in ids]
                lo, hi = lib.cluster_bootstrap(g, lambda ix: lib.auroc(y[ix], po[ix]) - lib.auroc(y[ix], pf[ix]), reps=1000)
                rows.append(dict(source=src, target=tgt, probe=probe, n=len(ids), positives=int(y.sum()), contexts=len(set(g)),
                    frozen_auroc_mean_seed=mean_seed_auroc([DIST / f'{corpus}__{probe}__s{s}' for s in lib.SEEDS]),
                    frozen_auroc_ens=lib.auroc(y, pf), oof_auroc_ens=lib.auroc(y, po), frozen_ap_ens=lib.ap(y, pf),
                    gap_oof_minus_frozen=lib.auroc(y, po) - lib.auroc(y, pf), ci_low=lo, ci_high=hi))
    d = pd.DataFrame(rows); lib.save_csv(d, 'B4a_domain_frozen_vs_oof.csv'); return d


def confidence_cues():
    """Training-free cues with a-priori orientation (no label-based sign selection)."""
    rows = []
    corpora = [(s, s, 'test') for s in lib.SOURCES] + [(f'{t}_{s}', s, None) for s in lib.SOURCES for t in lib.TARGETS]
    for corpus, src, split in corpora:
        man = lib.manifest(corpus)
        recs = [r for r in man.values() if r['label'] is not None and (split is None or r['split'] == split)]
        y = np.array([r['label'] for r in recs])
        cues = {'neg_mean_logprob': [-r['mean_token_logprob'] for r in recs], 'neg_min_logprob': [-r['min_token_logprob'] for r in recs],
                'mean_entropy': [r['mean_token_entropy'] for r in recs], 'n_gen_tokens': [r['n_gen_tokens'] for r in recs]}
        for k, v in cues.items():
            rows.append(dict(corpus=corpus, generator=src, cue=k, n=len(y), positives=int(y.sum()), auroc=lib.auroc(y, np.array(v, float))))
    d = pd.DataFrame(rows); lib.save_csv(d, 'B5a_training_free_cues.csv'); return d


REFUSAL = ('cannot', "can't", 'not provided', 'not mentioned', 'does not', "doesn't", 'no information', 'unanswerable', 'not specified', 'not stated', 'unable')


def strata():
    rows = []
    for src in lib.SOURCES:
        meta = lib.meta(src)
        for probe in lib.CORE + ['best_layer_linear']:
            S = source_test(src, probe)
            if not S:
                continue
            S, _ = S; ids = sorted(S); y = np.array([S[i][0] for i in ids]); p = np.array([S[i][1] for i in ids])
            ua = np.array([bool(meta[i]['unanswerable']) for i in ids]); L = np.array([meta[i]['n_tokens'] for i in ids], float)
            ref = np.array([any(w in (meta[i]['text'] or '').lower() for w in REFUSAL) for i in ids])
            terc = np.digitize(L, np.quantile(L, [1 / 3, 2 / 3]))
            groups = {'all': np.ones(len(ids), bool), 'answerable_q': ~ua, 'unanswerable_q': ua, 'refusal_like_text': ref,
                      'non_refusal_text': ~ref, 'len_short': terc == 0, 'len_mid': terc == 1, 'len_long': terc == 2}
            for k, m in groups.items():
                rows.append(dict(source=src, probe=probe, stratum=k, n=int(m.sum()), positives=int(y[m].sum()), auroc=lib.auroc(y[m], p[m])))
        # cue-only references on the same test responses
        S, _ = source_test(src, 'M1'); ids = sorted(S); y = np.array([S[i][0] for i in ids])
        for name, v in [('cue:unanswerable_flag', [float(bool(meta[i]['unanswerable'])) for i in ids]),
                        ('cue:refusal_like_text', [float(any(w in (meta[i]['text'] or '').lower() for w in REFUSAL)) for i in ids]),
                        ('cue:length', [meta[i]['n_tokens'] for i in ids])]:
            rows.append(dict(source=src, probe=name, stratum='all', n=len(ids), positives=int(y.sum()), auroc=lib.auroc(y, np.array(v, float))))
    d = pd.DataFrame(rows); lib.save_csv(d, 'B4b_strata_source.csv'); return d


def ranking():
    """Seed-mean AUROC of the 8 core probes in every setting; Kendall tau between settings."""
    table = {}
    for src in lib.SOURCES:
        table[f'source:{src}'] = {p: mean_seed_auroc([lib.PROBES / f'{src}__{p}__s{s}__scratch_full' for s in lib.SEEDS]) for p in lib.CORE}
        for tgt in lib.TARGETS:
            table[f'frozen:{tgt}_{src}'] = {p: mean_seed_auroc([DIST / f'{tgt}_{src}__{p}__s{s}' for s in lib.SEEDS]) for p in lib.CORE}
    h = pd.read_csv(lib.HUMAN / 'H4_frozen_probe_metrics.csv')
    for (corpus, gen), g in h.groupby(['corpus', 'generator']):
        table[f'human:{corpus}'] = {r.probe: r.auroc for r in g.itertuples() if r.probe in lib.CORE}
    M = pd.DataFrame(table).T[lib.CORE]
    M.index.name = 'setting'; lib.save_csv(M.reset_index(), 'B9_probe_auroc_by_setting.csv')
    keys = list(M.index); taus = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = M.loc[keys[i]], M.loc[keys[j]]; m = a.notna() & b.notna()
            if m.sum() >= 5:
                taus.append(dict(a=keys[i], b=keys[j], tau=kendalltau(a[m], b[m]).statistic, kind=keys[i].split(':')[0] + '-' + keys[j].split(':')[0]))
    T = pd.DataFrame(taus); lib.save_csv(T, 'B9_kendall_pairs.csv')
    best = M.idxmax(axis=1).value_counts(); ranks = M.rank(axis=1, ascending=False)
    summ = pd.DataFrame({'times_best': best, 'mean_rank': ranks.mean(), 'worst_rank': ranks.max()}).fillna(0)
    lib.save_csv(summ.reset_index().rename(columns={'index': 'probe'}), 'B9_rank_summary.csv')
    return M, T, summ


if __name__ == '__main__':
    d = domain_gaps(); print(d.groupby(['source', 'target'])[['frozen_auroc_mean_seed', 'oof_auroc_ens', 'gap_oof_minus_frozen']].mean().round(3))
    print((d.ci_low > 0).sum(), 'of', len(d), 'gaps with OOF reliably above frozen;', (d.ci_high < 0).sum(), 'with frozen reliably above OOF')
    c = confidence_cues(); print(c.pivot(index='corpus', columns='cue', values='auroc').round(3))
    s = strata(); print(s.pivot_table(index=['source', 'probe'], columns='stratum', values='auroc').round(3).to_string())
    M, T, summ = ranking(); print(summ); print(T.groupby('kind').tau.describe().round(2))
