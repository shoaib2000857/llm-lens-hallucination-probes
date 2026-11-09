"""B1 learning curves, B2 paired ablation/structure contrasts, B3 calibration (source SQuAD cohorts)."""
import numpy as np, pandas as pd
import lib

BUDGETS = ['25', '50', '100', '250', '500', '1000', 'full']


def fit_dir(src, probe, seed, arm):
    return lib.PROBES / f'{src}__{probe}__s{seed}__{arm}'


def b1():
    rows = []
    for src in lib.SOURCES:
        for probe in lib.CORE + ['best_layer_linear']:
            for b in BUDGETS:
                for s in lib.SEEDS:
                    f = fit_dir(src, probe, s, f'scratch_{b}') / 'evaluation.json'
                    if not f.exists():
                        continue
                    m = lib.jload(f)['metrics']; c = lib.jload(f.parent / 'config.json')
                    rows.append(dict(source=src, probe=probe, budget=b, labels=c.get('total_target_labels'), seed=s,
                                     auroc=m['auroc'], ap=m['average_precision'], f1_05=m['f1_05'], ece=m['ece'], brier=m['brier']))
    d = pd.DataFrame(rows)
    lib.save_csv(d, 'B1_learning_curves_fits.csv')
    g = d.groupby(['source', 'probe', 'budget'], sort=False).agg(labels=('labels', 'median'), n_seeds=('seed', 'count'),
        auroc=('auroc', 'mean'), auroc_sd=('auroc', 'std'), ap=('ap', 'mean'), ece=('ece', 'mean'), brier=('brier', 'mean'),
        f1_05=('f1_05', 'mean')).reset_index()
    lib.save_csv(g, 'B1_learning_curves_summary.csv')
    # fraction of full-budget AUROC above chance reached at each budget
    full = g[g.budget == 'full'].set_index(['source', 'probe']).auroc
    g['attainment'] = [(r.auroc - .5) / (full[(r.source, r.probe)] - .5) for r in g.itertuples()]
    lib.save_csv(g[['source', 'probe', 'budget', 'labels', 'auroc', 'attainment']], 'B1_attainment.csv')
    return g


def seed_mean_scores(src, probe, arm):
    ps = [lib.preds(fit_dir(src, probe, s, arm)) for s in lib.SEEDS]
    ps = [p for p in ps if p]
    if not ps:
        return None
    ids = sorted(set.intersection(*[set(p) for p in ps]))
    y = np.array([ps[0][i][0] for i in ids]); p = np.mean([[q[i][1] for i in ids] for q in ps], axis=0)
    return dict(zip(ids, zip(y, p))), len(ps)


CONTRASTS = [  # (label, probe A, arm A, probe B, arm B): delta = A - B
    ('M1 uniform weights vs M1', 'M1_uniform', 'ablation_full', 'M1', 'scratch_full'),
    ('M2 no positions vs M2', 'M2_no_positions', 'ablation_full', 'M2', 'scratch_full'),
    ('M3 no layer positions vs M3', 'M3_no_layer_positions', 'ablation_full', 'M3', 'scratch_full'),
    ('M4 no token positions vs M4', 'M4_no_token_positions', 'ablation_full', 'M4', 'scratch_full'),
    ('Capacity-matched global vs M0', 'capacity_global', 'ablation_full', 'M0', 'scratch_full'),
    ('Capacity-matched global vs M2', 'capacity_global', 'ablation_full', 'M2', 'scratch_full'),
] + [(f'{p} vs best-layer linear', p, 'scratch_full', 'best_layer_linear', 'scratch_full') for p in ['M0', 'M1', 'M2', 'M3', 'M4']] \
  + [(f'{p} vs mean linear', p, 'scratch_full', 'mean_linear', 'scratch_full') for p in ['M0', 'M1', 'M2', 'M3', 'M4']]


def b2():
    rows = []
    for src in lib.SOURCES:
        meta = lib.meta(src)
        for name, pa, aa, pb, ab in CONTRASTS:
            A = seed_mean_scores(src, pa, aa); B = seed_mean_scores(src, pb, ab)
            if not A or not B:
                rows.append(dict(source=src, contrast=name, note='missing')); continue
            (A, na), (B, nb) = A, B
            ids = sorted(set(A) & set(B))
            y = np.array([A[i][0] for i in ids]); pa_ = np.array([A[i][1] for i in ids]); pb_ = np.array([B[i][1] for i in ids])
            assert all(A[i][0] == B[i][0] for i in ids)
            g = [meta[i]['context'] for i in ids]
            d = lib.auroc(y, pa_) - lib.auroc(y, pb_)
            lo, hi = lib.cluster_bootstrap(g, lambda ix: lib.auroc(y[ix], pa_[ix]) - lib.auroc(y[ix], pb_[ix]))
            rows.append(dict(source=src, contrast=name, auroc_a=lib.auroc(y, pa_), auroc_b=lib.auroc(y, pb_), delta=d,
                             ci_low=lo, ci_high=hi, n=len(ids), contexts=len(set(g)), seeds_a=na, seeds_b=nb,
                             excludes_zero=bool(lo > 0 or hi < 0)))
    d = pd.DataFrame(rows)
    lib.save_csv(d, 'B2_paired_structure_contrasts.csv')
    return d


def b3():
    rows, bins = [], []
    for src in lib.SOURCES:
        for probe in lib.CORE + ['best_layer_linear']:
            for s in lib.SEEDS:
                p = lib.preds(fit_dir(src, probe, s, 'scratch_full'))
                if not p:
                    continue
                y = np.array([v[0] for v in p.values()]); q = np.array([v[1] for v in p.values()])
                rows.append(dict(source=src, probe=probe, seed=s, ece10=lib.ece(y, q), brier=float(np.mean((q - y) ** 2)),
                                 mean_p=float(q.mean()), prevalence=float(y.mean())))
                edges = np.linspace(0, 1, 11)
                for a, b in zip(edges[:-1], edges[1:]):
                    m = (q >= a) & ((q < b) if b < 1 else (q <= b))
                    if m.any():
                        bins.append(dict(source=src, probe=probe, seed=s, bin_low=a, n=int(m.sum()), mean_pred=float(q[m].mean()), frac_pos=float(y[m].mean())))
    d = pd.DataFrame(rows)
    lib.save_csv(d.groupby(['source', 'probe']).mean(numeric_only=True).reset_index().drop(columns='seed'), 'B3_calibration_summary.csv')
    lib.save_csv(pd.DataFrame(bins), 'B3_reliability_bins.csv')


if __name__ == '__main__':
    g = b1(); print(g[g.budget.isin(['25', '250', 'full'])].round(3).to_string())
    d = b2(); print(d.round(3).to_string())
    b3()
