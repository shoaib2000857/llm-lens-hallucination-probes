"""Paper A figures from saved analysis outputs (read-only). Run: python make_figures.py"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / 'figures'; FIG.mkdir(exist_ok=True)
WS = HERE.parents[3]
AN = WS / 'side_papers' / 'analysis' / 'outputs'
EMT = WS / 'manuscripts' / 'human_updated_20261104' / 'tables'

C = {'ministral3_8b': '#2a78d6', 'qwen3_8b': '#eb6834'}
NAME = {'ministral3_8b': 'Ministral-3-8B', 'qwen3_8b': 'Qwen3-8B'}
INK, INK2, GRID, NEUTRAL = '#0b0b0b', '#52514e', '#e4e3df', '#8c8b86'
PROBES = ['mean_linear', 'last_linear', 'last_mlp', 'M0', 'M1', 'M2', 'M3', 'M4']
PLAB = {'mean_linear': 'Mean-lin.', 'last_linear': 'Last-lin.', 'last_mlp': 'Last-MLP', 'best_layer_linear': 'Best-layer lin.'}
DOM = {'distribution_v1_hotpotqa': 'HotpotQA', 'distribution_v1_pubmedqa': 'PubMedQA', 'distribution_v2_nq_retrieval': 'NQ-retr.', 'distribution_v2_triviaqa': 'TriviaQA'}

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'legend.fontsize': 7, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6,
                     'axes.axisbelow': True, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'legend.frameon': False})


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(FIG / f'{name}.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig); print('fig', name)


def pl(p):
    return PLAB.get(p, p)


def source_dots():
    d = pd.read_csv(EMT / 'source.csv')
    order = PROBES + ['best_layer_linear']
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4))
    for ax, metric, lab in [(axes[0], 'auroc', 'AUROC'), (axes[1], 'ap', 'Average precision')]:
        for k, src in enumerate(['ministral3_8b', 'qwen3_8b']):
            s = d[d.setting == src].set_index('probe').loc[order]
            x = np.arange(len(order)) + (k - .5) * .22
            ax.errorbar(x, s[metric], yerr=s['auroc_seed_sd'] if metric == 'auroc' else None, fmt='o', ms=4.5, color=C[src], ecolor=C[src], elinewidth=1, capsize=0, label=NAME[src])
        ax.set_xticks(range(len(order)), [pl(p) for p in order], rotation=40, ha='right'); ax.set_ylabel(lab); ax.grid(axis='x', visible=False)
    axes[0].set_ylim(.4, 1.0); axes[1].set_ylim(.3, 1.0); axes[0].legend(loc='lower left')
    fig.tight_layout(w_pad=2.0)
    save(fig, 'source_detection')


def learning_curves():
    g = pd.read_csv(AN / 'B1_learning_curves_summary.csv')
    probes = ['M0', 'M1', 'mean_linear', 'last_mlp', 'last_linear']
    shades = dict(zip(probes, ['#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#8c8b86']))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
    for ax, src in zip(axes, ['ministral3_8b', 'qwen3_8b']):
        for p in probes:
            s = g[(g.source == src) & (g.probe == p)].sort_values('labels')
            ax.plot(s['labels'], s['auroc'], marker='o', ms=3.5, lw=1.6, color=shades[p], label=pl(p))
            ax.fill_between(s['labels'], s['auroc'] - s['auroc_sd'], s['auroc'] + s['auroc_sd'], color=shades[p], alpha=.12, lw=0)
        ax.set_xscale('log'); ax.set_xlabel('Target labels (train + development, log scale)'); ax.set_title(NAME[src], color=INK)
        ax.set_xticks([25, 50, 100, 250, 500, 1000, 6000], ['25', '50', '100', '250', '500', '1k', '~6k']); ax.minorticks_off()
    axes[0].set_ylabel('Test AUROC (mean ± seed SD)'); axes[0].legend(loc='lower right', ncol=1)
    save(fig, 'learning_curves')


def contrasts():
    d = pd.read_csv(AN / 'B2_paired_structure_contrasts.csv')
    keep = ['M1 uniform weights vs M1', 'M2 no positions vs M2', 'M3 no layer positions vs M3', 'M4 no token positions vs M4',
            'Capacity-matched global vs M0', 'Capacity-matched global vs M2', 'M1 vs best-layer linear', 'M2 vs best-layer linear',
            'M3 vs best-layer linear', 'M4 vs best-layer linear']
    lab = {'M1 uniform weights vs M1': 'M1: uniform − learned layer weights', 'M2 no positions vs M2': 'M2: no positions − original',
           'M3 no layer positions vs M3': 'M3: no layer positions − original', 'M4 no token positions vs M4': 'M4: no token positions − original',
           'Capacity-matched global vs M0': 'Capacity-matched global − M0', 'Capacity-matched global vs M2': 'Capacity-matched global − M2',
           'M1 vs best-layer linear': 'M1 − best-layer linear', 'M2 vs best-layer linear': 'M2 − best-layer linear',
           'M3 vs best-layer linear': 'M3 − best-layer linear', 'M4 vs best-layer linear': 'M4 − best-layer linear'}
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for k, src in enumerate(['ministral3_8b', 'qwen3_8b']):
        s = d[d.source == src].set_index('contrast').loc[keep]
        y = np.arange(len(keep))[::-1] + (.5 - k) * .28
        ax.errorbar(s.delta, y, xerr=[s.delta - s.ci_low, s.ci_high - s.delta], fmt='o', ms=4.5, color=C[src], elinewidth=1.2, capsize=0, label=NAME[src])
    ax.axvline(0, color=INK2, lw=.9); ax.set_yticks(np.arange(len(keep))[::-1], [lab[k] for k in keep]); ax.grid(axis='y', visible=False)
    ax.axhline(5.5, color=GRID, lw=1); ax.set_xlabel('Paired ΔAUROC on identical test responses (95% exact-context interval)')
    ax.legend(loc='lower right')
    save(fig, 'structure_contrasts')


def domain():
    d = pd.read_csv(AN / 'B4c_oof_pooled_vs_within_fold.csv')
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
    for ax, src in zip(axes, ['ministral3_8b', 'qwen3_8b']):
        s = d[d.source == src]
        for j, (t, tl) in enumerate(DOM.items()):
            r = s[s.target == t].set_index('probe').loc[PROBES]
            x = j + np.linspace(-.32, .32, len(PROBES))
            ax.vlines(x, r.oof_within_fold_auroc, r.frozen_auroc_mean_seed, color=GRID, lw=1.4)
            ax.scatter(x, r.oof_within_fold_auroc, s=12, color=NEUTRAL, zorder=3, label='In-domain OOF (~800 labels/fold)' if j == 0 else None)
            ax.scatter(x, r.frozen_auroc_mean_seed, s=14, color=C[src], zorder=4, label='Frozen SQuAD probe (~6k labels)' if j == 0 else None)
        ax.set_xticks(range(4), list(DOM.values())); ax.set_title(NAME[src], color=INK); ax.axhline(.5, color=INK2, lw=.7, ls=':')
        ax.grid(axis='x', visible=False); ax.legend(loc='lower left', fontsize=6.5)
    axes[0].set_ylabel('AUROC'); axes[0].set_ylim(.4, 1.0)
    save(fig, 'domain_transfer')


def ranks():
    M = pd.read_csv(AN / 'B9_probe_auroc_by_setting.csv').set_index('setting')[PROBES]
    R = M.rank(axis=1, ascending=False)
    def nice(s):
        kind, rest = s.split(':', 1)
        gen = 'Qwen' if 'qwen' in rest else 'Ministral'
        dom = next((v for k, v in DOM.items() if rest.startswith(k)), None)
        if kind == 'source': return f'Source SQuAD · {gen}'
        if kind == 'frozen': return f'Frozen → {dom} · {gen}'
        dom = {'distribution_v1_hotpotqa': 'HotpotQA', 'distribution_v1_pubmedqa': 'PubMedQA'}.get(rest.rsplit('_', 2)[0] if 'distribution' in rest else '', 'SQuAD')
        return f'Human ref. {dom} · {gen}'
    R.index = [nice(i) for i in R.index]
    R = R.loc[[i for i in R.index if not i.startswith('Human')] + sorted([i for i in R.index if i.startswith('Human')], key=lambda s: (['SQuAD', 'HotpotQA', 'PubMedQA'].index(s.split()[2]), s))]
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    im = ax.imshow(R.values, cmap='Blues_r', vmin=1, vmax=8, aspect='auto')
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            v = int(R.values[i, j]); ax.text(j, i, v, ha='center', va='center', fontsize=6.5, color='white' if v <= 3 else INK)
    ax.set_xticks(range(len(PROBES)), [pl(p) for p in PROBES], rotation=35, ha='right'); ax.set_yticks(range(len(R)), R.index)
    ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
    cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.02); cb.set_label('Rank by AUROC (1 = best)'); cb.ax.invert_yaxis()
    save(fig, 'rank_stability')


def judges_vs_humans():
    p = pd.read_csv(AN / 'B5c_per_judge_human_agreement.csv')
    c = pd.read_csv(AN / 'B5b_consensus_vs_human_confusion.csv')
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    sets = [('squad', 'ministral3_8b'), ('squad', 'qwen3_8b'), ('hotpotqa', 'ministral3_8b'), ('hotpotqa', 'qwen3_8b'), ('pubmedqa', 'ministral3_8b'), ('pubmedqa', 'qwen3_8b')]
    jcol = {'mistral': '#1baf7a', 'phi': '#6250d6', 'qwen': '#eb6834', 'consensus': INK}
    w = .19
    for k, j in enumerate(['mistral', 'phi', 'qwen', 'consensus']):
        vals = []
        for dset, gen in sets:
            if j == 'consensus':
                vals.append(float(c[c.group == f'{dset}/{gen}'].agreement.iloc[0]))
            else:
                vals.append(float(p[(p.judge == j) & (p.dataset == dset) & (p.generator == gen)]['mean'].iloc[0]))
        ax.scatter(np.arange(6) + (k - 1.5) * w, vals, s=34 if j == 'consensus' else 22, marker='D' if j == 'consensus' else 'o', color=jcol[j], zorder=3, label={'mistral': 'Mistral-7B judge', 'phi': 'Phi-4-mini judge', 'qwen': 'Qwen3-8B judge', 'consensus': 'Majority consensus'}[j])
    ax.set_xticks(range(6), [f"{ {'squad':'SQuAD','hotpotqa':'HotpotQA','pubmedqa':'PubMedQA'}[a]}\n{NAME[b].split('-')[0]}" for a, b in sets])
    [ax.axvline(x + .5, color=GRID, lw=.8) for x in range(5)]
    ax.set_ylim(.45, .92); ax.set_ylabel('Agreement with human reference'); ax.grid(axis='x', visible=False)
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.18))
    save(fig, 'judges_vs_humans')


def reliability():
    b = pd.read_csv(AN / 'B3_reliability_bins.csv')
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.6), sharey=True)
    pc = {'M1': '#eb6834', 'M3': '#2a78d6', 'mean_linear': '#1baf7a', 'last_mlp': '#6250d6'}
    for ax, src in zip(axes, ['ministral3_8b', 'qwen3_8b']):
        ax.plot([0, 1], [0, 1], color=INK2, lw=.8, ls=':')
        for p, col in pc.items():
            s = b[(b.source == src) & (b.probe == p)]
            g = s.groupby('bin_low').apply(lambda z: pd.Series({'x': np.average(z.mean_pred, weights=z.n), 'y': np.average(z.frac_pos, weights=z.n), 'n': z.n.sum()}), include_groups=False).reset_index()
            g = g[g.n >= 15]
            ax.plot(g.x, g.y, marker='o', ms=3.5, lw=1.5, color=col, label=pl(p))
        ax.set_xlabel('Mean predicted probability'); ax.set_title(NAME[src], color=INK); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel('Observed positive rate'); axes[0].legend(loc='upper left')
    save(fig, 'reliability')


if __name__ == '__main__':
    source_dots(); learning_curves(); contrasts(); domain(); ranks(); judges_vs_humans(); reliability()
