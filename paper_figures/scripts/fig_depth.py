import pandas as pd, numpy as np
from make_figures import plt, save, C, NAME, AN, INK2, DOM
s = pd.read_csv(AN / 'S1_layer_scan_long.csv')
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
for ax, src in zip(axes, ['ministral3_8b', 'qwen3_8b']):
    g = s[s.source == src]
    src_g = g[g.eval_corpus == src].sort_values('layer')
    ax.plot(src_g.layer + 1, src_g.consensus_auroc, color=C[src], lw=2, marker='o', ms=2.5, label='Source test (SQuAD)')
    tg = g[g.eval_corpus != src].groupby('layer').consensus_auroc.mean()
    ax.plot(tg.index + 1, tg.values, color=C[src], lw=1.4, ls='--', label='Frozen, mean of 4 shifted sets')
    dl = int(src_g.loc[src_g.dev_auroc.idxmax(), 'layer']) + 1
    ax.axvline(dl, color=INK2, lw=.8, ls=':'); ax.text(dl + .5, .955 if src == 'ministral3_8b' else .62, f'dev-selected\nlayer {dl}', fontsize=6.5, color=INK2)
    ax.set_title(NAME[src]); ax.set_xlabel('Layer'); ax.axhline(.5, color=INK2, lw=.6, ls=':')
axes[0].set_ylabel('Test AUROC (single-layer logistic probe)'); axes[0].set_ylim(.45, 1.0); axes[0].legend(loc='lower right')
save(fig, 'depth_profile')
