"""Targeted claim audit: recompute each headline number from its specific source and check the paper text.

Each claim: (id, value computed from evidence, decimals, papers it must appear in).
A claim FAILS if the rounded value is absent from a paper that should contain it.
"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
import lib

T = lib.WS / 'manuscripts' / 'human_updated_20261104' / 'tables'
src = pd.read_csv(T / 'source.csv').set_index(['setting', 'probe'])
lc = pd.read_csv(lib.OUT / 'B1_learning_curves_summary.csv').set_index(['source', 'probe', 'budget'])
at = pd.read_csv(lib.OUT / 'B1_attainment.csv').set_index(['source', 'probe', 'budget'])
con = pd.read_csv(lib.OUT / 'B2_paired_structure_contrasts.csv').set_index(['source', 'contrast'])
dom = pd.read_csv(lib.OUT / 'B4c_oof_pooled_vs_within_fold.csv')
ls = pd.read_csv(lib.OUT / 'S1_layer_scan_summary.csv').set_index(['source', 'corpus'])
hs = pd.read_csv(lib.OUT / 'S1_layer_scan_human.csv')
conf = pd.read_csv(lib.OUT / 'B5b_consensus_vs_human_confusion.csv').set_index('group')
pj = pd.read_csv(lib.OUT / 'B5c_per_judge_human_agreement.csv')
sf = pd.read_csv(T / 'samefamily.csv') if (T / 'samefamily.csv').exists() else None
rk = pd.read_csv(lib.OUT / 'B9_rank_summary.csv').set_index('probe')

M, Q = 'ministral3_8b', 'qwen3_8b'
C = []
def add(cid, v, nd, papers): C.append((cid, v, nd, papers))

add('src_qwen_M3', src.loc[(Q, 'M3'), 'auroc'], 3, 'AB')
add('src_min_M1', src.loc[(M, 'M1'), 'auroc'], 3, 'AB')
add('src_min_M4', src.loc[(M, 'M4'), 'auroc'], 3, 'AB')
add('src_qwen_lastlin', src.loc[(Q, 'last_linear'), 'auroc'], 3, 'AB')
add('src_qwen_M3_ap', src.loc[(Q, 'M3'), 'ap'], 3, 'AB')
add('src_min_M1_ap', src.loc[(M, 'M1'), 'ap'], 3, 'AB')
add('lc_qwen_M1_250', lc.loc[(Q, 'M1', '250'), 'auroc'], 3, 'AB')
add('lc_qwen_M1_500', lc.loc[(Q, 'M1', '500'), 'auroc'], 3, 'AB')
add('con_M1uniform_min', -con.loc[(M, 'M1 uniform weights vs M1'), 'delta'], 3, 'AB')
add('con_M2pos_min', con.loc[(M, 'M2 no positions vs M2'), 'delta'], 3, 'AB')
add('con_M4pos_min', con.loc[(M, 'M4 no token positions vs M4'), 'delta'], 3, 'AB')
add('con_M4pos_min_lo', con.loc[(M, 'M4 no token positions vs M4'), 'ci_low'], 3, 'AB')
add('layer_min_best', ls.loc[(M, M), 'dev_selected_test_auroc'], 3, 'AB')
add('layer_qwen_dev', ls.loc[(Q, Q), 'dev_selected_test_auroc'], 3, 'AB')
h = hs[(hs.corpus == Q) & (hs.layer == 21)].iloc[0]
add('human_layer_qwen_cons', h.consensus_auroc, 3, 'AB'); add('human_layer_qwen_human', h.human_auroc, 3, 'AB')
add('frozen_gt_oof_count', int((dom.frozen_auroc_mean_seed > dom.oof_within_fold_auroc).sum()), 0, 'AB')
add('cons_agree_all', 100 * conf.loc['all', 'agreement'], 1, 'AB')
add('cons_fp', conf.loc['all', 'fp'], 0, 'AB'); add('cons_fn', conf.loc['all', 'fn'], 0, 'AB')
add('prec_min_squad', conf.loc['squad/ministral3_8b', 'judge_precision'], 2, 'AB')
add('prec_qwen_hotpot', conf.loc['hotpotqa/qwen3_8b', 'judge_precision'], 2, 'AB')
add('M1_times_best', rk.loc['M1', 'times_best'], 0, 'AB')
add('M1_mean_rank', rk.loc['M1', 'mean_rank'], 1, 'A')

papers = {'A': lib.WS / 'side_papers' / '01_arxiv_comprehensive' / 'latex' / 'sections',
          'B': lib.WS / 'side_papers' / '02_journal_probing' / 'latex' / 'sections'}
text = {k: ' '.join(f.read_text(encoding='utf-8') for f in v.glob('*.tex')).replace('{,}', ',') for k, v in papers.items()}
fails = 0
for cid, v, nd, ps in C:
    s = f'{abs(v):.{nd}f}' if nd else str(int(round(v)))
    for p in ps:
        ok = re.search(r'(?<![\d.])' + re.escape(s) + r'(?![\d])', text[p]) is not None
        if not ok:
            fails += 1; print(f'FAIL {cid}: expected "{s}" in paper {p}')
print(f'{len(C)} claims checked, {fails} failures')
