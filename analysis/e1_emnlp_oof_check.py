"""E1: recompute EMNLP same-family OOF reference gaps with within-fold scoring (paired on identical responses).

Original export: reference AUROC = AUROC of pooled out-of-fold scores from 5 separately fitted models.
Corrected: for each fold, AUROC of the frozen probe and of the fold's reference on that fold's responses; average over folds.
"""
import json
import numpy as np, pandas as pd
import lib

g = pd.read_csv(lib.EV / 'exports' / 'paper_evidence_20261012' / 'samefamily_matched_gaps.csv')
rows = []
for r in g.itertuples():
    if r.reference_scope != 'context_grouped_oof_reference':
        continue
    ref = pd.DataFrame(lib.jload(lib.EV / r.reference_file / 'predictions.json'))
    fr = {d['sample_id']: d['probability'] for d in lib.jload(lib.EV / r.transfer_file / 'predictions.json')}
    ref['frozen'] = ref.sample_id.map(fr)
    ref = ref.dropna(subset=['frozen'])
    fa, ra = [], []
    for _, x in ref.groupby('fold'):
        if x.label.nunique() == 2:
            fa.append(lib.auroc(x.label, x.frozen)); ra.append(lib.auroc(x.label, x.probability))
    rows.append(dict(target=r.target, probe=r.probe, seed=r.seed,
                     frozen_pooled=r.transfer_auroc, ref_pooled=r.reference_auroc, gap_pooled=r.gap,
                     frozen_within=np.mean(fa), ref_within=np.mean(ra), gap_within=np.mean(ra) - np.mean(fa)))
d = pd.DataFrame(rows)
lib.save_csv(d, 'E1_emnlp_samefamily_oof_within_fold.csv')
d['generator'] = np.where(d.target.str.contains('qwen'), 'Qwen3->Base', 'Ministral Base->Instruct')
d['dataset'] = d.target.str.replace('samefamily_v2_', '').str.replace('_ministral3_8b', '').str.replace('_qwen3_8b', '')
s = d.groupby(['generator', 'dataset'])[['gap_pooled', 'gap_within']].mean().round(3)
print(s.to_string())
print('mean shift in gap (within - pooled):', round((d.gap_within - d.gap_pooled).mean(), 3))
print('sign flips:', int(((d.gap_pooled > 0) != (d.gap_within > 0)).sum()), 'of', len(d))
print('share positive gaps pooled -> within:', round((d.gap_pooled > 0).mean(), 2), '->', round((d.gap_within > 0).mean(), 2))
sp = d.groupby(['generator', 'probe'])[['gap_pooled', 'gap_within']].mean().round(3)
lib.save_csv(sp.reset_index(), 'E1_emnlp_oof_by_probe.csv'); print(sp.to_string())
