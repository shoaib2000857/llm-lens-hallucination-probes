from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
AN = HERE.parents[3] / 'side_papers' / 'analysis' / 'outputs'
d = pd.read_csv(AN / 'S3_answer_only_vs_full.csv')
DOM = {'distribution_v1_hotpotqa': 'HotpotQA', 'distribution_v1_pubmedqa': 'PubMedQA', 'distribution_v2_nq_retrieval': 'NQ (retrieved)', 'distribution_v2_triviaqa': 'TriviaQA (Wiki)'}
rows = []
for src, sn in [('ministral3_8b', 'Ministral'), ('qwen3_8b', 'Qwen3')]:
    for corpus in [src] + [f'{t}_{src}' for t in DOM]:
        lm = d[(d.source == src) & (d.corpus == corpus) & (d['head'] == 'layer_mean')].iloc[0]
        bl = d[(d.source == src) & (d.corpus == corpus) & (d['head'] == 'dev_best_layer')].iloc[0]
        name = 'SQuAD (source test)' if corpus == src else DOM[corpus[:-len(src) - 1]]
        rows.append(f'{sn} & {name} & {lm.answer_only_auroc:.3f} & {lm.full_mean_linear:.3f} & {bl.answer_only_auroc:.3f} & {bl.full_dev_best_layer_scan:.3f} \\\\')
    rows.append(r'\midrule')
rows = rows[:-1]
head = [r'\begin{tabular}{llrrrr}', r'\toprule',
        r' & & \multicolumn{2}{c}{Layer-mean linear} & \multicolumn{2}{c}{Dev-selected single layer} \\',
        r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}',
        r'Generator & Corpus & Answer only & Full response & Answer only & Full response \\', r'\midrule']
(HERE.parent / 'tables' / 'answer_only.tex').write_text('\n'.join(head + rows + [r'\bottomrule', r'\end{tabular}']) + '\n', encoding='utf-8')
print('ok')
