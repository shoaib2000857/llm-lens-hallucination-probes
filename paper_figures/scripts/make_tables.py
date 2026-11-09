"""Paper A LaTeX tables from saved analysis outputs (read-only)."""
from pathlib import Path
import pandas as pd, numpy as np

HERE = Path(__file__).resolve().parent
TAB = HERE.parent / 'tables'; TAB.mkdir(exist_ok=True)
WS = HERE.parents[3]
AN = WS / 'side_papers' / 'analysis' / 'outputs'
EMT = WS / 'manuscripts' / 'human_updated_20261104' / 'tables'
PN = {'mean_linear': 'Mean-linear', 'last_linear': 'Last-linear', 'last_mlp': 'Last-MLP', 'best_layer_linear': 'Best-layer linear'}
DOM = {'distribution_v1_hotpotqa': 'HotpotQA', 'distribution_v1_pubmedqa': 'PubMedQA', 'distribution_v2_nq_retrieval': 'NQ (retrieved)', 'distribution_v2_triviaqa': 'TriviaQA (Wiki)'}
f3 = lambda x: '--' if pd.isna(x) else f'{x:.3f}'


def write(name, body):
    (TAB / f'{name}.tex').write_text(body, encoding='utf-8'); print('table', name)


def source():
    d = pd.read_csv(EMT / 'source.csv'); c = pd.read_csv(AN / 'B3_calibration_summary.csv')
    order = ['mean_linear', 'last_linear', 'last_mlp', 'best_layer_linear', 'M0', 'M1', 'M2', 'M3', 'M4']
    rows = []
    for p in order:
        cells = [PN.get(p, p)]
        for s in ['ministral3_8b', 'qwen3_8b']:
            r = d[(d.setting == s) & (d.probe == p)].iloc[0]; e = c[(c.source == s) & (c.probe == p)].ece10.iloc[0]
            cells += [f'{r.auroc:.3f} $\\pm$ {r.auroc_seed_sd:.3f}', f'{r.ap:.3f}', f'{r.f1:.3f}', f'{e:.3f}']
        rows.append(' & '.join(cells) + r' \\')
        if p == 'best_layer_linear': rows.append(r'\midrule')
    write('source', '\\begin{tabular}{l' + 'rrrr' * 2 + '}\n\\toprule\n & \\multicolumn{4}{c}{Ministral-3-8B (993 / 230 pos.)} & \\multicolumn{4}{c}{Qwen3-8B (988 / 178 pos.)} \\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\nProbe & AUROC & AP & F1@0.5 & ECE & AUROC & AP & F1@0.5 & ECE \\\\\n\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')


def learning():
    g = pd.read_csv(AN / 'B1_learning_curves_summary.csv'); a = pd.read_csv(AN / 'B1_attainment.csv')
    rows = []
    for s, sn in [('ministral3_8b', 'Ministral'), ('qwen3_8b', 'Qwen3')]:
        for p in ['M0', 'M1', 'mean_linear', 'last_linear', 'last_mlp']:
            cells = [sn, PN.get(p, p)]
            for b in ['25', '100', '250', '500', '1000', 'full']:
                r = g[(g.source == s) & (g.probe == p) & (g.budget == b)].iloc[0]
                cells.append(f'{r.auroc:.3f}')
            at = a[(a.source == s) & (a.probe == p) & (a.budget == '250')].attainment.iloc[0]
            cells.append(f'{100 * at:.0f}\\%')
            rows.append(' & '.join(cells) + r' \\')
        rows.append(r'\midrule')
    rows = rows[:-1]
    write('learning', '\\begin{tabular}{llrrrrrrr}\n\\toprule\nGenerator & Probe & 25 & 100 & 250 & 500 & 1{,}000 & Full & Attain.@250 \\\\\n\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')


def contrasts():
    d = pd.read_csv(AN / 'B2_paired_structure_contrasts.csv')
    keep = ['M1 uniform weights vs M1', 'M2 no positions vs M2', 'M3 no layer positions vs M3', 'M4 no token positions vs M4',
            'Capacity-matched global vs M0', 'Capacity-matched global vs M2'] + [f'{p} vs best-layer linear' for p in ['M0', 'M1', 'M2', 'M3', 'M4']] + [f'{p} vs mean linear' for p in ['M0', 'M1', 'M2', 'M3', 'M4']]
    rows = []
    for k in keep:
        cells = [k.replace(' vs ', ' $-$ ')]
        for s in ['ministral3_8b', 'qwen3_8b']:
            r = d[(d.source == s) & (d.contrast == k)].iloc[0]
            cells.append(f'{r.delta:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]')
        rows.append(' & '.join(cells) + r' \\')
        if k in ('M4 no token positions vs M4', 'Capacity-matched global vs M2', 'M4 vs best-layer linear'): rows.append(r'\midrule')
    write('contrasts', '\\begin{tabular}{lll}\n\\toprule\nContrast (A $-$ B) & Ministral $\\Delta$AUROC [95\\%] & Qwen3 $\\Delta$AUROC [95\\%] \\\\\n\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')


def domain():
    d = pd.read_csv(AN / 'B4c_oof_pooled_vs_within_fold.csv'); g = pd.read_csv(AN / 'B4a_domain_frozen_vs_oof.csv')
    rows = []
    for s, sn in [('ministral3_8b', 'Ministral'), ('qwen3_8b', 'Qwen3')]:
        for t, tn in DOM.items():
            r = d[(d.source == s) & (d.target == t)].set_index('probe'); gg = g[(g.source == s) & (g.target == t)].iloc[0]
            cells = [sn, tn, f'{gg.n} / {gg.positives}']
            for p in ['mean_linear', 'M1', 'M3']:
                cells.append(f'{r.loc[p].frozen_auroc_mean_seed:.3f}')
            cells += [f'{r.frozen_auroc_mean_seed.mean():.3f}', f'{r.oof_within_fold_auroc.mean():.3f}', f'{int((r.frozen_auroc_mean_seed > r.oof_within_fold_auroc).sum())}/8']
            rows.append(' & '.join(cells) + r' \\')
        rows.append(r'\midrule')
    rows = rows[:-1]
    write('domain', '\\begin{tabular}{llrrrrrrr}\n\\toprule\n & & & \\multicolumn{3}{c}{Frozen SQuAD probe} & \\multicolumn{2}{c}{Mean over 8 probes} & \\\\\n\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\nGenerator & Target & $n$ / pos. & Mean-lin. & M1 & M3 & Frozen & OOF (in-fold) & Frozen $>$ OOF \\\\\n\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')


def cues():
    c = pd.read_csv(AN / 'B5a_training_free_cues.csv'); m = pd.read_csv(AN / 'B9_probe_auroc_by_setting.csv').set_index('setting')
    rows = []
    for s, sn in [('ministral3_8b', 'Ministral'), ('qwen3_8b', 'Qwen3')]:
        for corpus, tn, key in [(s, 'SQuAD (source test)', f'source:{s}')] + [(f'{t}_{s}', n, f'frozen:{t}_{s}') for t, n in DOM.items()]:
            x = c[c.corpus == corpus].set_index('cue').auroc
            rows.append(' & '.join([sn, tn, f3(x['n_gen_tokens']), f3(x['neg_mean_logprob']), f3(x['neg_min_logprob']), f3(x['mean_entropy']), f3(m.loc[key, 'M1'])]) + r' \\')
        rows.append(r'\midrule')
    rows = rows[:-1]
    write('cues', '\\begin{tabular}{llrrrrr}\n\\toprule\nGenerator & Corpus & Length & $-$mean log $p$ & $-$min log $p$ & Entropy & M1 (frozen) \\\\\n\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')


def confusion():
    c = pd.read_csv(AN / 'B5b_consensus_vs_human_confusion.csv')
    lab = {'all': 'All', 'squad': 'SQuAD', 'hotpotqa': 'HotpotQA', 'pubmedqa': 'PubMedQA'}
    rows = []
    for r in c.itertuples():
        parts = r.group.split('/'); name = lab[parts[0]] + (' / ' + ('Ministral' if 'ministral' in parts[1] else 'Qwen3') if len(parts) > 1 else '')
        rows.append(f'{name} & {r.n} & {r.human_pos} & {r.judge_pos} & {r.fp} & {r.fn} & {r.agreement:.3f} & {r.judge_precision:.3f} & {r.judge_recall:.3f} \\\\')
        if r.group in ('all', 'squad'): rows.append(r'\midrule')
    write('confusion', '\\begin{tabular}{lrrrrrrrr}\n\\toprule\nStratum & $n$ & Human pos. & Consensus pos. & FP & FN & Agreement & Precision & Recall \\\\\n\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')


def legacy():
    # Values transcribed verbatim from OLD-References/V1/main.tex (held-out test, 5,000 rows) and ARR 2721 (cross-dataset means).
    v1 = {'M0': (0.7441, 0.5124, 0.5316, 0.2092), 'M1': (0.8074, 0.6070, 0.6025, 0.1430), 'M2': (0.8490, 0.6960, 0.6475, 0.1747),
          'M3': (0.8577, 0.7057, 0.6644, 0.1025), 'M4': (0.8294, 0.6454, 0.6236, 0.1136)}
    xd = {'M0': 0.688, 'M1': 0.713, 'M2': 0.764, 'M3': 0.761, 'M4': 0.741}
    s = pd.read_csv(EMT / 'source.csv')
    rows = []
    for p in ['M0', 'M1', 'M2', 'M3', 'M4']:
        a, ap, f1, e = v1[p]
        q = s[(s.setting == 'qwen3_8b') & (s.probe == p)].auroc.iloc[0]; m = s[(s.setting == 'ministral3_8b') & (s.probe == p)].auroc.iloc[0]
        rows.append(f'{p} & {a:.3f} & {ap:.3f} & {f1:.3f} & {e:.3f} & {xd[p]:.3f} & {m:.3f} & {q:.3f} \\\\')
    write('legacy', '\\begin{tabular}{lrrrrrrr}\n\\toprule\n & \\multicolumn{5}{c}{Original study: LLaMA-2-7B (hybrid/judge labels)} & \\multicolumn{2}{c}{Modern (consensus labels)} \\\\\n\\cmidrule(lr){2-6}\\cmidrule(lr){7-8}\nProbe & AUROC & PR-AUC$^\\dagger$ & F1@0.5 & ECE & Cross-data mean & Ministral & Qwen3 \\\\\n\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')


if __name__ == '__main__':
    source(); learning(); contrasts(); domain(); cues(); confusion(); legacy()


def oof_bias():
    d = pd.read_csv(AN / 'B4c_oof_pooled_vs_within_fold.csv')
    g = d.groupby(['source', 'target'])[['frozen_auroc_mean_seed', 'oof_pooled_auroc', 'oof_within_fold_auroc']].mean().reset_index()
    rows = [f"{'Ministral' if 'ministral' in r.source else 'Qwen3'} & {DOM[r.target]} & {r.frozen_auroc_mean_seed:.3f} & {r.oof_pooled_auroc:.3f} & {r.oof_within_fold_auroc:.3f} \\\\" for r in g.itertuples()]
    head = r'\begin{tabular}{llrrr}' + '\n' + r'\toprule' + '\n' + r'Generator & Target & Frozen & OOF (pooled folds) & OOF (within fold) \\' + '\n' + r'\midrule' + '\n'
    write('oof_bias', head + '\n'.join(rows) + '\n' + r'\bottomrule' + '\n' + r'\end{tabular}' + '\n')


def policies():
    d = pd.read_csv(WS / 'paper_evidence_20261012' / 'results' / 'model_update_schedules_v1' / 'metrics.csv')
    g = d.groupby('policy').agg(auroc=('auroc', 'mean'), f1=('f1_05', 'mean'), ece=('ece', 'mean'), updates=('update_count', 'max'), labels=('total_target_labels', 'max'))
    names = {'never': 'Never refresh', 'platt_only': 'Platt recalibration only', 'every_2_checkpoints': 'Re-fit every second checkpoint',
             'every_checkpoint': 'Re-fit at every checkpoint', 'representation_drift': 'Re-fit when CKA drift exceeds threshold', 'validation_drop': 'Re-fit when validation AUROC drops'}
    rows = [f"{names[k]} & {r.auroc:.3f} & {r.f1:.3f} & {r.ece:.3f} & {int(r.updates)} & {int(r.labels):,} \\\\".replace(',', '{,}') for k, r in g.loc[list(names)].iterrows()]
    head = r'\begin{tabular}{lrrrrr}' + '\n' + r'\toprule' + '\n' + r'Policy & AUROC & F1@0.5 & ECE & Max.\ re-fits & Max.\ target labels \\' + '\n' + r'\midrule' + '\n'
    write('policies', head + '\n'.join(rows) + '\n' + r'\bottomrule' + '\n' + r'\end{tabular}' + '\n')


if __name__ == '__main__':
    oof_bias(); policies()
