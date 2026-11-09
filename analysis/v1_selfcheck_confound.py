"""V1: does generated self-verification text (e.g. 'The answer is incorrect') explain probe performance?

Qwen3-8B frequently continues past its answer with an explicit self-check. Probes read all 96 generated
positions, and judges see the same text. We stratify existing saved scores; no refitting.
"""
import re, numpy as np, pandas as pd
import lib

SELF = re.compile(r"(answer is (correct|incorrect|wrong|right|not correct)|is the answer correct|let me (check|verify)|step-by-step|i need to (evaluate|check|verify)|okay, let me|based on the context( provided)?, the answer|the answer provided)", re.I)
SELFNEG = re.compile(r"(answer is (incorrect|wrong|not correct)|not supported by the context|isn't correct|is not correct|answer is not (accurate|supported))", re.I)
DIST = lib.EV / 'results' / 'distribution_v1'


def tag(text):
    t = text or ''
    return 'self_negative' if SELFNEG.search(t) else ('self_check_other' if SELF.search(t) else 'no_self_check')


def scores(corpus, probe, src):
    if corpus == src:
        dirs = [lib.PROBES / f'{src}__{probe}__s{s}__scratch_full' for s in lib.SEEDS]
    else:
        dirs = [DIST / f'{corpus}__{probe}__s{s}' for s in lib.SEEDS]
    ps = [lib.preds(d) for d in dirs]; ps = [p for p in ps if p]
    if not ps:
        return None
    ids = sorted(set.intersection(*[set(p) for p in ps]))
    return ids, np.array([ps[0][i][0] for i in ids]), np.mean([[p[i][1] for i in ids] for p in ps], axis=0)


def main():
    rows = []
    for src in lib.SOURCES:
        for corpus in [src] + [f'{t}_{src}' for t in lib.TARGETS]:
            meta = lib.meta(corpus)
            for probe in lib.CORE:
                r = scores(corpus, probe, src)
                if not r:
                    continue
                ids, y, p = r
                tags = np.array([tag(meta[i]['text']) for i in ids])
                for k in ['all', 'no_self_check', 'self_check_other', 'self_negative', 'any_self_check']:
                    m = np.ones(len(ids), bool) if k == 'all' else (tags != 'no_self_check') if k == 'any_self_check' else (tags == k)
                    rows.append(dict(generator=src, corpus=corpus, probe=probe, stratum=k, n=int(m.sum()), positives=int(y[m].sum()),
                                     prevalence=float(y[m].mean()) if m.any() else np.nan, auroc=lib.auroc(y[m], p[m]) if m.sum() > 10 else np.nan))
                cue = (tags == 'self_negative').astype(float)
                rows.append(dict(generator=src, corpus=corpus, probe='cue:self_negative_text', stratum='all', n=len(ids), positives=int(y.sum()),
                                 prevalence=float(y.mean()), auroc=lib.auroc(y, cue)))
    d = pd.DataFrame(rows).drop_duplicates(['corpus', 'probe', 'stratum'])
    lib.save_csv(d, 'V1_selfcheck_strata.csv')
    # human references: does consensus-human disagreement concentrate in self-check responses?
    c = pd.read_csv(lib.HUMAN / 'case_ledger_deidentified.csv')
    c['tag'] = [tag(lib.meta(r.corpus)[r.sample_id]['text']) for r in c.itertuples()]
    c['disagree'] = (c.consensus != c.human).astype(int)
    h = c.groupby(['generator', 'dataset', 'tag']).agg(n=('disagree', 'size'), disagree_rate=('disagree', 'mean'), human_pos=('human', 'mean'), consensus_pos=('consensus', 'mean')).reset_index()
    lib.save_csv(h, 'V1_selfcheck_human_disagreement.csv')
    return d, h


if __name__ == '__main__':
    d, h = main()
    q = d[d.probe.isin(['M1', 'M3', 'mean_linear', 'cue:self_negative_text'])]
    print(q.pivot_table(index=['corpus', 'probe'], columns='stratum', values='auroc').round(3).to_string())
    print(d[(d.probe == 'M1')].pivot_table(index='corpus', columns='stratum', values='n').to_string())
    print(h.round(2).to_string())
