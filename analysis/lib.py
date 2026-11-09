"""Shared read-only loaders for side-paper analyses. Never writes into the evidence tree."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

WS = Path(__file__).resolve().parents[3]
EV = WS / 'paper_evidence_20261012'
PROBES = EV / 'artifacts' / 'probes'
OUT = WS / 'side_papers' / 'analysis' / 'outputs'
HUMAN = WS / 'manuscripts' / 'human_updated_20261104' / 'analysis' / 'human'
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = ['ministral3_8b', 'qwen3_8b']
CORE = ['M0', 'M1', 'M2', 'M3', 'M4', 'mean_linear', 'last_linear', 'last_mlp']
SEEDS = [42, 43, 44]
TARGETS = ['distribution_v1_hotpotqa', 'distribution_v1_pubmedqa', 'distribution_v2_nq_retrieval', 'distribution_v2_triviaqa']
NICE = {'ministral3_8b': 'Ministral-3-8B', 'qwen3_8b': 'Qwen3-8B', 'distribution_v1_hotpotqa': 'HotpotQA',
        'distribution_v1_pubmedqa': 'PubMedQA', 'distribution_v2_nq_retrieval': 'NQ (retrieved ctx)',
        'distribution_v2_triviaqa': 'TriviaQA (wiki ctx)', 'squad': 'SQuAD v2'}


def jload(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def preds(path):
    """Return dict sample_id -> (label, probability) for a fit folder."""
    p = Path(path) / 'predictions.json'
    if not p.exists():
        return None
    return {r['sample_id']: (r['label'], r['probability']) for r in jload(p)}


@lru_cache(maxsize=None)
def meta(corpus):
    """sample_id -> metadata dict from stage1 shards (no tensors)."""
    base = EV / 'artifacts' / f'stage1_{corpus}' / 'hidden_states'
    out = {}
    for f in sorted(base.glob('shard_*.json')):
        for r in jload(f):
            ctx = r.get('context_sha256') or hashlib.sha256(r.get('context', '').encode('utf-8')).hexdigest()
            out[r['sample_id']] = dict(context=ctx, unanswerable=r.get('is_unanswerable'),
                                       n_tokens=r.get('n_gen_tokens'), text=r.get('generated_text', ''),
                                       mean_logprob=r.get('mean_token_logprob'), dataset=r.get('dataset'))
    return out


@lru_cache(maxsize=None)
def manifest(corpus):
    m = jload(EV / 'artifacts' / f'stage1_{corpus}' / 'pooled' / 'manifest.json')
    return {r['sample_id']: r for r in m['records']}


def auroc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan


def ap(y, p):
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else np.nan


def ece(y, p, bins=10):
    y = np.asarray(y, float); p = np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & ((p < b) if b < 1 else (p <= b))
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def cluster_bootstrap(groups, stat, reps=2000, seed=0):
    """Resample whole contexts. stat(index_array)->float. Returns (low, high)."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    vals = []
    for _ in range(reps):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[k] for k in pick])
        v = stat(idx)
        if not np.isnan(v):
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def save_csv(df, name):
    p = OUT / name
    df.to_csv(p, index=False)
    print('wrote', p.relative_to(WS))
    return p
