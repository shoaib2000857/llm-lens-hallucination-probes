"""Audit: every decimal in a paper's prose must be traceable to a value in the evidence files (after rounding).

Usage: python audit_numbers.py <sections_dir> [<sections_dir> ...]
Reports numbers that match no evidence value at 2 or 3 decimals (as is, or as a difference/percentage of evidence values
is NOT attempted -- those are listed for manual review).
"""
import re, sys, json
from pathlib import Path
import numpy as np, pandas as pd
import lib

EVID = []
for f in list(lib.OUT.glob('*.csv')) + list((lib.WS / 'manuscripts' / 'human_updated_20261104' / 'tables').glob('*.csv')) \
        + list((lib.WS / 'manuscripts' / 'human_updated_20261104' / 'analysis').rglob('*.csv')):
    try:
        d = pd.read_csv(f)
        EVID.extend(d.select_dtypes('number').to_numpy().ravel().tolist())
    except Exception:
        pass
for f in list((lib.WS / 'manuscripts' / 'human_updated_20261104' / 'tables').glob('*.tex')) + list((lib.WS / 'side_papers' / '01_arxiv_comprehensive' / 'latex' / 'tables').glob('*.tex')):
    EVID.extend(float(x) for x in re.findall(r'[-+]?\d+\.\d+', f.read_text(encoding='utf-8')))
EVID = np.array([x for x in EVID if isinstance(x, (int, float)) and np.isfinite(x)])
R2, R3 = set(np.round(np.abs(EVID), 2)), set(np.round(np.abs(EVID), 3))

for sd in sys.argv[1:]:
    print('=' * 20, sd)
    for f in sorted(Path(sd).glob('*.tex')):
        txt = f.read_text(encoding='utf-8')
        txt = re.sub(r'\\input\{[^}]*\}|\\label\{[^}]*\}|\\ref\{[^}]*\}|\\cref\{[^}]*\}|\\cite[pt]?\{[^}]*\}', '', txt)
        bad = []
        for m in re.finditer(r'(?<![\w.])([-+$]*)(0\.\d{2,3})(?!\d)', txt):
            v = float(m.group(2)); nd = len(m.group(2).split('.')[1])
            ok = (round(v, 3) in R3) if nd == 3 else (round(v, 2) in R2)
            if not ok:
                line = txt.count('\n', 0, m.start()) + 1
                ctx = txt[max(0, m.start() - 60):m.end() + 20].replace('\n', ' ')
                bad.append((line, m.group(2), ctx))
        if bad:
            print(f'--- {f.name}: {len(bad)} unmatched')
            for b in bad:
                print(f'  L{b[0]} {b[1]} :: ...{b[2]}...')
