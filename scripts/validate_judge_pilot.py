#!/usr/bin/env python3
import json
from pathlib import Path
import sys
folder=Path(sys.argv[1])
rows=[r for p in sorted(folder.glob('batch_*.json')) for r in json.loads(p.read_text())]
if len(rows)<100:raise SystemExit('Fewer than 100 pilot labels committed')
bad=sum(r['label'] is None for r in rows)
print(json.dumps(dict(pilot_checked=len(rows),invalid=bad,invalid_rate=bad/len(rows))))
if bad/len(rows)>.1:raise SystemExit('More than 10% invalid judge outputs; inspect raw outputs before spending more GPU time')
