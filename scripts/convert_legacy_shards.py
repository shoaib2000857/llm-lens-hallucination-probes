#!/usr/bin/env python3
"""Convert trusted tensor/dict .pt shards with restricted loading and mmap.

No model inference, dtype conversion, labels relabeling, or deletion of originals.
"""
import argparse
import fcntl
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import torch
from portability.storage import ShardWriter, sample_id, sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--provenance', type=Path, required=True, help='JSON: actual dataset/generator revisions and protocol')
    ap.add_argument('--split', required=True)
    ap.add_argument('--label-key', choices=['judge_label', 'hybrid_label'], default='judge_label')
    ap.add_argument('--rows-per-shard', type=int, default=16)
    ap.add_argument('--limit', type=int, default=100, help='Maximum source records inspected; 0=all (user full run)')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'.writer.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    provenance = json.loads(args.provenance.read_text())
    provenance.update(label_key=args.label_key, split=args.split, source_format='legacy_torch_list')
    paths = sorted(args.input.glob('shard_*.pt'))
    if not paths: raise ValueError('No legacy shards')
    # Record source hashes even on resume: no reliance on file name or row index.
    provenance['source_shards'] = [{ 'name':p.name, 'sha256':sha256(p)} for p in paths]
    writer = None; inspected = 0
    for path in paths:
        records = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if not isinstance(records, list): raise ValueError('Expected list of records')
        for row_index, record in enumerate(records):
            if args.limit and inspected >= args.limit: break
            inspected += 1
            hidden = record['hidden'].detach().cpu().numpy()
            metadata = {k:v for k,v in record.items() if k != 'hidden'}
            metadata.update(original_id=str(record['id']), split=args.split,
                            label=int(record[args.label_key]), legacy_shard=path.name, legacy_row=row_index)
            if writer is None:
                writer = ShardWriter(args.output, provenance, hidden.shape, str(hidden.dtype), args.rows_per_shard)
            if sample_id(provenance, metadata) in writer.completed: continue
            writer.append(hidden, metadata)
        del records
        if args.limit and inspected >= args.limit: break
    if writer: writer.flush()
    print(json.dumps(dict(inspected=inspected, committed=len(writer.completed) if writer else 0)))


if __name__ == '__main__': main()
