"""Immutable NPY shards, JSON metadata, atomic commit markers, bounded mmap cache."""
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        f.write(canonical(value)); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def sample_id(provenance, record):
    required = ['dataset', 'split', 'original_id']
    if any(k not in record for k in required):
        raise ValueError('Metadata requires dataset, split, original_id')
    identity = dict(provenance=provenance, **{k: record[k] for k in required})
    # Exact response prevents accidental label reuse after different generations.
    identity['generated_text'] = record.get('generated_text')
    return hashlib.sha256(canonical(identity).encode()).hexdigest()


class ShardWriter:
    """Single writer. Use external job lock; committed shards are never overwritten."""
    def __init__(self, root, provenance, shape, dtype='float16', rows_per_shard=16):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.spec = dict(version=1, provenance=provenance, shape=list(shape), dtype=str(np.dtype(dtype)), rows_per_shard=rows_per_shard)
        if rows_per_shard < 1 or np.dtype(dtype).kind != 'f':
            raise ValueError('Invalid shard specification')
        spec_path = self.root/'spec.json'
        if spec_path.exists():
            if json.loads(spec_path.read_text()) != self.spec:
                raise ValueError('Resume config does not match immutable dataset specification')
        else:
            atomic_json(spec_path, self.spec)
        self.completed = set(); self.next_shard = 0
        for path in sorted(self.root.glob('shard_*.commit.json')):
            commit = json.loads(path.read_text())
            for name, digest in commit['checksums'].items():
                if sha256(self.root/name) != digest:
                    raise ValueError(f'Corrupt committed shard: {name}')
            records = json.loads((self.root/commit['metadata']).read_text())
            for record in records:
                if record['sample_id'] in self.completed:
                    raise ValueError('Duplicate sample ID in committed dataset')
                self.completed.add(record['sample_id'])
            self.next_shard = max(self.next_shard, commit['index']+1)
        self.arrays, self.records = [], []
        self.pending = set()

    def append(self, hidden, metadata):
        array = np.asarray(hidden)
        if list(array.shape) != self.spec['shape'] or str(array.dtype) != self.spec['dtype']:
            raise ValueError('Shape/dtype mismatch; conversion requires an explicit equivalence study')
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite hidden states')
        record = dict(metadata)
        sid = sample_id(self.spec['provenance'], record)
        if 'sample_id' in record and record['sample_id'] != sid:
            raise ValueError('Sample ID does not match content/provenance')
        if sid in self.completed or sid in self.pending:
            raise ValueError('Duplicate sample ID')
        if 'label' in record and record['label'] not in (0, 1):
            raise ValueError('Label must be binary if present')
        record['sample_id'] = sid
        self.arrays.append(array.copy()); self.records.append(record); self.pending.add(sid)
        if len(self.records) == self.spec['rows_per_shard']:
            self.flush()
        return sid

    def flush(self):
        if not self.records:
            return
        stem = f'shard_{self.next_shard:06d}'
        tensor = self.root/(stem+'.npy'); meta = self.root/(stem+'.json')
        marker = self.root/(stem+'.commit.json')
        if marker.exists():
            raise FileExistsError(marker)
        # Incomplete orphan payloads have no commit marker and may be rewritten.
        tmp = tensor.with_suffix('.npy.tmp')
        arr = np.lib.format.open_memmap(tmp, mode='w+', dtype=self.spec['dtype'], shape=(len(self.arrays), *self.spec['shape']))
        for i, row in enumerate(self.arrays):
            arr[i] = row
        arr.flush(); del arr
        with tmp.open('rb') as f:
            os.fsync(f.fileno())
        os.replace(tmp, tensor)
        atomic_json(meta, self.records)
        atomic_json(marker, dict(index=self.next_shard, rows=len(self.records), tensor=tensor.name,
                    metadata=meta.name, checksums={tensor.name: sha256(tensor), meta.name: sha256(meta)}))
        self.completed.update(self.pending)
        self.next_shard += 1; self.arrays.clear(); self.records.clear(); self.pending.clear()


class ShardDataset:
    """Loads small metadata only; tensor pages accessed lazily. Main-process loader."""
    def __init__(self, root, cache_shards=2, verify=False):
        self.root = Path(root); self.spec = json.loads((self.root/'spec.json').read_text())
        if cache_shards < 1:
            raise ValueError('cache_shards must be positive')
        self.cache_shards = cache_shards; self.cache = OrderedDict(); self.records = []; self.lookup = []
        seen = set()
        for marker in sorted(self.root.glob('shard_*.commit.json')):
            c = json.loads(marker.read_text())
            if verify:
                for name, digest in c['checksums'].items():
                    if sha256(self.root/name) != digest:
                        raise ValueError(f'Checksum mismatch: {name}')
            records = json.loads((self.root/c['metadata']).read_text())
            if len(records) != c['rows']:
                raise ValueError('Row count mismatch')
            arr = np.load(self.root/c['tensor'], mmap_mode='r', allow_pickle=False)
            if list(arr.shape) != [c['rows'], *self.spec['shape']] or str(arr.dtype) != self.spec['dtype']:
                raise ValueError('Shard header mismatch')
            del arr
            for i, rec in enumerate(records):
                if rec['sample_id'] != sample_id(self.spec['provenance'], rec) or rec['sample_id'] in seen:
                    raise ValueError('Invalid/duplicate sample ID')
                seen.add(rec['sample_id']); self.records.append(rec); self.lookup.append((c['tensor'], i))
        if not self.records:
            raise ValueError('No committed samples')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        file, row = self.lookup[index]
        if file not in self.cache:
            self.cache[file] = np.load(self.root/file, mmap_mode='r', allow_pickle=False)
            while len(self.cache) > self.cache_shards:
                self.cache.popitem(last=False)
        self.cache.move_to_end(file)
        # Copy exactly one row; caller can safely own/mutate it.
        return self.cache[file][row].copy(), dict(self.records[index])

    def read_batch(self, indices, out=None):
        """Copy rows straight into caller-owned batch storage, preserving order/dtype."""
        indices=list(indices)
        shape=(len(indices),*self.spec['shape'])
        if out is None:out=np.empty(shape,dtype=self.spec['dtype'])
        if out.shape!=shape or str(out.dtype)!=self.spec['dtype']:raise ValueError('Batch destination mismatch')
        for j,index in enumerate(indices):
            if not isinstance(index,(int,np.integer)) or index<0 or index>=len(self):raise IndexError(index)
            file,row=self.lookup[int(index)]
            if file not in self.cache:
                self.cache[file]=np.load(self.root/file,mmap_mode='r',allow_pickle=False)
                while len(self.cache)>self.cache_shards:self.cache.popitem(last=False)
            self.cache.move_to_end(file)
            np.copyto(out[j],self.cache[file][row],casting='no')
        return out
