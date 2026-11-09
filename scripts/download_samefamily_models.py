#!/usr/bin/env python3
"""Bounded project-cache downloads; exclude duplicate consolidated checkpoints."""
import fnmatch,json,shutil,sys
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from prepare_samefamily_round import VARIANTS
def main():
    print('RESOURCE PLAN: two8B modern same-family checkpoints, expected32-36GiB download into existing project HF cache; two download workers, sequential repositories, no GPU, bounded disk writes. Alternate consolidated checkpoints excluded. Reserve50GiB. No model weight copies.',flush=True)
    if shutil.disk_usage(ROOT).free<50*2**30:raise ValueError('Insufficient cache headroom')
    api=HfApi();total=0
    for model,revision in VARIANTS.values():
        info=api.model_info(model,revision=revision,files_metadata=True)
        needed=[s for s in info.siblings if fnmatch.fnmatch(s.rfilename,'model-*.safetensors') or s.rfilename=='model.safetensors']
        size=sum(s.size or 0 for s in needed);total+=size;print(json.dumps(dict(model=model,checkpoint_GiB=size/2**30,files=len(needed))),flush=True)
        snapshot_download(model,revision=revision,cache_dir=ROOT/'artifacts/hf_cache/hub',allow_patterns=['model-*.safetensors','model.safetensors','*.json','*.model','*.txt'],max_workers=2)
    folder=ROOT/'artifacts/samefamily_v1';folder.mkdir(exist_ok=True);(folder/'downloads_complete.json').write_text(json.dumps(dict(download_bytes=total,variants=VARIANTS)))
if __name__=='__main__':main()
