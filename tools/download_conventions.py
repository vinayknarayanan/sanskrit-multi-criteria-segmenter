"""Download extra segmentation datasets (SIGHUM, Hackathon, ...) for Paper 2 into this folder.
Run from this folder on your Windows machine:
    pip install -U huggingface_hub
    python download_conventions.py
Lists every dataset by chronbmm (Nehrdich, ByT5-Sanskrit author), downloads those whose name mentions
sandhi / split / segment / hackathon / sighum into ./hf/<name>/, and writes inventory.txt (files + sizes).
Nothing outside this folder is touched."""
import os
from huggingface_hub import HfApi, snapshot_download

KEYS = ('sandhi', 'split', 'segment', 'hackathon', 'sighum')
EXTRA = []   # add 'owner/name' strings here for datasets from other authors
here = os.path.dirname(os.path.abspath(__file__))
api = HfApi()
names = [d.id for d in api.list_datasets(author='chronbmm', limit=500)]
print(f'chronbmm has {len(names)} datasets')
for n in names: print('  ', n)
want = [n for n in names if any(k in n.lower() for k in KEYS)] + EXTRA
print('\ndownloading:', *want, sep='\n  ')
with open(os.path.join(here, 'inventory.txt'), 'w', encoding='utf-8') as inv:
    inv.write('all chronbmm datasets:\n' + '\n'.join(names) + '\n\n')
    for n in want:
        dest = os.path.join(here, '..', 'external', 'hf', n.split('/')[-1])
        try:
            snapshot_download(repo_id=n, repo_type='dataset', local_dir=dest)
        except Exception as e:
            print('FAILED', n, e); inv.write(f'FAILED {n}: {e}\n'); continue
        inv.write(f'== {n}\n')
        for root, _, files in os.walk(dest):
            if '.cache' in root: continue
            for f in files:
                p = os.path.join(root, f); inv.write(f'  {os.path.relpath(p, dest)}  {os.path.getsize(p):,} bytes\n')
print('\ndone - see inventory.txt')
