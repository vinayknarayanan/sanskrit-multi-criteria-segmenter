"""Benchmark check (paper, Benchmark problems): are the SIGHUM test and validation splits identical?
Run after tools/download_conventions.py. Prints, per Hugging Face dataset folder, the row count of every
non-train split and whether it equals the test split row for row. Commits checked for the paper are listed in
README.md."""
import glob, os, sys, pandas as pd
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'external', 'hf')
for d in sorted(glob.glob(os.path.join(root, 'sanskrit-sandhi-split-sighum*')) + glob.glob(os.path.join(root, 'sandhi-split-2018'))):
    fs = {os.path.basename(f).split('-')[0]: f for f in glob.glob(os.path.join(d, 'data', '*.parquet'))}
    t = {k: pd.read_parquet(v).reset_index(drop=True) for k, v in fs.items() if k != 'train'}
    same = {k: t[k].equals(t['test']) for k in t if k != 'test'} if 'test' in t else {}
    print(os.path.basename(d), {k: len(v) for k, v in t.items()}, 'identical to test:', same)
