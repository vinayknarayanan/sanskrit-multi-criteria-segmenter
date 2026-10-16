"""Score any system's results file (columns: compound, gold_split, split_count, predicted = space-separated IAST pieces)
with strict / conservative / lenient boundary-F1 (rescore.score), and print it next to the existing systems.
Usage: python score_system.py results_byt5_frontiers.csv NAME"""
import sys, pandas as pd
from indic_transliteration import sanscript
from rescore import score

slp = lambda s: sanscript.transliterate(s, 'iast', 'slp1')
path, name = sys.argv[1], sys.argv[2]
r = pd.read_csv(path); rows = []
for x in r.itertuples():
    if not isinstance(x.predicted, str) or not x.predicted.strip(): continue
    s = score([slp(p) for p in x.gold_split.split('+')], [slp(p) for p in x.predicted.split()], slp(x.compound))
    rows.append(dict(compound=x.compound, **{f'{name}_{k}': v for k, v in s.items()}))
new = pd.DataFrame(rows); new.to_csv(f'rescore_{name}.csv', index=False)
m = pd.read_csv('rescore_results.csv').merge(new, on='compound')
m = m.dropna(subset=['byt5_strict_f', 'inria_strict_f', f'{name}_strict_f'])
print(f'items scored by all: {len(m)}')
print(f"{'system':14s} {'strict F1':>9s} {'cons. F1':>9s} {'len. F1':>8s} {'P strict':>9s} {'R':>6s}  FP at plain junctions")
for s in ('byt5', 'inria', name):
    fp, fpp = m[f'{s}_fp'].sum(), m[f'{s}_fp_plain'].sum()
    print(f"{s:14s} {m[f'{s}_strict_f'].mean():9.3f} {m[f'{s}_plaindcs_f'].mean():9.3f} {m[f'{s}_plain_f'].mean():8.3f} "
          f"{m[f'{s}_strict_p'].mean():9.3f} {m[f'{s}_strict_r'].mean():6.3f}  {fpp/max(fp,1)*100:.1f}%")
print('\nstrict F1 by stratum:')
print(m.groupby('stratum')[[f'{s}_strict_f' for s in ('byt5', 'inria', name)]].mean().round(3).to_string())
