"""Score the returned pilot workbooks against the hidden key, following prereg_pilot.json (v2).
Usage: python analyse_pilot.py pilot_annotator_A.xlsx pilot_annotator_B.xlsx _key/pilot_key_DO_NOT_SEND.csv"""
import sys, math, numpy as np, pandas as pd
COL = {'Y': 'split', 'N': 'no_split', 'X': 'no_split', 'E': 'open', 'U': 'open'}
a = pd.read_excel(sys.argv[1], sheet_name='Items'); b = pd.read_excel(sys.argv[2], sheet_name='Items'); K = pd.read_csv(sys.argv[3]).fillna('')
for d in (a, b): d['Answer'] = d.Answer.where(d.Answer.notna(), '').astype(str).str.strip().str.upper()
m = K.merge(a[['ID', 'Answer']].rename(columns={'ID': 'id', 'Answer': 'a'}), on='id').merge(b[['ID', 'Answer']].rename(columns={'ID': 'id', 'Answer': 'b'}), on='id')
bad = m[~m.a.isin(list(COL)) | ~m.b.isin(list(COL))]
print(f'rows {len(m)}; answered validly by both: {len(m) - len(bad)}' + (f'  (problem rows: {list(bad.id)[:10]})' if len(bad) else ''))
rep = m[m.repeat_of != '']; orig = m.set_index('id')
for w in 'ab':
    same = [COL.get(r[w]) == COL.get(orig.loc[r.repeat_of, w]) for _, r in rep.iterrows()]
    print(f'self-agreement annotator {w.upper()} on repeats: {sum(same)}/{len(same)}' + ('  << below 80%' if sum(same) < .8 * len(same) else ''))
m = m[(m.repeat_of == '') & m.a.isin(list(COL)) & m.b.isin(list(COL))].copy()
m['ca'], m['cb'] = m.a.map(COL), m.b.map(COL)
m['either'] = (m.a == 'E') | (m.b == 'E')
m['conflict'] = ((m.a == 'Y') & m.b.isin(['N', 'X'])) | ((m.b == 'Y') & m.a.isin(['N', 'X']))
m['contested'] = (m.ca != m.cb) | m.either
m['cell'] = np.where(m.group == 'C', 'C', m.group + '_' + m.junction.replace({'prefix_negation': 'prefix', 'between_words': 'between'}))
def wilson(k, n, z=1.96):
    p = k/n; c = (p + z*z/(2*n))/(1 + z*z/n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1 + z*z/n); return c - h, c + h
def newcombe(x, y):
    k1, n1, k2, n2 = x.sum(), len(x), y.sum(), len(y); l1, u1 = wilson(k1, n1); l2, u2 = wilson(k2, n2); p1, p2 = k1/n1, k2/n2
    return p1 - p2, p1 - p2 - math.sqrt((p1-l1)**2 + (u2-p2)**2), p1 - p2 + math.sqrt((u1-p1)**2 + (p2-l2)**2)
def kappa(x, y):
    po = (x == y).mean(); pe = sum((x == c).mean()*(y == c).mean() for c in ('split', 'no_split', 'open')); return (po - pe)/(1 - pe) if pe < 1 else float('nan')
def verdict(d, lo, thr): return 'SUPPORTED' if d >= thr and lo > 0 else 'NOT SUPPORTED' if d <= 0 else 'INCONCLUSIVE (pilot) - needs the larger run'
print('\nper cell:')
for c, d in m.groupby('cell'):
    print(f'  {c:11s} n={len(d):3d}  either {d.either.mean():5.1%}  conflict {d.conflict.mean():5.1%}  contested {d.contested.mean():5.1%}  '
          f'kappa {kappa(d.ca, d.cb):5.2f}  X {((d.a == "X") | (d.b == "X")).mean():5.1%}')
cell = lambda c: m[m.cell == c]
d, lo, hi = newcombe(cell('D_compound').either, cell('A_compound').either)
print(f'\nP1 principled   either(D_compound) - either(A_compound) = {d:+.1%}  [{lo:+.1%}, {hi:+.1%}]  -> {verdict(d, lo, .10)}')
d, lo, hi = newcombe(cell('D_prefix').conflict, cell('D_compound').conflict)
print(f'P2 unstable     conflict(D_prefix) - conflict(D_compound) = {d:+.1%}  [{lo:+.1%}, {hi:+.1%}]  -> {verdict(d, lo, .10)}')
S = ('compound', 'between'); rng = np.random.default_rng(0)
cells = {(g, s): cell(f'{g}_{s}').contested.values for g in 'DA' for s in S}
pt = np.mean([cells[('D', s)].mean() - cells[('A', s)].mean() for s in S])
bs = [np.mean([rng.choice(cells[('D', s)], len(cells[('D', s)])).mean() - rng.choice(cells[('A', s)], len(cells[('A', s)])).mean() for s in S]) for _ in range(5000)]
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f'P3 original H2c contested D - A (within compound & between, pooled) = {pt:+.1%}  [{lo:+.1%}, {hi:+.1%}]  -> {verdict(pt, lo, .15)}')
L = pd.concat([m[['cell', 'a']].rename(columns={'a': 'label'}), m[['cell', 'b']].rename(columns={'b': 'label'})], ignore_index=True)
print('\nlabel counts per cell (both annotators):'); print(pd.crosstab(L.cell, L.label).to_string())
