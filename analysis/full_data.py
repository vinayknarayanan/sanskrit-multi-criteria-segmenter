"""Full Frontiers corpus (dataset.csv, 90,893 rows): corpus-level statistics, incl. the piece-count distribution used as population weights.
  1. population distribution of piece counts -> weights for the stratified sample results
  2. boundary profile of the gold (plain vs sandhi junctions), by derivation
  3. gold derivability: harvested (full 1,100-rule table + insertions) vs grammar table
  4. holdout_sweep.py protocol on the full corpus: harvested top-n vs grammar, same held-out 30%
  5. SandhiKosh rows inside the full corpus
"""
import paths as P
import csv, difflib, random, re, sys, yaml, glob
from collections import Counter, defaultdict
import pandas as pd
from indic_transliteration import sanscript
from derive_engine import derive, build_grammar_index, INSERTIONS
from rescore import spans

slp = lambda s: sanscript.transliterate(s, 'iast', 'slp1')
D = pd.read_csv(P.DATA + 'frontiers_dataset.csv')
D = D.dropna(); D['pieces_iast'] = D.split.str.split('+')
D['n'] = D.pieces_iast.str.len()
D = D[D.n >= 2].reset_index(drop=True)
print('rows (>=2 pieces):', len(D))

# 1. population distribution
pop = D.n.value_counts().sort_index(); print('\n1. pieces per item (full corpus):'); print((pop / len(D) * 100).round(2).to_string())
pop.to_csv('full_piece_distribution.csv')

# indexes
H = defaultdict(list)
for r in yaml.safe_load(open(P.DATA + 'sandhi_rules_harvested.yaml'))['rules']:
    outs = r['output'] if isinstance(r['output'], list) else [r['output']]
    for o in outs:
        if o not in H[(r['left'], r['right'])]: H[(r['left'], r['right'])].append(o)
for L, R, O in INSERTIONS:
    if O not in H[(L, R)]: H[(L, R)].append(O)
G = build_grammar_index()

rows = []
for i, r in enumerate(D.itertuples()):
    p = [slp(x) for x in r.pieces_iast]; s = slp(r.compound)
    sp, ok = spans(p, s)
    rows.append(dict(n=r.n, H=bool(derive(p, s, H)), G=bool(derive(p, s, G)),
                     placed=sum(1 for x in sp if x), plain=sum(1 for x in sp if x and x[1] == 'plain'), aligned=ok))
    if i % 10000 == 0: print(f'  {i}/{len(D)}', file=sys.stderr)
R = pd.DataFrame(rows); D = pd.concat([D, R.drop(columns='n')], axis=1)
D.drop(columns='pieces_iast').to_csv('full_gold_scan.csv', index=False)

print(f"\n2. gold boundaries placed {D.placed.sum():,}; plain junctions {D.plain.sum()/D.placed.sum()*100:.1f}%")
print('   by piece count:'); print((D.groupby('n').apply(lambda g: g.plain.sum() / max(g.placed.sum(), 1)) * 100).round(1).to_string())
print(f"\n3. gold derivability: harvested (1,100 + insertions) {D.H.mean()*100:.1f}% | grammar {D.G.mean()*100:.1f}% | both fail {(~D.H & ~D.G).mean()*100:.1f}% ({(~D.H & ~D.G).sum():,} items)")
print('   by piece count (H / G):'); print((D.groupby('n')[['H', 'G']].mean() * 100).round(1).to_string())

# 4. holdout protocol (seed 1, 70/30) on full corpus
rs = [dict(surface=slp(c), pieces=[slp(x) for x in s.split('+')]) for c, s in zip(D.compound, D.split)]
random.Random(1).shuffle(rs); cut = int(len(rs) * 0.7); train, test = rs[:cut], rs[cut:]
cnt = Counter()
for row in train:
    pieces, surf = row['pieces'], row['surface']; concat = ''.join(pieces); juncs, acc = [], 0
    for q in pieces[:-1]: acc += len(q); juncs.append(acc)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, concat, surf, autojunk=False).get_opcodes():
        if tag == 'equal': continue
        near = [j for j in juncs if i1 - 1 <= j <= i2 + 1]
        if not near: continue
        j = near[0]; cnt[(concat[i1:min(i2, j)], concat[max(i1, j):i2], surf[j1:j2])] += 1
def idx_top(n, drop=()):
    ix = defaultdict(list)
    for (L, R, O), c in cnt.most_common(n):
        if (not L and not R) or (L, R, O) in drop: continue
        if O not in ix[(L, R)]: ix[(L, R)].append(O)
    for L, R, O in INSERTIONS:
        if O not in ix[(L, R)]: ix[(L, R)].append(O)
    return ix
cov = lambda ix: sum(1 for t in test if derive(t['pieces'], t['surface'], ix)) / len(test) * 100
ART = {('aH', 'i', 'E'), ('aH', 'u', 'O'), ('e', 'i', 'E'), ('', 'a', 's'), ('aH', 'a', 'os')}
print(f'\n4. holdout protocol, full corpus: train {len(train):,} / held-out {len(test):,}; harvested {len(cnt):,} triples')
for n in (100, 800):
    print(f'   harvested top-{n} + insertions: {cov(idx_top(n)):.2f}%')
print(f'   harvested top-800 + insertions, 5 artefact rules removed: {cov(idx_top(800, ART)):.2f}%')
print(f'   grammar table: {cov(G):.2f}%')

# 5. SandhiKosh inside the full corpus
iast = lambda s: sanscript.transliterate(str(s), 'devanagari', 'iast')
sk = []
for f in glob.glob(P.SANDHIKOSH + '*.xls'):
    for sheet, df in pd.read_excel(f, sheet_name=None).items():
        df.columns = [str(c).strip().lower() for c in df.columns]
        if 'word' not in df.columns: df.columns = [str(c).strip().lower() for c in df.iloc[0]]; df = df.iloc[1:]
        for w in df['word'].dropna(): sk.append((f.split('/')[-1].replace('.xls', ''), re.sub(r'\s+', '', iast(w)).strip()))
full_surf = set(D.compound.str.strip())
skd = pd.DataFrame(sk, columns=['sheet', 'surf']); skd['in_frontiers'] = skd.surf.isin(full_surf)
print(f'\n5. SandhiKosh rows found verbatim in the full corpus: {skd.in_frontiers.sum():,} of {len(skd):,}; '
      f'= {D.compound.isin(set(skd.surf)).mean()*100:.1f}% of Frontiers items')
print(skd.groupby('sheet').in_frontiers.agg(['sum', 'size']).to_string())
