"""Same-domain check: is the Frontiers-trained model's lead memorised vocabulary?
Group eval items by whether their gold pieces occur in the fine-tune training data (dataset.csv minus eval
overlap by surface and gold, as in finetune_byt5_frontiers.py), then compare the ByT5-Sanskrit vs ft3 gap per group."""
import paths as P
import numpy as np, pandas as pd
from collections import Counter
POP = {2: .709, 3: .212, 4: .057, 5: .017, 6: .005}
norm = lambda s: ' '.join(str(s).strip().split())
d = pd.read_csv(P.DATA + 'frontiers_dataset.csv').dropna()
ev = pd.read_csv(P.DATA + 'shared_eval_sample.csv')
es, eg = set(ev.compound.map(norm)), set(ev.gold_split.map(norm))
tr = d[~d.compound.map(norm).isin(es) & ~d.split.map(norm).isin(eg)]
PC = Counter(p.strip() for s in tr.split for p in s.split('+'))
PAIR = Counter(); [PAIR.update(zip(p, p[1:])) for p in (s.split('+') for s in tr.split)]
print('training rows', len(tr), 'piece types', len(PC))
def grp(g):
    p = g.split('+'); unseen = sum(PC[x] == 0 for x in p); pairs_seen = all(PAIR[(a, b)] > 0 for a, b in zip(p, p[1:]))
    if unseen: return '3 some piece unseen'
    return '1 all pieces + all junction pairs seen' if pairs_seen else '2 all pieces seen, some pair unseen'
ev['grp'] = ev.gold_split.map(grp)
S = pd.read_csv('sandhi_fp_scores.csv').merge(ev[['compound', 'grp']].drop_duplicates('compound'), on='compound')
W = S.pivot_table(index=['compound', 'stratum', 'grp'], columns='sys', values=['strict', 'plain', 'dcsall']).reset_index()
W.columns = ['_'.join(c).strip('_') if isinstance(c, tuple) else c for c in W.columns]
def wmean(df, col):
    g = df.groupby('stratum')[col].mean(); w = {k: v for k, v in POP.items() if k in g.index}; t = sum(w.values())
    return sum(g[k] * v for k, v in w.items()) / t
rng = np.random.default_rng(0)
for gname, g in W.groupby('grp'):
    print(f'\n== {gname}: n={len(g)}  strata {g.stratum.value_counts().sort_index().to_dict()}')
    for m in ('strict', 'dcsall', 'plain'):
        a, b = wmean(g, f'{m}_byt5'), wmean(g, f'{m}_ft3')
        boots = []
        for _ in range(500):
            s = g.sample(len(g), replace=True, random_state=int(rng.integers(1e9)))
            boots.append(wmean(s, f'{m}_ft3') - wmean(s, f'{m}_byt5'))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f'  {m:7s} byt5 {a:.3f}  ft3 {b:.3f}  gap {b-a:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]')
