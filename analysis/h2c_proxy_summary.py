"""Summary of h2c_proxy_junctions.csv with a fast word-cluster bootstrap.
Construction check: on training words the <dcs> setting mostly reproduces the DCS MAJORITY analysis, so group
membership is partly fixed by whether DCS's majority cuts. The fair comparison holds that constant:
among junctions the DCS majority cuts (p > .5): A (Frontiers setting also cuts) vs D (Frontiers setting does not)."""
import paths as P
import numpy as np, pandas as pd
R = pd.read_csv('h2c_proxy_junctions.csv')
R = R[R.src == 'dcs_frequent']
def boot(df, m1, m2, reps=2000):
    w = pd.factorize(df.word)[0]; nW = w.max() + 1; rng = np.random.default_rng(0)
    def agg(mask):
        c = np.bincount(w[mask], weights=df.contested.values[mask], minlength=nW); n = np.bincount(w[mask], minlength=nW); return c, n
    c1, n1 = agg(m1.values); c2, n2 = agg(m2.values); pt = c1.sum()/n1.sum() - c2.sum()/n2.sum(); v = []
    for _ in range(reps):
        k = np.bincount(rng.integers(0, nW, nW), minlength=nW)
        v.append((k@c1)/(k@n1) - (k@c2)/(k@n2))
    return pt, *np.percentile(v, [2.5, 97.5])
print('frequent DCS words:', R.word.nunique(), 'junctions', len(R))
maj = R[R.p > .5]; mino = R[R.p <= .5]
print('\nA) junctions the DCS majority CUTS: does the Frontiers setting agree?')
for jt in ('all', 'compound', 'between_words'):
    s = maj if jt == 'all' else maj[maj.junction == jt]
    a, d = s.cut_by == 'both', s.cut_by == 'dcs_only'
    pt, lo, hi = boot(s, d, a)
    print(f'  [{jt:13s}] contested: fr also cuts {s[a].contested.mean():.3f} (n={a.sum()})  fr does not {s[d].contested.mean():.3f} (n={d.sum()})  diff {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]')
print('\nB) junctions the DCS majority does NOT cut: does the Frontiers setting cut anyway?')
for jt in ('all', 'compound', 'between_words'):
    s = mino if jt == 'all' else mino[mino.junction == jt]
    f, nn = s.cut_by == 'fr_only', s.cut_by == 'neither'
    if f.sum() >= 10 and nn.sum() >= 10:
        pt, lo, hi = boot(s, f, nn); print(f'  [{jt:13s}] contested: fr cuts {s[f].contested.mean():.3f} (n={f.sum()})  neither {s[nn].contested.mean():.3f} (n={nn.sum()})  diff {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]')
    else: print(f'  [{jt:13s}] too few: fr_only {f.sum()}, neither {nn.sum()}')
print('\nC) how often does the <dcs> setting follow the DCS majority? cut where p>.5:',
      round(((R.cut_by.isin(['both', 'dcs_only'])) == (R.p > .5)).mean(), 3))
print('\nexamples, D (dcs majority cuts, fr does not) that ARE contested:')
print(maj[(maj.cut_by == 'dcs_only') & maj.contested].sample(8, random_state=1)[['word', 'cut', 'p', 'junction']].to_string(index=False))
