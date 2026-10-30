"""Ranking under two authentic keys on the two-key words: ByT5-Sanskrit vs the held-out tagged model.
Form-agnostic F1, unweighted means, paired bootstrap (2,000) over words. Run after run_byt5skt.py."""
import os, random, pandas as pd, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import difflib, unicodedata
nf = lambda s: unicodedata.normalize('NFC', str(s).strip())
def cuts(pieces, surf):
    c = ''.join(pieces); offs, acc = [], 0
    for p in pieces[:-1]: acc += len(p); offs.append(acc)
    ops = difflib.SequenceMatcher(None, c, surf, autojunk=False).get_opcodes(); out = []
    for i in offs:
        for t, i1, i2, j1, j2 in ops:
            if i1 <= i <= i2 and (i < i2 or t == 'equal' or i2 == len(c)):
                out.append(j1 + (i - i1) if t == 'equal' else j1 + round((i - i1) * (j2 - j1) / max(i2 - i1, 1))); break
    return out
def match(a, b):
    used, tp = set(), 0
    for x in a:
        for k, y in enumerate(b):
            if k not in used and abs(x - y) <= 1: used.add(k); tp += 1; break
    return tp
def f1(c, g):
    tp = match(c, g); p = tp / len(c) if c else 1.0; r = tp / len(g) if g else 1.0
    return 2 * p * r / (p + r) if p + r else 0.0
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, '..', 'predictions')
ev = pd.read_csv(os.path.join(H, 'twokey_eval.csv'))
sys_ = {'byt5skt': pd.read_csv(os.path.join(H, 'results_byt5skt_twokey.csv'))}
for s in ('13', '42'):
    for t in ('fr', 'dcs'): sys_[f'{t}_s{s}'] = pd.read_csv(os.path.join(R, f'twokey_s{s}', f'results_{t}.csv'))
rows = []
for x in ev.itertuples():
    w = nf(x.compound); F = cuts([nf(p) for p in x.gold_split.split('+')], w); D = cuts([nf(p) for p in x.dcs_split.split('+')], w)
    r = dict(compound=x.compound, group=x.group)
    for k, df in sys_.items():
        p = df.set_index('compound').predicted.get(x.compound)
        c = cuts(str(p).split(), w) if isinstance(p, str) and p.strip() else []
        r[k + '_F'] = f1(c, F); r[k + '_D'] = f1(c, D)
    rows.append(r)
S = pd.DataFrame(rows)
def boot(d, a, b, n=2000):
    v = (d[a] - d[b]).tolist(); k = len(v); g = random.Random(0)
    bs = sorted(sum(v[g.randrange(k)] for _ in range(k)) / k for _ in range(n)); return sum(v) / k, bs[int(.025 * n)], bs[int(.975 * n)]
for g, d in S.groupby('group'):
    print(f'\n== {g} (n={len(d)})')
    for k in sys_: print(f'  {k:10s} Frontiers key {d[k+"_F"].mean():.3f}   DCS key {d[k+"_D"].mean():.3f}')
    for s in ('13', '42'):
        a = boot(d, f'fr_s{s}_F', 'byt5skt_F'); b = boot(d, 'byt5skt_D', f'fr_s{s}_D')
        print(f'  seed {s}: <fr> - ByT5-Sanskrit on Frontiers key {a[0]:+.3f} [{a[1]:+.3f}, {a[2]:+.3f}];  ByT5-Sanskrit - <fr> on DCS key {b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}]')
S.to_csv(os.path.join(H, 'ranking_item_scores.csv'), index=False)
