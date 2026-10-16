"""Pairwise convention distance between segmentation datasets, measured on word types both datasets contain.
For each shared written form: each dataset's most frequent division. Reported per pair:
  n        shared types where at least one side splits (>=2 pieces)
  exact    identical pieces
  samecut  identical cut positions on the written form (form-agnostic, +-1 char) -- forms may differ
  F1       form-agnostic boundary F1 (symmetric)
  A>B      share of types where A has strictly more cuts than B (A splits deeper), and B>A
Baseline: each dataset against itself (random halves of its rows), i.e. its own consistency."""
import paths as P
import io, contextlib, pickle, random, unicodedata, re, sys, pandas as pd
from collections import Counter, defaultdict
with contextlib.redirect_stdout(io.StringIO()):
    from robust_ci import cuts_on_surface
b = lambda s: unicodedata.normalize('NFC', str(s).strip().lower()).replace('ṃ', 'm')
key = lambda s: re.sub(r'(?:[ḥmsrtdn]|o|e)$', '', b(s)) if len(b(s)) > 3 else b(s)

def table(pairs):            # iterable of (surface, pieces tuple) -> {surface: most frequent pieces}
    c = defaultdict(Counter)
    for s, p in pairs: c[b(s)][tuple(b(x) for x in p)] += 1
    return {s: v.most_common(1)[0][0] for s, v in c.items()}

R = pd.read_csv(P.CACHE + 'units_sighum_hackathon.csv')
F = pd.read_csv(P.DATA + 'frontiers_dataset.csv').dropna()
src = {'sighum': [(s, p.split('+')) for s, p in zip(R[R.dataset == 'sighum'].surface, R[R.dataset == 'sighum'].split_pieces)],
       'hackathon': [(s, p.split('+')) for s, p in zip(R[R.dataset == 'hackathon'].surface, R[R.dataset == 'hackathon'].split_pieces)],
       'frontiers': [(s, [x.strip() for x in p.split('+')]) for s, p in zip(F.compound, F.split)]}
T = {k: table(v) for k, v in src.items()}
D = pickle.load(open(P.CACHE + 'dcs_surface.pkl', 'rb'))['idx']
def dcs(s):
    c = D.get(key(s))
    return tuple(b(m[0]) for m in c.most_common(1)[0][0]) if c else None

def compare(A, B, surfaces, la, lb):
    n = ex = same = 0; tp = na = nb = 0; agt = bgt = 0
    for s in surfaces:
        a, bb = A(s), B(s)
        if a is None or bb is None or (len(a) < 2 and len(bb) < 2): continue
        n += 1; ex += a == bb
        ca, cb = cuts_on_surface(list(a), s), cuts_on_surface(list(bb), s)
        used = set(); t = 0
        for x in ca:
            for k, y in enumerate(cb):
                if k not in used and abs(x - y) <= 1: used.add(k); t += 1; break
        tp += t; na += len(ca); nb += len(cb)
        same += t == len(ca) == len(cb)
        agt += len(ca) > len(cb); bgt += len(cb) > len(ca)
    f1 = 2 * tp / (na + nb) if na + nb else 0
    return dict(pair=f'{la} vs {lb}', n=n, exact=ex / n, samecut=same / n, F1=f1, A_deeper=agt / n, B_deeper=bgt / n)

out = []
names = ['frontiers', 'sighum', 'hackathon']
for i, x in enumerate(names):
    for y in names[i + 1:]:
        shared = set(T[x]) & set(T[y])
        out.append(compare(T[x].get, T[y].get, shared, x, y))
    out.append(compare(T[x].get, dcs, set(T[x]), x, 'dcs'))
# self-consistency baselines: random halves of each dataset's rows
random.seed(0)
for x in names:
    rows = src[x][:]; random.shuffle(rows); h = len(rows) // 2
    A, B = table(rows[:h]), table(rows[h:])
    out.append(compare(A.get, B.get, set(A) & set(B), x + '[half1]', x + '[half2]'))
O = pd.DataFrame(out); O.to_csv('convention_distance.csv', index=False)
pd.set_option('display.width', 200)
print(O.round(3).to_string(index=False))
