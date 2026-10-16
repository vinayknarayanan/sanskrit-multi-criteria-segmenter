"""Pooled strict boundary F1 (derivation-based), population-weighted, with paired stratified bootstrap CIs."""
import io, contextlib, numpy as np, pandas as pd
from indic_transliteration import sanscript
with contextlib.redirect_stdout(io.StringIO()):
    from rescore import spans, overlaps
    from robust_ci import POP, agnostic, cuts_on_surface
slp = lambda s: sanscript.transliterate(s, 'iast', 'slp1')
import paths as P
U = P.RUNS
def strict_counts(gold, cand, surf):
    gs, _ = spans([slp(x) for x in gold], slp(surf)); cs, _ = spans([slp(x) for x in cand], slp(surf))
    used, tp = set(), 0
    for c in cs:
        if c is None: continue
        for j, g in enumerate(gs):
            if j not in used and g is not None and overlaps(c[0], g[0]): used.add(j); tp += 1; break
    return tp, len(cs), len(gs)
def agn_counts(gold, cand, surf):
    g, c = cuts_on_surface(gold, surf), cuts_on_surface(cand, surf); used, tp = set(), 0
    for x in c:
        for k, y in enumerate(g):
            if k not in used and abs(x - y) <= 1: used.add(k); tp += 1; break
    return tp, len(c), len(g)
def f(tp, nc, ng):
    p = tp / nc if nc else 1; r = tp / ng if ng else 1; return 2 * p * r / (p + r) if p + r else 0
def table(ev, files):
    R = []
    for k, fn in files.items():
        pr = dict(zip(*pd.read_csv(fn)[['compound', 'predicted']].values.T))
        for x in ev.itertuples():
            p = pr.get(x.compound)
            if not isinstance(p, str) or not p.strip(): continue
            g, c = x.gold_split.split('+'), p.split()
            R.append((k, x.compound, x.split_count, *strict_counts(g, c, x.compound), *agn_counts(g, c, x.compound)))
    d = pd.DataFrame(R, columns=['sys', 'compound', 'stratum', 'stp', 'snc', 'sng', 'atp', 'anc', 'ang'])
    keep = set.intersection(*[set(g.compound) for _, g in d.groupby('sys')]); return d[d.compound.isin(keep)]
def gap(d, a, b, pre, reps=2000):
    A = d[d.sys == a].set_index('compound').sort_index(); B = d[d.sys == b].set_index('compound').loc[A.index]
    st = A.stratum.values; idx = {s: np.where(st == s)[0] for s in POP if (st == s).any()}
    def val(ii):
        out = []
        for X in (A, B):
            tp = nc = ng = 0
            for s, ix in ii.items():
                w = POP[s] / len(ix); tp += w * X[pre + 'tp'].values[ix].sum(); nc += w * X[pre + 'nc'].values[ix].sum(); ng += w * X[pre + 'ng'].values[ix].sum()
            out.append(f(tp, nc, ng))
        return out
    pa, pb = val(idx); rng = np.random.default_rng(0)
    v = [np.subtract(*val({s: ix[rng.integers(0, len(ix), len(ix))] for s, ix in idx.items()})) for _ in range(reps)]
    lo, hi = np.percentile(v, [2.5, 97.5]); return pa, pb, pa - pb, lo, hi, len(A)
ev = pd.read_csv(P.DATA + 'shared_eval_sample.csv').drop_duplicates('compound')
F = table(ev, {'fr1': U + 'byt5_tagged/results_fr.csv', 'dcs1': U + 'byt5_tagged/results_dcs.csv', 'fr2': U + 'byt5_tagged_s2/results_fr.csv',
               'dcs2': U + 'byt5_tagged_s2/results_dcs.csv', 'ft3': U + 'byt5_frontiers_v2/results_byt5_frontiers.csv', 'byt5': 'results_byt5.csv'})
hv = pd.read_csv(P.DATA + 'dcs_holdout_eval.csv')
D = table(hv, {'fr1': U + 'tagged_on_dcs/results_fr.csv', 'dcs1': U + 'tagged_on_dcs/results_dcs.csv',
               'fr2': U + 'tagged_s2_on_dcs/results_fr.csv', 'dcs2': U + 'tagged_s2_on_dcs/results_dcs.csv'})
print('Frontiers items', F.compound.nunique(), '| DCS items', D.compound.nunique())
for pre, lab in (('s', 'STRICT'), ('a', 'FORM-AGNOSTIC')):
    print(f'== pooled {lab}, population-weighted')
    for d, a, b, name in ((F, 'fr1', 'dcs1', 'Frontiers flip s1'), (F, 'fr2', 'dcs2', 'Frontiers flip s2'), (D, 'dcs1', 'fr1', 'DCS mirror s1'),
                          (D, 'dcs2', 'fr2', 'DCS mirror s2'), (F, 'fr1', 'ft3', '<fr> s1 - Fr-only'), (F, 'fr2', 'ft3', '<fr> s2 - Fr-only'),
                          (F, 'byt5', 'dcs1', 'ByT5-Skt - <dcs> s1'), (F, 'byt5', 'dcs2', 'ByT5-Skt - <dcs> s2'), (F, 'ft3', 'byt5', 'Fr-only - ByT5-Skt')):
        pa, pb, g, lo, hi, n = gap(d, a, b, pre)
        print(f'  {name:22s} {pa:.3f} vs {pb:.3f}  gap {g:+.3f} [{lo:+.3f}, {hi:+.3f}]  n={n}')
