"""Five-seed analysis, as pre-specified before the extra runs (16 Oct 2026): tag flip per seed, the +-0.01 equivalence
test (TOST, 90% interval, bootstrap over seeds and items), the 2-epoch control and the 293-word reconstructed-key diagnostic.
Run from results/:   python ../analysis/five_seed.py      -> writes seeds_summary.json
Run folders are predictions/<run>/ (see README). Uses robust_ci.py from this folder."""
import io, contextlib, json, os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
with contextlib.redirect_stdout(io.StringIO()):
    from robust_ci import score_file, wmean, boot_gap, POP, agnostic
O = N = P.RUNS
ev = pd.read_csv(P.DATA + 'shared_eval_sample.csv').drop_duplicates('compound')
hv = pd.read_csv(P.DATA + 'dcs_holdout_eval.csv')
KEEP = set(pd.read_csv('robust_scores_frontiers.csv').compound)          # the 3,211 common items
TAG = {13: (O + 'byt5_tagged/', O + 'tagged_on_dcs/'), 42: (O + 'byt5_tagged_s2/', O + 'tagged_s2_on_dcs/')}
for s in (7, 21, 99): TAG[s] = (N + f'tagged_s{s}/', N + f'tagged_s{s}_on_dcs/')
FON = {13: O + 'byt5_frontiers_v2/results_byt5_frontiers.csv'}
for s in (7, 21, 42, 99): FON[s] = N + f'frontiers_s{s}/results_byt5_frontiers.csv'
F2EP = N + 'frontiers_2ep_s13/results_byt5_frontiers.csv'
ok = os.path.exists
def preds(f): r = pd.read_csv(f); return dict(zip(r.compound, r.predicted)), r
def sc(e, f, keep=None):
    p, _ = preds(f); d = score_file(e, p); return d[d.compound.isin(keep)] if keep is not None else d
def exact(f, e):
    _, r = preds(f); r = r[r.compound.isin(set(e.compound))]
    return float(r.acc_exact.mean())
out = {'tagged': {}, 'frontiers_only': {}}
print('== per seed: tag flip')
for s, (fd, dd) in sorted(TAG.items()):
    if not (ok(fd + 'results_fr.csv') and ok(fd + 'results_dcs.csv')): continue
    fr, dc = sc(ev, fd + 'results_fr.csv', KEEP), sc(ev, fd + 'results_dcs.csv', KEEP)
    row = dict(fr_strict=wmean(fr, 'strict'), dcs_strict=wmean(dc, 'strict'), fr_agn=wmean(fr, 'agnostic'), dcs_agn=wmean(dc, 'agnostic'),
               flip_strict=boot_gap(fr, dc, 'strict')[:3], flip_agn=boot_gap(fr, dc, 'agnostic')[:3],
               exact_fr=exact(fd + 'results_fr.csv', ev), exact_dcs=exact(fd + 'results_dcs.csv', ev))
    if ok(dd + 'results_dcs.csv'):
        a, b = sc(hv, dd + 'results_dcs.csv'), sc(hv, dd + 'results_fr.csv')
        row.update(m_dcs_strict=wmean(a, 'strict'), m_fr_strict=wmean(b, 'strict'), mirror_strict=boot_gap(a, b, 'strict')[:3],
                   mirror_agn=boot_gap(a, b, 'agnostic')[:3], m_exact_dcs=exact(dd + 'results_dcs.csv', hv), m_exact_fr=exact(dd + 'results_fr.csv', hv))
    out['tagged'][s] = row
    print(s, ' '.join(f'{k}={v if isinstance(v, tuple) else round(v, 3)}' for k, v in row.items() if 'flip' in k or 'mirror' in k or k == 'fr_strict'))
print('== per seed: Frontiers-only')
for s, f in sorted(FON.items()):
    if ok(f): d = sc(ev, f, KEEP); out['frontiers_only'][s] = dict(strict=wmean(d, 'strict'), agn=wmean(d, 'agnostic')); print(s, out['frontiers_only'][s])
def agg(vals):
    v = np.array(vals, float); return dict(n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else None, min=float(v.min()), max=float(v.max()))
T = out['tagged']; A = {}
for k in ('flip_strict', 'flip_agn', 'mirror_strict', 'mirror_agn'):
    A[k] = agg([r[k][0] for r in T.values() if k in r])
for k in ('fr_strict', 'dcs_strict', 'exact_fr', 'exact_dcs', 'm_dcs_strict', 'm_fr_strict', 'm_exact_dcs', 'm_exact_fr'):
    A[k] = agg([r[k] for r in T.values() if k in r])
A['fonly_strict'] = agg([r['strict'] for r in out['frontiers_only'].values()])
print('== aggregates'); [print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()}) for k, v in A.items()]
# ---- pre-specified equivalence test (TOST, margin 0.01, 90% interval; bootstrap over seeds and items)
tg = {s: sc(ev, TAG[s][0] + 'results_fr.csv', KEEP).set_index('compound') for s in T}
fo = {s: sc(ev, f, KEEP).set_index('compound') for s, f in FON.items() if ok(f)}
items = sorted(KEEP); strat = ev.set_index('compound').loc[items].split_count.values
X = np.array([tg[s].loc[items, 'strict'].values for s in sorted(tg)]); Y = np.array([fo[s].loc[items, 'strict'].values for s in sorted(fo)])
idx = {k: np.where(strat == k)[0] for k in POP}
def wm(M, ii): return sum(POP[k] * M[:, ii[k]].mean(axis=1) for k in POP) / sum(POP.values())
pt = wm(X, idx).mean() - wm(Y, idx).mean(); rng = np.random.default_rng(0); B = []
for _ in range(4000):
    ii = {k: v[rng.integers(0, len(v), len(v))] for k, v in idx.items()}
    xs, ys = rng.integers(0, len(X), len(X)), rng.integers(0, len(Y), len(Y))
    B.append(wm(X[xs], ii).mean() - wm(Y[ys], ii).mean())
lo, hi = np.percentile(B, [5, 95])
A['nocost'] = dict(diff=float(pt), ci90=[float(lo), float(hi)], margin=0.01, equivalent=bool(lo > -0.01 and hi < 0.01), n_tagged=len(X), n_fonly=len(Y))
print('== no-cost TOST:', A['nocost'])
if ok(F2EP):
    d2 = sc(ev, F2EP, KEEP); A['fonly_2ep_s13'] = dict(strict=wmean(d2, 'strict'), agn=wmean(d2, 'agnostic'),
        vs_3ep_s13=boot_gap(d2, sc(ev, FON[13], KEEP), 'strict')[:3], tagged_fr_s13_minus_2ep=boot_gap(sc(ev, TAG[13][0] + 'results_fr.csv', KEEP), d2, 'strict')[:3])
    print('== 2-epoch control:', A['fonly_2ep_s13'])
# ---- 293-word diagnostic, all seeds (reconstructed DCS-style key, unweighted)
f_ = pd.read_csv('corpus_frontiers_vs_dcs_forward.csv'); bad_ = set(f_[f_.status == 'not in DCS'].compound); conv = {}
for c, g in f_.groupby('compound', sort=False):
    if c in bad_: continue
    pcs = []
    for r in g.itertuples(): pcs += [x.strip() for x in r.dcs.split('+')] if r.status == 'DCS divides, Frontiers does not' else [r.piece]
    conv[c] = pcs
diff_items = [c for c, dg in conv.items() if len(dg) != len(ev.set_index('compound').loc[c, 'gold_split'].split('+'))]
def diag(f):
    p, _ = preds(f); vf, vd = [], []
    for c in diff_items:
        q = p.get(c)
        if not isinstance(q, str) or not q.strip(): continue
        fg = ev.set_index('compound').loc[c, 'gold_split'].split('+'); vf.append(agnostic(fg, q.split(), c)); vd.append(agnostic(conv[c], q.split(), c))
    return float(np.mean(vf)), float(np.mean(vd))
D = {'tag_fr': [], 'tag_dcs': [], 'fonly': []}
for s in T: D['tag_fr'].append(diag(TAG[s][0] + 'results_fr.csv')); D['tag_dcs'].append(diag(TAG[s][0] + 'results_dcs.csv'))
for s, f in FON.items():
    if ok(f): D['fonly'].append(diag(f))
byt5 = diag('results_byt5.csv'); rcnn = diag('results_rcnn.csv')
A['diag293'] = {k: dict(n=len(v), fr_key=agg([x[0] for x in v]), dcs_key=agg([x[1] for x in v])) for k, v in D.items()}
A['diag293']['byt5'] = byt5; A['diag293']['rcnn'] = rcnn; A['diag293']['n_items'] = len(diff_items)
print('== 293 diagnostic (unweighted means over seeds):')
for k, v in A['diag293'].items():
    print(' ', k, v if not isinstance(v, dict) else {kk: (round(vv['mean'], 3), round(vv['sd'] or 0, 3), vv['n']) if isinstance(vv, dict) else vv for kk, vv in v.items()})
json.dump(dict(per_seed=out, agg=A), open('seeds_summary.json', 'w'), indent=1, default=str)
