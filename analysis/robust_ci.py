"""Robustness of the tag-flip result + bootstrap CIs.

Scorers per item:
  strict    derivation-based boundary F1 (rescore.spans), every unmatched candidate boundary counts
  noform    same, but candidate boundaries that cannot be placed on the surface ('unplaced', usually because the
            piece is written in an underlying/unsandhied form, e.g. tad for tat) are dropped instead of counted
  agnostic  form-agnostic boundary F1: cut offsets in the concatenated pieces are mapped onto the surface by
            character alignment (difflib), matched to gold cuts within +-1 character; no sandhi rules, no forms
Plus plain/dcsall from sandhi_fp_scores.csv (Frontiers eval only).
CIs: stratified bootstrap over items (resample within piece-count strata), population-weighted, 2,000 reps.
"""
import paths as P
import io, contextlib, difflib, sys, numpy as np, pandas as pd
from indic_transliteration import sanscript
with contextlib.redirect_stdout(io.StringIO()):
    from rescore import spans, overlaps

slp = lambda s: sanscript.transliterate(s, 'iast', 'slp1')
POP = {2: .709, 3: .212, 4: .057, 5: .017, 6: .005}

def f1(tp, nc, ng):
    p = tp / nc if nc else 1.0; r = tp / ng if ng else 1.0
    return 2 * p * r / (p + r) if p + r else 0.0

def derivation_scores(gold, cand, surf):
    gs, _ = spans([slp(x) for x in gold], slp(surf)); cs, _ = spans([slp(x) for x in cand], slp(surf))
    used, tp, unplaced_fp = set(), 0, 0
    for c in cs:
        hit = False
        if c is not None:
            for j, g in enumerate(gs):
                if j not in used and g is not None and overlaps(c[0], g[0]):
                    used.add(j); hit = True; break
        if hit: tp += 1
        elif c is None: unplaced_fp += 1
    nc, ng = len(cs), len(gs)
    return f1(tp, nc, ng), f1(tp, nc - unplaced_fp, ng), unplaced_fp

def cuts_on_surface(pieces, surf):
    c = ''.join(pieces); offs, acc = [], 0
    for p in pieces[:-1]: acc += len(p); offs.append(acc)
    ops = difflib.SequenceMatcher(None, c, surf, autojunk=False).get_opcodes()
    out = []
    for i in offs:
        for tag, i1, i2, j1, j2 in ops:
            if i1 <= i <= i2 and (i < i2 or tag == 'equal' or i2 == len(c)):
                out.append(j1 + (i - i1) if tag == 'equal' else j1 + round((i - i1) * (j2 - j1) / max(i2 - i1, 1)))
                break
    return out

def agnostic(gold, cand, surf):
    g, c = cuts_on_surface(gold, surf), cuts_on_surface(cand, surf)
    used, tp = set(), 0
    for x in c:
        for k, y in enumerate(g):
            if k not in used and abs(x - y) <= 1: used.add(k); tp += 1; break
    return f1(tp, len(c), len(g))

def score_file(ev, preds):
    rows = []
    for x in ev.itertuples():
        p = preds.get(x.compound)
        if not isinstance(p, str) or not p.strip(): continue
        gold, cand = x.gold_split.split('+'), p.split()
        s, nf, up = derivation_scores(gold, cand, x.compound)
        rows.append(dict(compound=x.compound, stratum=x.split_count, strict=s, noform=nf, agnostic=agnostic(gold, cand, x.compound), unplaced=up))
    return pd.DataFrame(rows)

def wmean(df, col):
    g = df.groupby('stratum')[col].mean(); w = {k: v for k, v in POP.items() if k in g.index}
    return sum(g[k] * v for k, v in w.items()) / sum(w.values())

def boot_gap(a, b, col, reps=2000, seed=0):
    """paired stratified bootstrap of weighted mean(a) - weighted mean(b) over shared items"""
    m = a[['compound', 'stratum', col]].merge(b[['compound', col]], on='compound', suffixes=('_a', '_b'))
    rng = np.random.default_rng(seed); groups = {k: g for k, g in m.groupby('stratum')}
    point = wmean(m.rename(columns={f'{col}_a': 'x'}), 'x') - wmean(m.rename(columns={f'{col}_b': 'x'}), 'x')
    vals = []
    for _ in range(reps):
        tot, wsum = 0.0, 0.0
        for k, g in groups.items():
            idx = rng.integers(0, len(g), len(g)); w = POP.get(k, 0)
            tot += w * (g[f'{col}_a'].values[idx].mean() - g[f'{col}_b'].values[idx].mean()); wsum += w
        vals.append(tot / wsum)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, lo, hi, len(m)

if __name__ == '__main__':
    U = P.RUNS + ''
    ev = pd.read_csv(P.DATA + 'shared_eval_sample.csv').drop_duplicates('compound')
    fr_sys = {'byt5': pd.read_csv('results_byt5.csv'),
              'ft3': pd.read_csv(U + 'byt5_frontiers_v2/results_byt5_frontiers.csv'),
              'tagfr': pd.read_csv(U + 'byt5_tagged/results_fr.csv'),
              'tagdcs': pd.read_csv(U + 'byt5_tagged/results_dcs.csv')}
    S = {k: score_file(ev, dict(zip(v.compound, v.predicted))) for k, v in fr_sys.items()}
    keep = set.intersection(*[set(d.compound) for d in S.values()])
    old = pd.read_csv('sandhi_fp_scores.csv')
    for k in S:
        S[k] = S[k][S[k].compound.isin(keep)].merge(old[old.sys == k][['compound', 'plain', 'dcsall']], on='compound', how='left')
    print(f'Frontiers eval, items scored by all: {len(keep)}')
    print(f"{'system':7s} {'strict':>7s} {'noform':>7s} {'agnost':>7s} {'DCS-all':>7s} {'lenient':>7s}  unplaced FP")
    for k, d in S.items():
        print(f"{k:7s} " + ' '.join(f'{wmean(d, c):7.3f}' for c in ('strict', 'noform', 'agnostic', 'dcsall', 'plain')) + f"  {int(d.unplaced.sum())}")
    print('\nGaps with 95% CI (paired stratified bootstrap, population-weighted):')
    for a, b, lab in (('tagfr', 'tagdcs', 'TAG FLIP on Frontiers (<fr> - <dcs>)'), ('ft3', 'byt5', 'Frontiers-trained - ByT5-Sanskrit'),
                      ('tagdcs', 'byt5', '<dcs> - ByT5-Sanskrit'), ('tagfr', 'ft3', '<fr> - Frontiers-only')):
        print(' ', lab)
        for col in ('strict', 'noform', 'agnostic', 'dcsall', 'plain'):
            p, lo, hi, n = boot_gap(S[a], S[b], col)
            print(f'    {col:8s} {p:+.3f}  [{lo:+.3f}, {hi:+.3f}]')
    # mirror on DCS holdout
    hv = pd.read_csv(P.DATA + 'dcs_holdout_eval.csv')
    D = {t: score_file(hv, dict(zip(r.compound, r.predicted))) for t, r in
         (('fr', pd.read_csv(U + 'tagged_on_dcs/results_fr.csv')), ('dcs', pd.read_csv(U + 'tagged_on_dcs/results_dcs.csv')))}
    print(f'\nDCS holdout ({len(hv)} items; weighted with Frontiers population weights):')
    for t, d in D.items():
        print(f"  <{t}> strict {wmean(d, 'strict'):.3f}  noform {wmean(d, 'noform'):.3f}  agnostic {wmean(d, 'agnostic'):.3f}  unplaced FP {int(d.unplaced.sum())}")
    print('  MIRROR gap (<dcs> - <fr>) on DCS gold:')
    for col in ('strict', 'noform', 'agnostic'):
        p, lo, hi, n = boot_gap(D['dcs'], D['fr'], col)
        print(f'    {col:8s} {p:+.3f}  [{lo:+.3f}, {hi:+.3f}]')
    ex = {}
    for name, d in (('fr_on_fr', S['tagfr']), ('dcs_on_fr', S['tagdcs'])): pass
    pd.concat([d.assign(sys=k) for k, d in S.items()]).to_csv('robust_scores_frontiers.csv', index=False)
    pd.concat([d.assign(sys=k) for k, d in D.items()]).to_csv('robust_scores_dcsholdout.csv', index=False)
