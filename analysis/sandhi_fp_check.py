"""Point 1 of the fine-tune weak points: does lenient scoring undercount convention?

For every UNMATCHED candidate boundary (false positive) of each system, record:
  kind      plain / sandhi junction (from the candidate's derivation)
  dcs       DCS evidence level for the pair (A_pair, B_lemma, S_policy = "DCS splits here")
  fr_split  how often the Frontiers gold (full corpus, 90,893 rows) splits this exact piece pair elsewhere
Then re-score with one more variant, applied identically to every system:
  dcsall    any false positive (plain OR sandhi) excused if DCS splits there (A/B/S)
and report population-weighted F1 for strict / plaindcs / plain / dcsall.
"""
import paths as P
import io, contextlib, sys, pandas as pd
from collections import Counter
from indic_transliteration import sanscript
from rescore import spans, overlaps
with contextlib.redirect_stdout(io.StringIO()):
    from dcs_match import level as dcs_level
from kt4_suffix_cuts import label as cut_label

slp = lambda s: sanscript.transliterate(s, 'iast', 'slp1')
POP = {2: .709, 3: .212, 4: .057, 5: .017, 6: .005}
ATTEST = ('A_pair', 'B_lemma', 'S_policy')

full = pd.read_csv(P.DATA + 'frontiers_dataset.csv').dropna()
PAIR = Counter()
for s in full.split:
    p = [x.strip() for x in s.split('+')]
    PAIR.update(zip(p, p[1:]))

def fps(gold, cand, surf):
    gp, cp = [slp(p) for p in gold], [slp(p) for p in cand]
    gs, _ = spans(gp, slp(surf)); cs, _ = spans(cp, slp(surf))
    used, out, tp = set(), [], 0
    for k, c in enumerate(cs):
        hit = False
        if c is not None:
            for j, g in enumerate(gs):
                if j not in used and g is not None and overlaps(c[0], g[0]):
                    used.add(j); hit = True; break
        if hit: tp += 1; continue
        l, r = cand[k], cand[k + 1]
        cls = cut_label(r); cls = 'suffix' if cls.startswith('-') else ('other' if cls in ('other', 'ādi') else cls)
        out.append(dict(left=l, right=r, kind=(c[1] if c else 'unplaced'), dcs=dcs_level(l, r, cls),
                        fr_split=PAIR.get((l, r), 0)))
    return tp, len(cs), len(gs), out

ev = pd.read_csv(P.DATA + 'shared_eval_sample.csv')[['compound', 'gold_split', 'split_count']]
systems = {'byt5': pd.read_csv('results_byt5.csv'), 'rcnn': pd.read_csv('results_rcnn.csv'),
           'ft1': pd.read_csv(P.RUNS + 'byt5_ep1/results_byt5_frontiers.csv'),
           'ft3': pd.read_csv(P.RUNS + 'byt5_frontiers_v2/results_byt5_frontiers.csv'),
           'tagfr': pd.read_csv(P.RUNS + 'byt5_tagged/results_fr.csv'),
           'tagdcs': pd.read_csv(P.RUNS + 'byt5_tagged/results_dcs.csv')}
preds = {k: dict(zip(v.compound, v.predicted)) for k, v in systems.items()}
keep = [c for c in ev.compound if all(isinstance(preds[k].get(c), str) and preds[k][c].strip() for k in preds)]
ev = ev[ev.compound.isin(keep)].drop_duplicates('compound')
print('items with predictions from all systems:', len(ev), file=sys.stderr)

rows, fprows = [], []
for i, x in enumerate(ev.itertuples()):
    gold = x.gold_split.split('+')
    for s in systems:
        tp, nc, ng, fp = fps(gold, preds[s][x.compound].split(), x.compound)
        n_plain = sum(f['kind'] == 'plain' for f in fp)
        n_plaindcs = sum(f['kind'] == 'plain' and f['dcs'] in ATTEST for f in fp)
        n_dcsall = sum(f['dcs'] in ATTEST for f in fp)
        def f1(nc_):
            p = tp / nc_ if nc_ else 1.0; r = tp / ng if ng else 1.0
            return 2 * p * r / (p + r) if p + r else 0.0
        rows.append(dict(compound=x.compound, stratum=x.split_count, sys=s, fp=len(fp),
                         strict=f1(nc), plaindcs=f1(nc - n_plaindcs), plain=f1(nc - n_plain), dcsall=f1(nc - n_dcsall)))
        for f in fp: fprows.append(dict(compound=x.compound, sys=s, **f))
    if i % 500 == 0: print(i, file=sys.stderr)

R, F = pd.DataFrame(rows), pd.DataFrame(fprows)
R.to_csv('sandhi_fp_scores.csv', index=False); F.to_csv('sandhi_fp_rows.csv', index=False)
w = lambda g, col: sum(g.groupby('stratum')[col].mean()[k] * v for k, v in POP.items())
print(f'\nitems {len(ev)}; population-weighted F1')
print(f"{'system':6s} {'strict':>7s} {'cons':>7s} {'lenient':>8s} {'DCS-all':>8s}")
for s in systems:
    g = R[R.sys == s]
    print(f"{s:6s} {w(g,'strict'):7.3f} {w(g,'plaindcs'):7.3f} {w(g,'plain'):8.3f} {w(g,'dcsall'):8.3f}")
print('\nfalse positives by junction kind and DCS evidence (row %):')
F['dcs_splits'] = F.dcs.isin(ATTEST)
print(pd.crosstab([F.sys, F.kind], F.dcs_splits, margins=False).assign(
    pct_attested=lambda t: (t[True] / t.sum(axis=1) * 100).round(1)).to_string())
print('\nsandhi-junction false positives: Frontiers gold splits the same pair elsewhere (>=1 / >=3 times):')
for s in systems:
    q = F[(F.sys == s) & (F.kind == 'sandhi')]
    print(f"  {s:5s} n={len(q):5d}  >=1: {(q.fr_split>=1).mean()*100:5.1f}%   >=3: {(q.fr_split>=3).mean()*100:5.1f}%   "
          f"DCS-attested: {q.dcs_splits.mean()*100:5.1f}%")
q = F[(F.sys == 'byt5') & (F.kind == 'sandhi')]
print('\ntop ByT5 sandhi-junction FP pairs (count, DCS level, Frontiers splits elsewhere):')
top = q.groupby(['left', 'right']).agg(n=('dcs', 'size'), dcs=('dcs', 'first'), fr=('fr_split', 'first')).sort_values('n', ascending=False)
print(top.head(25).to_string())
