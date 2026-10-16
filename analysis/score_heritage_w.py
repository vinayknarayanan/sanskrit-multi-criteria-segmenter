"""Heritage scoring, population-weighted, three item sets:
  answered  items where that mode returned >=1 split (what score_heritage.py reports; flatters Heritage)
  all       all 3,259 items, no output = F1 0 (no split proposed -> recall 0)
  common    items where BOTH modes answered (fair p vs t comparison)
Oracle over up to 1,000 distinct candidates."""
import io, contextlib, json, sys, pandas as pd
from indic_transliteration import sanscript
with contextlib.redirect_stdout(io.StringIO()):
    from rescore import score
slp = lambda s: sanscript.transliterate(s, 'iast', 'slp1')
POP = {2: .709, 3: .212, 4: .057, 5: .017, 6: .005}
def wm(df, c):
    g = df.groupby('stratum')[c].mean(); return sum(g[k]*v for k, v in POP.items() if k in g.index)/sum(v for k, v in POP.items() if k in g.index)
h = pd.read_csv(sys.argv[1]); rows = []
for x in h.itertuples():
    raw = json.loads(x.candidates_json or '[]'); cands = []
    for c in raw:
        if c not in cands: cands.append(c)
    g, s = [slp(p) for p in x.gold_split.split('+')], slp(x.compound)
    if cands:
        sc = [score(g, [slp(p) for p in c], s) for c in cands[:1000]]
        t1, orc = sc[0]['strict_f'], max(d['strict_f'] for d in sc); t1l = sc[0]['plain_f']; t1c = sc[0]['plaindcs_f']
    else: t1 = orc = t1l = t1c = 0.0
    rows.append(dict(compound=x.compound, mode=x.mode, stratum=x.split_count, answered=bool(cands), n=len(cands),
                     top1=t1, cons=t1c, len_=t1l, oracle=orc, err=str(x.error) if isinstance(x.error, str) else ''))
r = pd.DataFrame(rows); r.to_csv('rescore_heritage_modes_w.csv', index=False)
both = set(r[r['mode'] == 'p'].query('answered').compound) & set(r[r['mode'] == 't'].query('answered').compound)
for m in ('p', 't'):
    g = r[r['mode'] == m]
    for lab, sub in (('answered', g[g.answered]), ('all', g), ('common', g[g.compound.isin(both)])):
        print(f"{m} {lab:8s} n={len(sub):4d}  top1 strict {wm(sub,'top1'):.3f} cons {wm(sub,'cons'):.3f} lenient {wm(sub,'len_'):.3f}  oracle {wm(sub,'oracle'):.3f}")
print('common items', len(both))
