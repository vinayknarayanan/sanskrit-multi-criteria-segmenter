"""Score heritage_modes.csv (from run_heritage_modes.py): per mode, top-1 and oracle-of-n boundary-F1,
strict / conservative / lenient, next to ByT5, the old INRIA run and rcNN.
Usage: python score_heritage.py heritage_modes.csv"""
import sys, json, pandas as pd
from indic_transliteration import sanscript
from rescore import score

slp = lambda s: sanscript.transliterate(s, 'iast', 'slp1')
h = pd.read_csv(sys.argv[1])
rows = []
for x in h.itertuples():
    raw = json.loads(x.candidates_json or '[]')
    cands = []                         # Heritage lists morphological variants of the same split: keep distinct splits, in rank order
    for c in raw:
        if c not in cands: cands.append(c)
    if not cands: continue
    g, s = [slp(p) for p in x.gold_split.split('+')], slp(x.compound)
    sc = [score(g, [slp(p) for p in c], s) for c in cands[:100]]
    best = max(sc, key=lambda d: d['strict_f'])
    rows.append(dict(compound=x.compound, mode=x.mode, n=len(cands), n_raw=len(raw), stratum=x.split_count,
                     **{f'top1_{k}': v for k, v in sc[0].items() if k.endswith(('_p', '_r', '_f'))},
                     **{f'oracle_{k}': v for k, v in best.items() if k.endswith('_f')}))
r = pd.DataFrame(rows); r.to_csv('rescore_heritage_modes.csv', index=False)
for mode, g in r.groupby('mode'):
    print(f'\n== mode {mode}: items {len(g)}, median distinct splits {g.n.median():.0f} (raw {g.n_raw.median():.0f}), mean {g.n.mean():.1f}')
    print(f"  top-1  strict F1 {g.top1_strict_f.mean():.3f}  conservative {g.top1_plaindcs_f.mean():.3f}  lenient {g.top1_plain_f.mean():.3f}")
    print(f"  oracle strict F1 {g.oracle_strict_f.mean():.3f}")
    print(g.groupby('stratum')[['top1_strict_f', 'oracle_strict_f']].mean().round(3).to_string())
