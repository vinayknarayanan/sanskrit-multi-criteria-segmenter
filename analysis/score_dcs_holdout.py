import paths as P
import io, contextlib, pandas as pd
from indic_transliteration import sanscript
with contextlib.redirect_stdout(io.StringIO()):
    from rescore import score
slp = lambda s: sanscript.transliterate(s, 'iast', 'slp1')
base = P.RUNS + 'tagged_on_dcs/'
out = {}
for t in ('fr', 'dcs'):
    r = pd.read_csv(base + f'results_{t}.csv'); rows = []
    for x in r.itertuples():
        if not isinstance(x.predicted, str) or not x.predicted.strip(): continue
        s = score([slp(p) for p in x.gold_split.split('+')], [slp(p) for p in x.predicted.split()], slp(x.compound))
        rows.append(dict(stratum=x.split_count, exact=x.acc_exact, over=s['n_cand'] > s['n_gold'], under=s['n_cand'] < s['n_gold'], **{k: s[k] for k in ('strict_p', 'strict_r', 'strict_f', 'plain_f', 'gold_derived')}))
    out[t] = pd.DataFrame(rows)
for t, d in out.items():
    print(f"<{t}> on DCS gold: n={len(d)} exact {d.exact.mean():.3f} | strict P {d.strict_p.mean():.3f} R {d.strict_r.mean():.3f} F1 {d.strict_f.mean():.3f} | lenient F1 {d.plain_f.mean():.3f} | over {d.over.mean():.3f} under {d.under.mean():.3f} | gold derivable {d.gold_derived.mean():.3f}")
print(pd.DataFrame({t: d.groupby('stratum').strict_f.mean() for t, d in out.items()}).round(3).to_string())
