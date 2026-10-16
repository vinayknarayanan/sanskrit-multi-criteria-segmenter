import paths as P
import io, contextlib, pandas as pd
from indic_transliteration import sanscript
with contextlib.redirect_stdout(io.StringIO()):
    import robust_ci as R
    from rescore import score
slp = lambda s: sanscript.transliterate(s, 'iast', 'slp1')
U = P.RUNS + ''
ev = pd.read_csv(P.DATA + 'shared_eval_sample.csv').drop_duplicates('compound')
old = pd.read_csv('robust_scores_frontiers.csv')
keep = set(old[old.sys == 'byt5'].compound)
S = {}
for t in ('fr', 'dcs'):
    r = pd.read_csv(U + f'byt5_tagged_s2/results_{t}.csv'); p = dict(zip(r.compound, r.predicted))
    d = R.score_file(ev[ev.compound.isin(keep)], p)
    d['plain'] = [score([slp(x) for x in g.split('+')], [slp(x) for x in p[c].split()], slp(c))['plain_f']
                  for c, g in zip(d.compound, ev.set_index('compound').loc[d.compound].gold_split)]
    S[t] = d
old_plain = pd.read_csv('sandhi_fp_scores.csv')
O = {k: old[old.sys == k] for k in ('tagfr', 'tagdcs', 'byt5')}
print('items', len(S['fr']))
print(f"{'':12s} {'strict':>7s} {'agnost':>7s} {'lenient':>7s}")
for lab, d in (('seed2 <fr>', S['fr']), ('seed2 <dcs>', S['dcs']), ('seed1 <fr>', O['tagfr']), ('seed1 <dcs>', O['tagdcs']), ('ByT5-Skt', O['byt5'])):
    print(f"{lab:12s} " + ' '.join(f'{R.wmean(d, c):7.3f}' for c in ('strict', 'agnostic', 'plain')))
print('\nseed-2 tag-flip gaps (95% CI):')
for c in ('strict', 'agnostic', 'plain'):
    p, lo, hi, n = R.boot_gap(S['fr'], S['dcs'], c); print(f'  {c:8s} {p:+.3f} [{lo:+.3f}, {hi:+.3f}]')
print('seed-2 <dcs> - ByT5-Sanskrit:')
for c in ('strict', 'agnostic', 'plain'):
    p, lo, hi, n = R.boot_gap(S['dcs'], O['byt5'], c); print(f'  {c:8s} {p:+.3f} [{lo:+.3f}, {hi:+.3f}]')
f1, f2 = pd.read_csv(U + 'byt5_tagged_s2/results_fr.csv'), pd.read_csv(U + 'byt5_tagged_s2/results_dcs.csv')
print('\nexact match seed2 <fr>', round(f1.acc_exact.mean(), 3), '<dcs>', round(f2.acc_exact.mean(), 3), '| outputs differ', round((f1.predicted != f2.predicted).mean(), 3))
pd.concat([d.assign(sys=k) for k, d in S.items()]).to_csv('robust_scores_seed2.csv', index=False)
