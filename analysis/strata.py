"""Per-piece-count tag-flip scores (5 seeds) and the DCS-side gap under DCS's own piece-count distribution."""
# Run from results/:   python ../analysis/strata.py      -> writes strata_scores.csv (Appendix B)
import io, contextlib, os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
with contextlib.redirect_stdout(io.StringIO()):
    from robust_ci import score_file, POP
O = N = P.RUNS
ev = pd.read_csv(P.DATA + 'shared_eval_sample.csv').drop_duplicates('compound')
hv = pd.read_csv(P.DATA + 'dcs_holdout_eval.csv')
KEEP = set(pd.read_csv('robust_scores_frontiers.csv').compound)
TAG = {13: (O+'byt5_tagged/', O+'tagged_on_dcs/'), 42: (O+'byt5_tagged_s2/', O+'tagged_s2_on_dcs/')}
for s in (7, 21, 99): TAG[s] = (N+f'tagged_s{s}/', N+f'tagged_s{s}_on_dcs/')
def sc(e, f, keep=None):
    r = pd.read_csv(f); d = score_file(e, dict(zip(r.compound, r.predicted)))
    d = d[d.compound.isin(keep)] if keep is not None else d
    return d.merge(e[['compound', 'split_count']], on='compound')
# DCS natural distribution over distinct multi-member forms
# dcs_pairs_all.csv is written by build_dcs_pairs.py (run from results/)
dp = pd.read_csv('dcs_pairs_all.csv'); dp['n'] = dp.split.str.count(r'\+') + 1
nat = dp[dp.n.between(2, 6)].drop_duplicates('compound').n.value_counts(normalize=True).sort_index()
DW = nat.to_dict(); print('DCS natural weights (2-6):', {k: round(v, 3) for k, v in DW.items()}, 'n forms', dp.n.between(2,6).sum())
rows = []
for s, (fd, dd) in sorted(TAG.items()):
    for test, e, a, b, keep in (('Frontiers', ev, fd+'results_fr.csv', fd+'results_dcs.csv', KEEP), ('DCS', hv, dd+'results_dcs.csv', dd+'results_fr.csv', None)):
        A, B = sc(e, a, keep), sc(e, b, keep)
        for k in range(2, 7):
            rows.append(dict(seed=s, test=test, n=k, home=A[A.split_count == k].strict.mean(), other=B[B.split_count == k].strict.mean(),
                             home_agn=A[A.split_count == k].agnostic.mean(), other_agn=B[B.split_count == k].agnostic.mean(), items=(A.split_count == k).sum()))
R = pd.DataFrame(rows); R['gap'] = R.home - R.other; R['gap_agn'] = R.home_agn - R.other_agn
M = R.groupby(['test', 'n']).agg(items=('items', 'first'), home=('home', 'mean'), other=('other', 'mean'), gap=('gap', 'mean'), gap_min=('gap', 'min'), gap_max=('gap', 'max')).round(3)
print(M.to_string())
for test, W, lab in (('Frontiers', POP, 'Frontiers weights'), ('DCS', POP, 'Frontiers weights'), ('DCS', DW, 'DCS natural weights'), ('DCS', {k: .2 for k in range(2, 7)}, 'equal weights')):
    g = []
    for s in TAG:
        x = R[(R.test == test) & (R.seed == s)].set_index('n')
        g.append(sum(W[k] * x.loc[k, 'gap'] for k in W) / sum(W.values()))
    print(f'{test} gap, {lab}: mean {np.mean(g):.3f}  range {min(g):.3f}-{max(g):.3f}')
R.to_csv('strata_scores.csv', index=False)
