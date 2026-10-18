"""Ranking reversal on each dataset's own test set and gold (Section 5.1): the Frontiers-only model against the <dcs>
setting of the tagged model, per seed pair. Run from results/:   python ../analysis/authentic_reversal.py"""
import io, contextlib, os, sys, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
with contextlib.redirect_stdout(io.StringIO()):
    from robust_ci import score_file, wmean, boot_gap
R = P.RUNS
ev = pd.read_csv(P.DATA + 'shared_eval_sample.csv').drop_duplicates('compound')
hv = pd.read_csv(P.DATA + 'dcs_holdout_eval.csv')
KEEP = set(pd.read_csv('robust_scores_frontiers.csv').compound)
def sc(e, f, keep=None):
    r = pd.read_csv(f); d = score_file(e, dict(zip(r.compound, r.predicted)))
    return d[d.compound.isin(keep)] if keep is not None else d
# seed pairs: tagged run (Frontiers eval, DCS eval)  x  Frontiers-only run (Frontiers eval, DCS eval)
TAG = {1: ('byt5_tagged', 'tagged_on_dcs'), 2: ('byt5_tagged_s2', 'tagged_s2_on_dcs')}
for i, s in ((3, 7), (4, 21), (5, 99)): TAG[i] = (f'tagged_s{s}', f'tagged_s{s}_on_dcs')
FON = {1: ('byt5_frontiers_v2', 'frontiers_s13_on_dcs'), 2: ('frontiers_s42', 'frontiers_s42_on_dcs'), 3: ('frontiers_s7', 'frontiers_s7_on_dcs'),
       4: ('frontiers_s21', 'frontiers_s21_on_dcs'), 5: ('frontiers_s99', 'frontiers_s99_on_dcs')}
rows = []
for i in sorted(TAG):
    tf, td = TAG[i]; ff, fd = FON[i]
    a, b = sc(ev, R + ff + '/results_byt5_frontiers.csv', KEEP), sc(ev, R + tf + '/results_dcs.csv', KEEP)
    c, d = sc(hv, R + fd + '/results_byt5_frontiers.csv'), sc(hv, R + td + '/results_dcs.csv')
    g1, g2 = boot_gap(a, b, 'strict')[:3], boot_gap(c, d, 'strict')[:3]
    rows.append(dict(seed=i, fr_fonly=wmean(a, 'strict'), fr_dcs=wmean(b, 'strict'), fr_gap=g1[0], fr_lo=g1[1], fr_hi=g1[2],
                     dcs_fonly=wmean(c, 'strict'), dcs_dcs=wmean(d, 'strict'), dcs_gap=g2[0], dcs_lo=g2[1], dcs_hi=g2[2]))
T = pd.DataFrame(rows).round(3); print(T.to_string(index=False)); T.to_csv('authentic_reversal.csv', index=False)
