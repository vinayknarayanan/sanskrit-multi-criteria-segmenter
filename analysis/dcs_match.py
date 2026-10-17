"""Label every ByT5 extra cut by DCS evidence.

Levels (first that applies):
  A_pair   the exact form pair (left, right) occurs adjacently in DCS
  B_lemma  some lemma of `left` is followed by some lemma of `right` in DCS
  C_both   both parts are attested DCS forms, but never adjacent
  D_unatt  at least one part is not an attested DCS form (candidate ByT5 error, or DCS coverage gap)
A and B count as "DCS splits here" (convention evidence). C means both parts are real words, which is necessary
for a valid division but not sufficient. D is the weakest.
"""
import paths as P
import pickle, pandas as pd
from dcs_index import norm

X = pickle.load(open(P.CACHE + 'dcs_index.pkl', 'rb'))
f2l, AC, AW, FP = X['f2l'], X['adj_cpd'], X['adj_word'], X['fpairs']
adj = AC + AW

SHORT_OK = {'a', 'an', 'sa', 'go', 'su', 'ku', 'dus', 'duḥ', 'niḥ', 'sva'}
def level(l, r, cls):
    L, R = norm(l), norm(r)
    if (len(L) < 2 and L not in SHORT_OK) or len(R) < 2: return 'X_fragment'   # letter-level cut: unscorable
    if cls == 'suffix': return 'S_policy'   # DCS lemmatises tva/tā/vat as separate tokens (checked 23 Sept 2026)
    inside = cls in ('other', 'ādi')        # compound-internal cut -> require compound-internal evidence
    if FP.get((L, R, True), 0) or (not inside and FP.get((L, R, False), 0)): return 'A_pair'
    if L in f2l and R in f2l:
        A = AC if inside else adj
        if any(A.get((a, b), 0) for a in f2l[L] for b in f2l[R]): return 'B_lemma'
        return 'C_both'
    return 'D_unatt'

c = pd.read_csv('suffix_extra_cuts.csv')
c['cls'] = c.suffix.replace({'-tva': 'suffix', '-tā': 'suffix', '-vat/-vant': 'suffix', '-mat/-mant': 'suffix'})
c['dcs'] = [level(l, r, k) for l, r, k in zip(c.left, c.right, c.cls)]
c.to_csv('suffix_extra_cuts_dcs.csv', index=False)
print(f"DCS: {X['nfiles']} files, {X['ntok']:,} tokens\n")
t = pd.crosstab(c.cls, c.dcs, margins=True); print(t, '\n')
print((pd.crosstab(c.cls, c.dcs, normalize='index') * 100).round(1), '\n')
ok = c[c.align_cost <= 1.0]
print('well-aligned only (cost<=1.0), share "DCS splits here" (A or B):',
      round(ok.dcs.isin(['A_pair', 'B_lemma', 'S_policy']).mean() * 100, 1), '% of', len(ok))
print('\nby gold split_count, "other" cuts, share A or B:')
o = c[c.cls == 'other']
print((o.groupby('split_count').dcs.apply(lambda s: s.isin(['A_pair', 'B_lemma']).mean()) * 100).round(1).to_string())
print('\nexamples per level (other):')
for lv in ['A_pair', 'B_lemma', 'C_both', 'D_unatt']:
    print(lv, list(o[o.dcs == lv].apply(lambda r: f'{r.left}|{r.right}', axis=1).head(10)))
