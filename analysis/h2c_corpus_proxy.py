"""H2c proxy without annotators: are junctions where the <fr>/<dcs> settings disagree also junctions where DCS's own
annotators were inconsistent (same written word, cut here in some occurrences and not in others)?

Words: Frontiers eval (3,259) + held-out DCS (3,000) + 6,000 frequent DCS words (dcs_frequent_eval.csv), all that occur >= 5 times in DCS.
Candidate junctions per word: any cut proposed by <fr>, <dcs>, or any DCS analysis (merged within +-1 char).
For each junction: model group  A = both settings cut, D = exactly one cuts, N = neither cuts (only DCS analyses do)
                   p = share of the word's DCS occurrences that cut here
                   contested = minority share min(p, 1-p) >= 0.10
Not circular by construction: the model's DCS training pairs kept ONE (most frequent) analysis per written form,
and held-out DCS words were removed from training entirely.
Caveat: DCS index keys strip one final letter, so a few distinct words may share an entry."""
import paths as P
import io, contextlib, pickle, re, sys, unicodedata, numpy as np, pandas as pd
sys.path.insert(0, P.ROOT + '/analysis')
with contextlib.redirect_stdout(io.StringIO()):
    from robust_ci import cuts_on_surface
b = lambda s: unicodedata.normalize('NFC', str(s).strip().lower()).replace('ṃ', 'm')
key = lambda s: re.sub(r'(?:[ḥmsrtdn]|o|e)$', '', b(s)) if len(b(s)) > 3 else b(s)
IDX = pickle.load(open(P.CACHE + 'dcs_surface.pkl', 'rb'))['idx']
CS = pickle.load(open(P.CACHE + 'cpd_share.pkl', 'rb'))
U = P.RUNS + ''
rows = []
for src, ff, fd in (('frontiers', U + 'byt5_tagged/results_fr.csv', U + 'byt5_tagged/results_dcs.csv'),
                    ('dcs_holdout', U + 'tagged_on_dcs/results_fr.csv', U + 'tagged_on_dcs/results_dcs.csv'),
                    ('dcs_frequent', U + 'tagged_on_dcsfreq/results_fr.csv', U + 'tagged_on_dcsfreq/results_dcs.csv')):
    fr = pd.read_csv(ff).drop_duplicates('compound').set_index('compound').predicted
    dc = pd.read_csv(fd).drop_duplicates('compound').set_index('compound').predicted
    for w in fr.index.intersection(dc.index):
        if not isinstance(fr[w], str) or not isinstance(dc[w], str) or len(w) < 6: continue
        an = IDX.get(key(w))
        if not an or sum(an.values()) < 5: continue
        wb = b(w); tot = sum(an.values())
        dcs_cuts = [(cuts_on_surface([m[0] for m in tup], wb), n, tup) for tup, n in an.items()]
        cf, cd = cuts_on_surface(fr[w].split(), w), cuts_on_surface(dc[w].split(), w)
        cand = sorted(set(cf) | set(cd) | {c for cs, _, _ in dcs_cuts for c in cs})
        merged = []
        for c in cand:
            if 0 < c < len(w) - 1 and (not merged or c - merged[-1] > 1): merged.append(c)
        for c in merged:
            hit = lambda cs: any(abs(c - x) <= 1 for x in cs)
            p = sum(n for cs, n, _ in dcs_cuts if hit(cs)) / tot
            # left piece (for junction type): from whichever analysis cuts here
            left = next((tup[[abs(c - x) <= 1 for x in cs].index(True)][0] for cs, n, tup in dcs_cuts if hit(cs)), None)
            if left is None:
                for pcs in (fr[w].split(), dc[w].split()):
                    cc = cuts_on_surface(pcs, w)
                    if hit(cc): left = pcs[[abs(c - x) <= 1 for x in cc].index(True)]; break
            v = CS.get(b(left)) if left else None
            jt = 'unknown' if v is None else 'compound' if v >= .8 else 'between_words' if v <= .2 else 'mixed'
            nf, nd = hit(cf), hit(cd)
            rows.append(dict(src=src, word=w, cut=c, n_dcs=tot, p=p, group='A' if nf and nd else 'D' if nf != nd else 'N',
                             cut_by='both' if nf and nd else 'fr_only' if nf else 'dcs_only' if nd else 'neither', junction=jt))
R = pd.DataFrame(rows); R['minority'] = np.minimum(R.p, 1 - R.p); R['contested'] = R.minority >= .10
R.to_csv('h2c_proxy_junctions.csv', index=False)
print('words', R.word.nunique(), 'junctions', len(R)); print(R.groupby(['group']).size().to_dict())
def boot(df, g1, g2, reps=2000):
    words = df.word.unique(); rng = np.random.default_rng(0); by = {w: d for w, d in df.groupby('word')}
    pt = df[df.group == g1].contested.mean() - df[df.group == g2].contested.mean(); vals = []
    for _ in range(reps):
        s = pd.concat([by[w] for w in rng.choice(words, len(words))])
        vals.append(s[s.group == g1].contested.mean() - s[s.group == g2].contested.mean())
    return pt, *np.nanpercentile(vals, [2.5, 97.5])
print('\ncontested rate (DCS annotators inconsistent here, minority >= 10%):')
print(R.groupby(['group']).agg(contested=('contested', 'mean'), minority=('minority', 'mean'), n=('word', 'size')).round(3).to_string())
for jt in ('compound', 'between_words'):
    s = R[R.junction == jt]
    print(f'\n[{jt}]'); print(s.groupby('group').agg(contested=('contested', 'mean'), n=('word', 'size')).round(3).to_string())
    for g2 in ('A', 'N'):
        if (s.group == g2).sum() >= 20: pt, lo, hi = boot(s, 'D', g2); print(f'  D - {g2}: {pt:+.3f}  [{lo:+.3f}, {hi:+.3f}]  (word-cluster bootstrap)')
print('\nby source:'); print(R.groupby(['src', 'group']).contested.agg(['mean', 'size']).round(3).to_string())
print('\nwithin D by which side cuts:'); print(R[R.group == 'D'].groupby('cut_by').contested.agg(['mean', 'size']).round(3).to_string())
