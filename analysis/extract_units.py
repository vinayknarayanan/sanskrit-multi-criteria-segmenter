"""Verse-level SIGHUM / Hackathon -> word-unit pairs (written token -> its pieces), comparable to Frontiers rows.
Each output piece is assigned to the input token where its first character aligns (difflib on the space-free strings).
A verse is kept only if every token receives >=1 piece; per-unit similarity of token vs joined pieces is recorded."""
import paths as P
import difflib, glob, unicodedata, sys, pandas as pd
nf = lambda s: unicodedata.normalize('NFC', s).strip()
U = P.HF + ''

def units(sent, uns):
    toks, pcs = nf(sent).split(), nf(uns).split()
    s, t = ''.join(toks), ''.join(pcs)
    ends, a = [], 0
    for w in toks: a += len(w); ends.append(a)          # token k covers s[ends[k-1]:ends[k]]
    ops = difflib.SequenceMatcher(None, t, s, autojunk=False).get_opcodes()
    def map_(i):
        for tag, i1, i2, j1, j2 in ops:
            if i1 <= i < i2 or (i == i2 == len(t)):
                return j1 + (i - i1) if tag == 'equal' else j1 + min(round((i - i1) * (j2 - j1) / max(i2 - i1, 1)), max(j2 - j1 - 1, 0))
        return len(s) - 1
    groups = [[] for _ in toks]; off = 0
    for p in pcs:
        j = map_(off); k = next(n for n, e in enumerate(ends) if j < e) if j < len(s) else len(toks) - 1
        groups[k].append(p); off += len(p)
    if any(not g for g in groups): return None
    return [(w, '+'.join(g), difflib.SequenceMatcher(None, w, ''.join(g)).ratio()) for w, g in zip(toks, groups)]

rows, bad = [], {}
for ds in ('sighum', 'hackathon'):
    for split in ('train', 'test', 'validation'):
        fs = glob.glob(f'{U}sanskrit-sandhi-split-{ds}/data/{split}-*.parquet')
        if not fs: continue
        df = pd.read_parquet(fs[0]); nb = 0
        for vid, r in enumerate(df.itertuples()):
            u = units(r.sentence, r.unsandhied)
            if u is None: nb += 1; continue
            for w, sp, sim in u: rows.append((ds, split, vid, w, sp, sp.count('+') + 1, round(sim, 3)))
        bad[(ds, split)] = (nb, len(df))
R = pd.DataFrame(rows, columns=['dataset', 'split', 'verse', 'surface', 'split_pieces', 'n_pieces', 'sim'])
R.to_csv(P.CACHE + 'units_sighum_hackathon.csv', index=False)
print('verses dropped (no clean alignment):', bad)
print(R.groupby(['dataset', 'split']).agg(units=('surface', 'size'), multi=('n_pieces', lambda x: (x > 1).sum()), low_sim=('sim', lambda x: (x < .6).sum())))
