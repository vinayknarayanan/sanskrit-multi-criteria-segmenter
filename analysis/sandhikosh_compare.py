"""SandhiKosh (Bhardwaj et al., LREC 2018) vs DCS and vs the Frontiers sample. No model involved.

SandhiKosh is a sandhi benchmark: each row gives a written sequence and a split that marks the junction(s) under test.
For each SandhiKosh piece we ask the same question as for Frontiers: does DCS divide this piece further?
Also: how many SandhiKosh rows appear in the Frontiers sample (Frontiers derives from UoH; SandhiKosh has a UoH sheet).
"""
import paths as P
import glob, re, pickle, pandas as pd
from collections import Counter
from indic_transliteration import sanscript
from dcs_surface_index import key, base
from corpus_compare import boundary_type, IDX

iast = lambda s: sanscript.transliterate(str(s), 'devanagari', 'iast')
rows = []
for f in glob.glob(P.SANDHIKOSH + '*.xls'):
    for sheet, df in pd.read_excel(f, sheet_name=None).items():
        df.columns = [str(c).strip().lower() for c in df.columns]
        if 'word' not in df.columns:  # header sits in the first data row
            df.columns = [str(c).strip().lower() for c in df.iloc[0]]; df = df.iloc[1:]
        name = (f.split('/')[-1].replace('.xls', '') + ('' if sheet.startswith('Sheet') else f' / {sheet}'))
        for w, s in zip(df['word'], df['split']):
            if pd.isna(w) or pd.isna(s): continue
            rows.append(dict(corpus=name, word=iast(w).strip(), split=iast(s).strip()))
sk = pd.DataFrame(rows)
sk['n_pieces'] = sk.split.str.count(r'\+') + 1
sk['has_space'] = sk.word.str.contains(' ')

out = []
for r in sk.itertuples():
    for p in r.split.split('+'):
        p = p.strip()
        if len(p) <= 3: continue
        an = IDX.get(key(p))
        if not an: out.append(dict(corpus=r.corpus, piece=p, status='not in DCS')); continue
        top = an.most_common(1)[0][0]
        st = 'agree: both one unit' if len(top) == 1 else 'DCS divides, SandhiKosh does not'
        out.append(dict(corpus=r.corpus, piece=p, status=st, dcs=' + '.join(m[0] for m in top),
                        types='; '.join(boundary_type(a, b) for a, b in zip(top, top[1:])) if len(top) > 1 else ''))
pw = pd.DataFrame(out); pw.to_csv('corpus_sandhikosh_vs_dcs.csv', index=False)

print('SandhiKosh rows by sheet:'); print(sk.groupby('corpus').agg(rows=('word', 'size'), mean_pieces=('n_pieces', 'mean'),
      with_space=('has_space', 'mean')).round(2).to_string(), '\n')
t = pw.groupby('corpus').status.value_counts().unstack(fill_value=0)
t['found %'] = (1 - t['not in DCS'] / t.sum(axis=1)) * 100
m = t['agree: both one unit'] + t['DCS divides, SandhiKosh does not']
t['DCS divides % of found'] = t['DCS divides, SandhiKosh does not'] / m * 100
print(t.round(1).to_string(), '\n')
div = pw[pw.status == 'DCS divides, SandhiKosh does not']
print('boundary types:', dict(Counter(x for s in div.types for x in s.split('; '))))
print(div.sample(min(12, len(div)), random_state=2)[['corpus', 'piece', 'dcs', 'types']].to_string())

# overlap with Frontiers sample
fr = pd.read_csv(P.DATA + 'shared_eval_sample.csv')
fr_surf = set(fr.compound.map(base)); fr_split = {base(g.replace('+', '')) for g in fr.gold_split}
sk['surf'] = sk.word.map(lambda w: base(w.replace(' ', '')))
hit = sk[sk.surf.isin(fr_surf)]
print(f'\nSandhiKosh rows whose written form appears in the Frontiers sample: {len(hit)} of {len(sk)}')
print(hit.groupby('corpus').size().to_string())
if len(hit):
    j = hit.merge(fr.assign(surf=fr.compound.map(base)), on='surf')
    j['same_split'] = [base(a.replace(' ', '')) == base(b.replace('+', '')) and a.count('+') == b.count('+') for a, b in zip(j.split, j.gold_split)]
    print('identical split:', j.same_split.sum(), 'of', len(j))
    print(j[['word', 'split', 'gold_split']].head(15).to_string())
    j.to_csv('corpus_sandhikosh_frontiers_overlap.csv', index=False)
