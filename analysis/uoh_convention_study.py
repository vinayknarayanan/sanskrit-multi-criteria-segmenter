"""UoH / SaCTI segmentation convention study.
SaCTI rows = sentences; one target compound per sentence carries the compound-type label (last row repeats it).
1) consistency of UoH splitting: annotated targets vs context words
2) UoH vs DCS on target compounds, incl. cases where DCS has the whole as ONE word
3) what those cases are (negation / prādi / verbal prefix (gati) / sa- / lexicalised noun+noun) x compound type
Needs: SaCTI repo (github.com/Jivnesh/SaCTI), corpus_uoh_vs_dcs.csv, dcs_surface.pkl (from earlier kt4 scripts)."""
import paths as P
import glob, pickle, unicodedata, pandas as pd
from indic_transliteration import sanscript
b = lambda s: unicodedata.normalize('NFC', str(s).strip().lower()).replace('ṃ', 'm')
iast = lambda s: sanscript.transliterate(s, 'devanagari', 'iast')
rows = []
for f in glob.glob(P.SACTI + '*.conll'):
    sent = []
    for line in list(open(f, encoding='utf-8')) + ['\n']:
        p = line.rstrip('\n').split('\t')
        if len(p) < 9:
            if sent:
                typ, tw = sent[0][7], sent[-1][1]
                hit = next((i for i, x in enumerate(sent[:-1]) if x[1] == tw), None)
                for i, x in enumerate(sent[:-1]):
                    parts = tuple(b(y) for y in iast(x[1]).split('-') if y.strip())
                    rows.append((''.join(parts), parts, i == hit, typ if i == hit else ''))
            sent = []; continue
        sent.append(p)
R = pd.DataFrame(rows, columns=['w', 'parts', 'target', 'type'])
print('1) consistency (words seen >=2x, share divided more than one way)')
for lab, sub in (('targets', R[R.target]), ('context', R[~R.target])):
    g = sub.groupby('w').parts.nunique(); c = sub.groupby('w').size()
    print(f'   {lab}: {(g[c >= 2] > 1).mean():.1%} of {int((c >= 2).sum())}')
both = R[R.w.isin(set(R[R.target].w) & set(R[~R.target].w))]
print('   same words, split rate context vs target:', both.assign(s=both.parts.map(len) > 1).groupby('target').s.mean().round(3).to_dict())
tg = R[R.target]; typ = tg.assign(k=tg.parts.map(' - '.join)).groupby('k').type.agg(lambda x: x.mode()[0])
u = pd.read_csv('corpus_uoh_vs_dcs.csv'); W = pickle.load(open(P.CACHE + 'dcs_surface.pkl', 'rb'))['words']
u['k'] = u.uoh.map(lambda s: ' - '.join(b(x) for x in s.split(' - '))); u = u[u.k.isin(typ.index)].copy(); u['type'] = u.k.map(typ)
u['one'] = (u.status == 'not in DCS') & u.k.map(lambda s: W.get(''.join(s.split(' - ')), 0) > 0)
comp = u[(u.status != 'not in DCS') | u.one]
print(f'\n2) UoH targets vs DCS, comparable {len(comp)}:'); print((comp.status.where(~comp.one, 'UoH splits, DCS one word').value_counts(normalize=True) * 100).round(1).to_string())
PRE = {'pra', 'sam', 'vi', 'nir', 'nis', 'pari', 'anu', 'upa', 'prati', 'ā', 'abhi', 'adhi', 'ava', 'ut', 'ud', 'apa', 'ati'}
def cat(f):
    if f in ('a', 'an', 'na'): return 'negation (nañ)'
    if f in ('su', 'dur', 'dus', 'ku'): return 'su-/dur- (prādi)'
    if f in ('sa', 'saha'): return 'sa- (bahuvrīhi)'
    if f in PRE: return 'verbal prefix (gati/avyayībhāva)'
    if f in ('yathā', 'sva', 'eka', 'mahat', 'mahā'): return 'yathā-/sva-/mahā-'
    return 'noun+noun (lexicalised in DCS)'
d = u[u.one].copy(); d['cat'] = d.k.str.split(' - ').str[0].map(cat)
print('\n3) UoH splits where DCS keeps one word, by kind x compound type'); print(pd.crosstab(d.cat, d.type, margins=True).to_string())
d[['k', 'type', 'cat']].rename(columns={'k': 'uoh'}).to_csv('uoh_target_vs_dcs_oneword.csv', index=False)
