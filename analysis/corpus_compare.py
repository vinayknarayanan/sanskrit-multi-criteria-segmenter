"""Frontiers gold vs DCS, no model involved.

Forward: each Frontiers gold piece is one unit in Frontiers. Look it up by written form in DCS.
  DCS keeps it as one word -> agree.  DCS divides it -> classify every DCS-internal boundary.
Reverse: two adjacent Frontiers pieces a+b (b consonant-initial, a not ending in a final-sandhi consonant, so plain
  concatenation equals the written form) -> does DCS ever have a+b as ONE word? -> Frontiers divides where DCS doesn't.
"""
import paths as P
import re, pickle, pandas as pd
from collections import Counter
from dcs_surface_index import key, base

X = pickle.load(open(P.CACHE + 'dcs_surface.pkl', 'rb')); IDX, W = X['idx'], X['words']
SUF = {'tva', 'tā', 'vat', 'mat', 'vant', 'mant'}
PART = {'iti', 'api', 'eva', 'ca', 'tu', 'hi', 'na', 'vā', 'iva', 'atra', 'tatra', 'yatra', 'atas', 'tatas', 'punar', 'tathā', 'yathā', 'evam', 'iha', 'nu'}

def boundary_type(left, right):
    (lu, ll, lc), (ru, rl, rc) = left, right
    if rl in SUF: return 'secondary suffix (-tva/-tā/-vat)'
    if ll in PART or rl in PART: return 'particle'
    if ll in {'a', 'an', 'su', 'dus', 'nis', 'sa', 'ku'} or lu in {'a', 'an', 'su', 'dus', 'duḥ', 'niḥ', 'sa', 'ku'}: return 'prefix (a-/an-/su-/dus-)'
    if rl == 'ādi' or ru.startswith('ād'): return 'ādi'
    if lc: return 'compound member'
    return 'separate words'

d = pd.read_csv(P.DATA + 'shared_eval_sample.csv')
rows = []
for item in d.itertuples():
    pieces = item.gold_split.split('+')
    for i, p in enumerate(pieces):
        an = IDX.get(key(p))
        if not an:
            rows.append(dict(compound=item.compound, stratum=item.split_count, piece=p, status='not in DCS')); continue
        top, n = an.most_common(1)[0]; tot = sum(an.values())
        dcs_div = ' + '.join(m[0] for m in top)
        r = dict(compound=item.compound, stratum=item.split_count, piece=p, dcs=dcs_div, dcs_n=len(top),
                 dcs_freq=tot, dcs_variants=len({tuple(m[0] for m in t) for t in an}), dcs_top_share=round(n / tot, 3))
        if len(p) <= 3 and len(top) == 1:
            r['status'] = 'agree (short word)'
        elif len(top) == 1:
            r['status'] = 'agree: both one unit'
        else:
            r['status'] = 'DCS divides, Frontiers does not'
            r['types'] = '; '.join(boundary_type(a, b) for a, b in zip(top, top[1:]))
        rows.append(r)
fw = pd.DataFrame(rows); fw.to_csv('corpus_frontiers_vs_dcs_forward.csv', index=False)

rev = []
for item in d.itertuples():
    ps = item.gold_split.split('+')
    for a, b in zip(ps, ps[1:]):
        if re.match(r'^[aāiīuūṛeo]', b) or re.search(r'[ḥmtdnṃ]$', a): continue
        joined = base(a + b)
        rev.append(dict(compound=item.compound, a=a, b=b, joined=joined, dcs_one_word=W.get(joined, 0)))
rv = pd.DataFrame(rev); rv.to_csv('corpus_frontiers_vs_dcs_reverse.csv', index=False)

print('FORWARD  gold pieces:', len(fw), '| found in DCS:', (fw.status != 'not in DCS').sum(),
      f"({(fw.status != 'not in DCS').mean()*100:.1f}%)")
print(fw.status.value_counts().to_string(), '\n')
div = fw[fw.status == 'DCS divides, Frontiers does not']
bt = Counter(t for s in div.types for t in s.split('; '))
print('DCS-internal boundaries inside Frontiers units, by type:', dict(bt))
m = fw[fw.status.str.startswith('agree: both') | (fw.status == 'DCS divides, Frontiers does not')]
print(f'Among multi-char matched pieces: DCS divides {len(div)}/{len(m)} = {len(div)/len(m)*100:.1f}%')
print('DCS self-consistency: matched pieces whose DCS form has >1 analysis:',
      f"{(m.dcs_variants > 1).mean()*100:.1f}%  (median top-analysis share {m.dcs_top_share.median()})")
print('\nby stratum, share DCS divides:'); print((m.groupby('stratum').status.apply(lambda s: (s != 'agree: both one unit').mean())*100).round(1).to_string())
print('\nREVERSE  adjacent Frontiers pairs testable:', len(rv), '| DCS has them as ONE word:', (rv.dcs_one_word > 0).sum(),
      f"({(rv.dcs_one_word > 0).mean()*100:.1f}%)")
print(rv[rv.dcs_one_word > 0].sort_values('dcs_one_word', ascending=False)[['a', 'b', 'dcs_one_word']].head(12).to_string())
print('\nexamples DCS divides:'); print(div.sample(15, random_state=1)[['piece', 'dcs', 'types', 'dcs_freq']].to_string())
