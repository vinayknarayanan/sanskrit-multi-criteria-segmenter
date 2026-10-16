"""Frontiers gold vs UoH (SaCTI) compound annotation, no model.

A UoH compound (members in dictionary form) is matched to a run of 1..6 consecutive Frontiers gold pieces whose
concatenation equals the concatenated UoH members. Then the boundaries are compared in that shared string:
Frontiers' piece boundaries inside the run vs UoH's member boundaries.
Caveat: a Frontiers piece that applies internal vowel sandhi (maṅgalācāra) cannot match UoH's unsandhied
concatenation (maṅgalaācāra); such compounds are simply not found, which biases the found set toward
consonant-joined compounds.
"""
import paths as P
import glob, pandas as pd
from collections import Counter
from indic_transliteration import sanscript
from uoh_vs_dcs import base, offsets

iast = lambda s: sanscript.transliterate(s, 'devanagari', 'iast')
U = {}
for f in glob.glob(P.SACTI + '*.conll'):
    for line in open(f, encoding='utf-8'):
        p = line.split('\t')
        if len(p) > 1 and '-' in p[1]:
            parts = [base(iast(x)) for x in p[1].strip().split('-') if x.strip()]
            if len(parts) > 1: U.setdefault(base(''.join(parts)), parts)

d = pd.read_csv(P.DATA + 'shared_eval_sample.csv')
rows = []
for it in d.itertuples():
    ps = [base(p) for p in it.gold_split.split('+')]
    for i in range(len(ps)):
        for j in range(i + 1, min(len(ps), i + 6) + 1):
            k = ''.join(ps[i:j])
            if k in U and len(k) > 5:
                up = U[k]; fb, ub = set(offsets(ps[i:j])), set(offsets(up))
                st = ('identical' if fb == ub else 'UoH divides further' if ub > fb else
                      'Frontiers divides further' if fb > ub else 'different boundaries')
                rows.append(dict(compound=it.compound, frontiers=' + '.join(ps[i:j]), uoh=' - '.join(up), status=st,
                                 uoh_extra='; '.join(k[b:min([x for x in ub | {len(k)} if x > b])] for b in sorted(ub - fb))))
x = pd.DataFrame(rows).drop_duplicates(['compound', 'frontiers'])
x.to_csv('corpus_frontiers_vs_uoh.csv', index=False)
print('matched Frontiers runs:', len(x), 'in', x.compound.nunique(), 'items')
print(x.status.value_counts().to_string()); print((x.status.value_counts(normalize=True) * 100).round(1).to_string())
print('\nUoH boundaries Frontiers drops, right-hand member (top):',
      Counter(t for s in x.uoh_extra.fillna('') for t in s.split('; ') if t).most_common(20))
for st in ['UoH divides further', 'Frontiers divides further', 'different boundaries']:
    s = x[x.status == st]
    if len(s): print(f'\n== {st}'); print(s.sample(min(10, len(s)), random_state=2)[['frontiers', 'uoh']].to_string())
