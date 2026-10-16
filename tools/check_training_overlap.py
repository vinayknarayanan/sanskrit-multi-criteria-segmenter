"""Leakage check (paper, Method): does any training row share a written form or a gold division with an
evaluation item? Also counts DCS rows whose division contains a '_' placeholder (a DCS token without an
unsandhied form).  Usage: python tools/check_training_overlap.py data/third_party/dataset_tagged.csv"""
import os, sys, pandas as pd
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
t = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, 'third_party', 'dataset_tagged.csv'))
t['tag'] = t.compound.str.extract(r'^<(\w+)>')[0]; t['surface'] = t.compound.str.replace(r'^<\w+> ', '', regex=True)
under = lambda s: s.str.split('+').apply(lambda p: any(x.strip() in ('_', '') for x in p))
print('training rows per tag:', t.tag.value_counts().to_dict())
for tag in t.tag.dropna().unique(): print(f'  <{tag}> rows with a "_" piece:', int(under(t[t.tag == tag].split).sum()))
for name in ('shared_eval_sample.csv', 'dcs_holdout_eval.csv'):
    e = pd.read_csv(os.path.join(D, name))
    print(f'{name}: {len(e)} items; training rows sharing the written form: {int(t.surface.isin(set(e.compound)).sum())}; '
          f'sharing the gold division: {int(t.split.isin(set(e.gold_split)).sum())}; '
          f'items whose gold division occurs in training: {int(e.gold_split.isin(set(t.split)).sum())}; '
          f'items with a "_" piece: {int(under(e.gold_split).sum())}')
