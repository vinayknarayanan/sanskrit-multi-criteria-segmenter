"""Build the two-key test (Section 5.1): written forms that both Frontiers and DCS annotate, split into those the two
datasets divide at the same positions ("agree") and differently ("disagree"); hold them out of the tagged training
data. Also writes the shuffled-tag control. Run from results/ after build_dcs_pairs.py:
    python ../twokey/make_twokey.py
Inputs: ../data/third_party/frontiers_dataset.csv, ../data/third_party/dataset_tagged.csv, dcs_pairs_all.csv
Outputs (in ../twokey/): twokey_eval.csv, dataset_tagged_twokey.csv, dataset_shuffled_tags.csv"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from twokey_lib import nf, cuts_on_surface, match
F = pd.read_csv('../data/third_party/frontiers_dataset.csv').dropna()
F['c'] = F.compound.map(nf); F['s'] = F.split.map(lambda x: '+'.join(nf(p) for p in x.split('+')))
amb = F.drop_duplicates(['c', 's']).groupby('c').s.nunique(); amb = set(amb[amb > 1].index)   # forms with two divisions
Fu = F[~F.c.isin(amb)].drop_duplicates('c')[['c', 's']]
D = pd.read_csv('dcs_pairs_all.csv'); D['c'] = D.compound.map(nf)                             # DCS most frequent analysis
S = Fu.merge(D[['c', 'split']].rename(columns={'split': 'd'}), on='c')
S['fp'] = S.s.str.split('+'); S['dp'] = S.d.str.split('+')
S['fc'] = [cuts_on_surface(a, c) for a, c in zip(S.fp, S.c)]; S['dc'] = [cuts_on_surface(a, c) for a, c in zip(S.dp, S.c)]
S['same'] = [match(a, b) == len(a) == len(b) for a, b in zip(S.fc, S.dc)]
S['nf'] = S.fp.str.len()
dis, agr = S[~S.same], S[S.same]
print('shared forms', len(S), 'agree', len(agr), 'disagree', len(dis))
parts = []
for k, g in dis.groupby('nf'):                       # agree forms matched to the disagree forms by piece count
    pool = agr[agr.nf == k]; parts.append(pool.sample(min(len(g), len(pool)), random_state=0))
ag = pd.concat(parts)
if len(ag) < len(dis): ag = pd.concat([ag, agr.drop(ag.index).sample(len(dis) - len(ag), random_state=0)])
test = pd.concat([dis.assign(group='disagree'), ag.assign(group='agree')])
test = pd.DataFrame({'compound': test.c, 'gold_split': test.s, 'dcs_split': test.d, 'split_count': test.nf, 'group': test.group})
T = pd.read_csv('../data/third_party/dataset_tagged.csv')
w = T.compound.str.split(' ', n=1); c = w.str[1].map(nf); s = T.split.map(lambda x: '+'.join(nf(p) for p in str(x).split('+')))
keep = ~c.isin(set(test.compound)) & ~s.isin(set(test.gold_split) | set(test.dcs_split))
print('training rows removed', (~keep).sum(), 'kept', keep.sum())
out = os.path.join(HERE, '')
test.to_csv(out + 'twokey_eval.csv', index=False)
T[keep][['compound', 'split']].to_csv(out + 'dataset_tagged_twokey.csv', index=False)
d = T[keep][['compound', 'split']].reset_index(drop=True); body = d.compound.str.split(' ', n=1).str[1]
tag = np.where(np.random.default_rng(0).random(len(d)) < 0.5, '<fr>', '<dcs>')
pd.DataFrame({'compound': [t + ' ' + b for t, b in zip(tag, body)], 'split': d.split}).to_csv(out + 'dataset_shuffled_tags.csv', index=False)
