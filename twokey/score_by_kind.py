"""Split the 596 disagreeing two-key forms by kind (DCS deeper / Frontiers deeper / same number of cuts at different
positions) and score both tag settings against both keys. Run: python score_by_kind.py"""
import pandas as pd, sys, difflib, unicodedata, os
H = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(H, 'score_twokey.py')).read().split("R = sys.argv")[0])
ev = pd.read_csv(os.path.join(H, 'twokey_eval.csv'))
def kind(r):
    s = nf(r.compound); F = cuts([nf(p) for p in r.gold_split.split('+')], s); D = cuts([nf(p) for p in r.dcs_split.split('+')], s)
    return 'dcs_deeper' if len(D) > len(F) else 'fr_deeper' if len(F) > len(D) else 'same_count_diff_pos'
ev['kind'] = [kind(r) if g == 'disagree' else 'agree' for r, g in zip(ev.itertuples(), ev.group)]
for seed in ('13', '42'):
    S = pd.read_csv(os.path.join(H, '..', 'predictions', f'twokey_s{seed}', 'twokey_item_scores.csv')); S['kind'] = ev.kind.values
    for k, g in S[S.kind != 'agree'].groupby('kind'):
        print(seed, k, len(g), 'Frontiers key: <fr> %.3f <dcs> %.3f | DCS key: <fr> %.3f <dcs> %.3f' % (g.fr_F.mean(), g.dcs_F.mean(), g.fr_D.mean(), g.dcs_D.mean()))
