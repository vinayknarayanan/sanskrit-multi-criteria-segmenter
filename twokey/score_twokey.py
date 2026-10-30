"""Score the two-key run: both tags against both real keys, per group (agree / disagree). Form-agnostic boundary F1,
unweighted mean over items. Run: python score_twokey.py ../runs/twokey_s13"""
import sys, os, difflib, unicodedata, random, pandas as pd
nf = lambda s: unicodedata.normalize('NFC', str(s).strip())
def cuts(pieces, surf):
    c = ''.join(pieces); offs, acc = [], 0
    for p in pieces[:-1]: acc += len(p); offs.append(acc)
    ops = difflib.SequenceMatcher(None, c, surf, autojunk=False).get_opcodes(); out = []
    for i in offs:
        for t, i1, i2, j1, j2 in ops:
            if i1 <= i <= i2 and (i < i2 or t == 'equal' or i2 == len(c)):
                out.append(j1 + (i - i1) if t == 'equal' else j1 + round((i - i1) * (j2 - j1) / max(i2 - i1, 1))); break
    return out
def match(a, b):
    used, tp = set(), 0
    for x in a:
        for k, y in enumerate(b):
            if k not in used and abs(x - y) <= 1: used.add(k); tp += 1; break
    return tp
def f1(c, g):
    tp = match(c, g); p = tp / len(c) if c else 1.0; r = tp / len(g) if g else 1.0
    return 2 * p * r / (p + r) if p + r else 0.0
R = sys.argv[1] if len(sys.argv) > 1 else '../runs/twokey_s13'
a = pd.read_csv(os.path.join(R, 'results_fr.csv')); b = pd.read_csv(os.path.join(R, 'results_dcs.csv'))
m = a[['compound', 'gold_split', 'dcs_split', 'group', 'predicted']].merge(b[['compound', 'predicted']], on='compound', suffixes=('_fr', '_dcs'))
rows = []
for x in m.itertuples():
    s = nf(x.compound); F = cuts([nf(p) for p in x.gold_split.split('+')], s); D = cuts([nf(p) for p in x.dcs_split.split('+')], s)
    pf = cuts(str(x.predicted_fr).split(), s); pd_ = cuts(str(x.predicted_dcs).split(), s)
    same = match(pf, pd_) == len(pf) == len(pd_)
    add = sum(1 for y in pd_ if not any(abs(y - z) <= 1 for z in pf)); rem = sum(1 for y in pf if not any(abs(y - z) <= 1 for z in pd_))
    rows.append(dict(group=x.group, fr_F=f1(pf, F), fr_D=f1(pf, D), dcs_F=f1(pd_, F), dcs_D=f1(pd_, D), same_cuts=same,
                     same_text=str(x.predicted_fr) == str(x.predicted_dcs), dcs_adds=add > 0 and rem == 0, dcs_removes=rem > 0 and add == 0, mixed=add > 0 and rem > 0))
S = pd.DataFrame(rows)
def boot(d, col_a, col_b, n=2000):
    rng = random.Random(0); v = (d[col_a] - d[col_b]).tolist(); k = len(v); bs = sorted(sum(v[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return sum(v) / k, bs[int(.025 * n)], bs[int(.975 * n)]
for g, d in S.groupby('group'):
    print(f'\n== {g} (n={len(d)})')
    print(f'  vs Frontiers key:  <fr> {d.fr_F.mean():.3f}   <dcs> {d.dcs_F.mean():.3f}')
    print(f'  vs DCS key:        <fr> {d.fr_D.mean():.3f}   <dcs> {d.dcs_D.mean():.3f}')
    m1, l1, h1 = boot(d, 'fr_F', 'dcs_F'); m2, l2, h2 = boot(d, 'dcs_D', 'fr_D')
    print(f'  tag gap on Frontiers key (<fr> - <dcs>): {m1:+.3f} [{l1:+.3f}, {h1:+.3f}]')
    print(f'  tag gap on DCS key (<dcs> - <fr>):       {m2:+.3f} [{l2:+.3f}, {h2:+.3f}]')
    print(f'  same cuts under both tags: {d.same_cuts.mean():.1%}   identical text: {d.same_text.mean():.1%}')
    print(f'  where cuts differ: <dcs> only adds {d.dcs_adds.mean():.1%}, only removes {d.dcs_removes.mean():.1%}, both {d.mixed.mean():.1%}')
dis, agr = S[S.group == 'disagree'], S[S.group == 'agree']
g1 = (dis.fr_F - dis.dcs_F).mean(); g2 = (dis.dcs_D - dis.fr_D).mean()
ga = max(abs((agr.fr_F - agr.dcs_F).mean()), abs((agr.dcs_D - agr.fr_D).mean()))
print('\n== Pre-registered predictions (PLAN.txt)')
print(f'  P1 disagree gap >= 0.15 both directions: {g1:.3f} and {g2:.3f} -> {"PASS" if g1 >= .15 and g2 >= .15 else "FAIL"}')
print(f'  P2 agree: same cuts >= 90% ({agr.same_cuts.mean():.1%}) and gap < 0.05 ({ga:.3f}) -> {"PASS" if agr.same_cuts.mean() >= .9 and ga < .05 else "FAIL"}')
S.to_csv(os.path.join(R, 'twokey_item_scores.csv'), index=False)
