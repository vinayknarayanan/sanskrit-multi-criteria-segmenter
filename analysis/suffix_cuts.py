"""Suffix cuts: what share of ByT5's extra cuts fall before -tā / -tva / -vat(-mat)?

An "extra cut" is a predicted cut with no matching gold cut, i.e. it falls inside a gold piece.
Engine-free approximation: monotone DP alignment of predicted pieces to gold pieces, where
one gold piece may map to a run of 1-4 predicted pieces (over-split), or a run of 1-4 gold pieces may map
to one predicted piece (under-split). Cost = normalised edit distance between the concatenated runs.
Cuts inside an over-split run are the extra cuts. Each extra cut is labelled by the start of the piece to its right.

Caveat: a cut is compared by its underlying form, not its surface position (sandhi.py is not used here).
Suffix boundaries (t-/v-/m- initial) take no sandhi, so for these cuts the approximation is exact.
Cuts elsewhere may be misplaced by the alignment. Check any surprising number against boundary_score.py.
"""
import re, sys
from collections import Counter
from functools import lru_cache
import pandas as pd

def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def ncost(a, b):
    return lev(a, b) / max(len(a), len(b), 1)

def align(gold, pred, K=4):
    """Returns list of (gold_slice, pred_slice) groups, min total cost."""
    G, P = len(gold), len(pred)
    @lru_cache(None)
    def f(i, j):
        if i == G and j == P: return (0.0, ())
        if i == G or j == P: return (1e9, ())
        best = (1e9, ())
        for k in range(1, K + 1):  # one gold -> k preds
            if j + k <= P:
                c = ncost(gold[i], ''.join(pred[j:j + k])) + 0.05 * (k - 1)
                r = f(i + 1, j + k)
                if c + r[0] < best[0]: best = (c + r[0], (((i, i + 1), (j, j + k)),) + r[1])
        for k in range(2, K + 1):  # k golds -> one pred
            if i + k <= G:
                c = ncost(''.join(gold[i:i + k]), pred[j]) + 0.05 * (k - 1)
                r = f(i + k, j + 1)
                if c + r[0] < best[0]: best = (c + r[0], (((i, i + k), (j, j + 1)),) + r[1])
        return best
    return f(0, 0)

# Piece-initial patterns for the secondary-derivative suffixes (IAST). Order matters: first match wins.
SUFFIX = [
    ('-tva', re.compile(r'^tv(?:[aāe]|ai)')),
    ('-tā',  re.compile(r'^t[āa](?:[ḥmṃ]|yā|yāḥ|yai|yām|yāṃ|sām|bhiḥ|bhyaḥ|su)?$|^t[āa]y')),
    ('-vat/-vant', re.compile(r'^va[td]$|^vat[īi]|^vant|^vān$|^val$')),
    ('-mat/-mant', re.compile(r'^ma[td]$|^mat[īi]|^mant|^mān$')),
]

PARTICLES = {'iti','api','eva','atra','tatra','yatra','atas','tatas','ataḥ','tataḥ','ca','tu','hi','na','vā','iva','tad','tat','yad','yat','idam','evam','punar','punaḥ','tathā','yathā','ayam','asya','tasya','tasmāt','iha'}

def label(piece):
    if piece in PARTICLES: return 'particle/pronoun'
    if piece.startswith('ādi') or piece.startswith('ādy') or piece.startswith('ādī'): return 'ādi'
    for name, rx in SUFFIX:
        if rx.search(piece): return name
    return 'other'

def main(path='results_byt5.csv'):
    d = pd.read_csv(path)
    rows, per_item = [], []
    for _, r in d.iterrows():
        if not isinstance(r.predicted, str):
            per_item.append(dict(compound=r.compound, status='no_prediction')); continue
        gold = r.gold_split.split('+'); pred = r.predicted.split()
        cost, groups = align(tuple(gold), tuple(pred))
        n_extra = 0
        for (gi, gj), (pi, pj) in groups:
            if gj - gi == 1 and pj - pi > 1:
                for k in range(pi + 1, pj):
                    n_extra += 1
                    rows.append(dict(compound=r.compound, split_count=r.split_count, group=r.group,
                                     gold_piece=gold[gi], left=pred[k - 1], right=pred[k],
                                     suffix=label(pred[k]), align_cost=round(cost, 3)))
        per_item.append(dict(compound=r.compound, status='ok', align_cost=cost, n_extra=n_extra))
    cuts = pd.DataFrame(rows); items = pd.DataFrame(per_item)
    cuts.to_csv('suffix_extra_cuts.csv', index=False)

    # Gold convention: does Frontiers gold ever split before these suffixes, and how often are they kept inside a piece?
    g_split, g_internal = Counter(), Counter()
    for gs in d.gold_split:
        ps = gs.split('+')
        for k, p in enumerate(ps):
            if k > 0 and label(p) != 'other': g_split[label(p)] += 1
            for m in re.finditer(r'(?<=[aāiīuū])(tv[aā]|tā)', p):  # suffix inside a gold piece (rough)
                g_internal['-tva' if m.group(1).startswith('tv') else '-tā'] += 1

    # Conditional rate: of gold pieces containing a vowel+tva/tā sequence, how many did ByT5 cut there?
    opp = Counter(); hit = Counter()
    cut_keys = set((c.compound, c.gold_piece, c.suffix) for c in cuts.itertuples())
    for r in d.itertuples():
        if not isinstance(r.predicted, str): continue
        for p in r.gold_split.split('+'):
            for suf, rx in (('-tva', r'(?<=[aāiīuū])tv(?:[aāe]|ai)'), ('-tā', r'(?<=[aāiīuū])tā')):
                if re.search(rx, p):
                    opp[suf] += 1; hit[suf] += (r.compound, p, suf) in cut_keys
    ok = items[items.status == 'ok']
    print(f'items: {len(items)}  no prediction: {(items.status=="no_prediction").sum()}')
    print(f'alignment cost: median {ok.align_cost.median():.3f}; items with cost>1.0 (poor alignment): {(ok.align_cost>1).sum()}')
    print(f'extra cuts found: {len(cuts)} in {(ok.n_extra>0).sum()} items')
    print('\nExtra cuts by right-hand piece start:')
    vc = cuts.suffix.value_counts(); print(pd.DataFrame({'n': vc, 'share': (vc / len(cuts)).round(3)}))
    print('\nSuffix share of extra cuts by gold split_count:')
    print(cuts.assign(is_suffix=cuts.suffix != 'other').groupby('split_count').is_suffix.agg(['sum', 'count', 'mean']).round(3))
    clean = cuts[cuts.align_cost <= 1.0]
    print(f'\nRestricted to well-aligned items (cost<=1.0): {len(clean)} extra cuts, suffix share {(clean.suffix!="other").mean():.3f}')
    print('\nGold convention (Frontiers): gold pieces that START with a suffix:', dict(g_split))
    print('Gold convention: suffix occurrences INSIDE a gold piece (rough regex):', dict(g_internal))
    print('ByT5 cut rate given a suffix inside a gold piece:', {k: f"{hit[k]}/{opp[k]} = {hit[k]/opp[k]:.2f}" for k in opp})
    print('\nTop right-hand pieces among "other" extra cuts:')
    print(cuts[cuts.suffix == 'other'].right.value_counts().head(25).to_string())

if __name__ == '__main__':
    main(*sys.argv[1:])
