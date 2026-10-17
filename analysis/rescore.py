"""Re-score ByT5 and INRIA (top-1) against Frontiers gold: strict vs convention-aware boundary-F1.

Alignment follows the original boundary_score.py: gold and candidate are each derived onto the surface; boundaries are spans
in surface coordinates; a candidate boundary is correct if its span overlaps an unused gold span. Partial derivation
when a full one fails (unplaceable boundaries count against the candidate, as in boundary_score.py).
Rule index = grammar table (vidyut + 8.4.59 + 8.4.46) UNION harvested top-100 + insertions, so alignment does not
depend on one rule source. Every boundary is labelled by what happened at the junction:
  plain  = plain concatenation (no sound change)      sandhi = a rule changed the sound
Scoring variants for UNMATCHED candidate boundaries (recall is never changed):
  strict        every unmatched candidate boundary is a false positive (current practice)
  plain         unmatched boundaries at plain junctions are excused (dropped from the candidate count)
  plain+DCS     excused only if plain AND DCS divides there (evidence A/B, or the -tva/-tā/-vat policy)
"""
import json, pandas as pd
from collections import defaultdict
from indic_transliteration import sanscript
from derive_engine import build_index, build_grammar_index
from dcs_match import level as dcs_level
from suffix_cuts import label as cut_label

slp = lambda s: sanscript.transliterate(s, 'iast', 'slp1')
iast = lambda s: sanscript.transliterate(s, 'slp1', 'iast')
IDX = defaultdict(list)
for src in (build_grammar_index(), build_index(True)):
    for k, v in src.items():
        for o in v:
            if o not in IDX[k]: IDX[k].append(o)

def spans(pieces, surface, idx=IDX, max_span=6, budget=20000):
    """-> list of (span, kind) per internal boundary (None where unplaceable), complete flag."""
    n = len(pieces)
    if n == 1: return [], pieces[0] == surface
    memo = set(); b = [budget]; best = {'d': -1, 's': []}
    def final_ok(pos, rem):
        if surface[pos:] == rem: return True
        for tail in range(1, min(max_span, len(rem)) + 1):
            head, L = rem[:len(rem) - tail], rem[len(rem) - tail:]
            if any(surface[pos:] == head + o for o in idx.get((L, ''), [])): return True
        return False
    def rec(i, pos, rem, acc):
        if b[0] <= 0: return None
        b[0] -= 1
        key = (i, pos, rem)
        if key in memo: return None
        memo.add(key)
        if len(acc) > best['d']: best.update(d=len(acc), s=list(acc))
        if i == n - 1: return acc if final_ok(pos, rem) else None
        nxt = pieces[i + 1]
        for tail in range(0, min(max_span, len(rem)) + 1):
            emit = rem[:len(rem) - tail]
            if not surface.startswith(emit, pos): continue
            jpos = pos + len(emit); L = rem[len(rem) - tail:]
            for hlen in range(0, min(max_span, len(nxt)) + 1):
                R = nxt[:hlen]
                outs = list(idx.get((L, R), []))
                if not L and not R: outs.append('')
                for out in outs:
                    if not surface.startswith(out, jpos): continue
                    if not L and not R: sp = (jpos, jpos)
                    elif not R: sp = (jpos + len(out), jpos + len(out))
                    elif not L: sp = (jpos, jpos)
                    else: sp = (jpos, jpos + len(out))
                    kind = 'plain' if out == L + R else 'sandhi'
                    got = rec(i + 1, jpos + len(out), nxt[hlen:], acc + [(sp, kind)])
                    if got is not None: return got
        return None
    full = rec(0, 0, pieces[0], [])
    if full is not None: return full, True
    s = best['s']; return s + [None] * (n - 1 - len(s)), False

def overlaps(a, b): return a[0] <= b[1] and b[0] <= a[1]

def score(gold_pieces, cand_pieces, surface):
    gs, gok = spans(gold_pieces, surface); cs, cok = spans(cand_pieces, surface)
    used, matched = set(), []
    for c in cs:
        hit = False
        if c is not None:
            for j, g in enumerate(gs):
                if j not in used and g is not None and overlaps(c[0], g[0]):
                    used.add(j); hit = True; break
        matched.append(hit)
    # which candidate boundaries are excusable
    plain, plain_dcs = 0, 0
    for k, (c, hit) in enumerate(zip(cs, matched)):
        if hit or c is None or c[1] != 'plain': continue
        plain += 1
        l, r = iast(cand_pieces[k]), iast(cand_pieces[k + 1])
        cls = cut_label(r); cls = 'suffix' if cls.startswith('-') else ('other' if cls in ('other', 'ādi') else cls)
        if dcs_level(l, r, cls) in ('A_pair', 'B_lemma', 'S_policy'): plain_dcs += 1
    tp, nc, ng = sum(matched), len(cs), len(gs)
    def prf(ncand):
        p = tp / ncand if ncand else (1.0 if not ng else 0.0)
        r = tp / ng if ng else 1.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)
    return dict(tp=tp, n_cand=nc, n_gold=ng, gold_derived=gok, cand_derived=cok,
                gold_plain=sum(1 for g in gs if g and g[1] == 'plain'), gold_placed=sum(1 for g in gs if g),
                fp=nc - tp, fp_plain=plain, fp_plain_dcs=plain_dcs,
                **{f'{v}_{m}': x for v, (ncand) in (('strict', nc), ('plain', nc - plain), ('plaindcs', nc - plain_dcs))
                   for m, x in zip('prf', prf(ncand))})

if __name__ == '__main__':
    by = pd.read_csv('results_byt5.csv'); inr = pd.read_csv('results_inria.csv').drop_duplicates('compound')
    inr = inr.set_index('compound')
    rows = []
    for r in by.itertuples():
        surf = slp(r.compound); gold = [slp(p) for p in r.gold_split.split('+')]
        out = dict(compound=r.compound, stratum=r.split_count, group=r.group)
        if isinstance(r.predicted, str):
            out.update({f'byt5_{k}': v for k, v in score(gold, [slp(p) for p in r.predicted.split()], surf).items()})
        if r.compound in inr.index:
            c = json.loads(inr.loc[r.compound, 'candidates_json'] or '[]')
            if c: out.update({f'inria_{k}': v for k, v in score(gold, c[0], surf).items()})
        rows.append(out)
    x = pd.DataFrame(rows); x.to_csv('rescore_results.csv', index=False)
    print('items', len(x), '| byt5 scored', x.byt5_strict_f.notna().sum(), '| inria scored', x.inria_strict_f.notna().sum())
    print('gold fully derived %.1f%% | gold boundaries at plain junctions %.1f%%' % (
        x.byt5_gold_derived.mean() * 100, x.byt5_gold_plain.sum() / x.byt5_gold_placed.sum() * 100))
    for sysn in ('byt5', 'inria'):
        print(f'\n== {sysn}: false positives {int(x[sysn+"_fp"].sum())}, at plain junctions {int(x[sysn+"_fp_plain"].sum())} '
              f'({x[sysn+"_fp_plain"].sum()/x[sysn+"_fp"].sum()*100:.1f}%), plain+DCS {int(x[sysn+"_fp_plain_dcs"].sum())}')
        t = x.groupby('stratum')[[f'{sysn}_{v}_{m}' for v in ('strict', 'plain', 'plaindcs') for m in 'prf']].mean().round(3)
        t.loc['mean (unweighted)'] = x[[f'{sysn}_{v}_{m}' for v in ('strict', 'plain', 'plaindcs') for m in 'prf']].mean().round(3)
        print(t.to_string())
    both = x.dropna(subset=['byt5_strict_f', 'inria_strict_f'])
    print(f'\nboth scored: {len(both)}')
    for v in ('strict', 'plain', 'plaindcs'):
        print(f'  {v:9s} F1 byt5 {both[f"byt5_{v}_f"].mean():.3f}  inria {both[f"inria_{v}_f"].mean():.3f}  gap {both[f"byt5_{v}_f"].mean()-both[f"inria_{v}_f"].mean():+.3f}'
              f'  | byt5 P {both[f"byt5_{v}_p"].mean():.3f} R {both[f"byt5_{v}_r"].mean():.3f}')
