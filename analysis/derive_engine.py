"""Faithful port of the original sandhi.py derivation (2026-08-09),
with the rule table _rules_100.yaml (top-100 harvested rules, SLP1) and sandhi_rules_insertions.yaml.

Rules are written compactly as "LEFT+RIGHT->OUTPUT" exactly as their `name` fields in _rules_100.yaml.
All rules are obligatory (no identity alternative), as in the harvested table.
derive(): exact derivation of the surface from pieces; a rule may also apply at the END of the last piece
(allow_final=True, as in sandhi.py). Node budget as in boundary_score.py (20,000).
"""
import paths as P
from collections import defaultdict

RULES_100 = """m+->M a+a->A H+->r t+->d aH+->o i+->y +a-> aH+a->o H+-> H+->s H+->S a+i->e a+-> a+e->E a+u->o
i+i->I aH+i->E t+->n u+->v A+i->e m+->Y O+->Av t+->c +A-> m+->n A+e->E t+-> +i-> t+S->cC A+u->o k+->g n+->MS
e+i->E +a->na t+->j +a->s aH+u->O n+->Ms t+->l I+->y e+->a va+i-> t+h->dD +e->ne +a->A +A->nA AUM+->oM a+I->e
m+->N a+a->nA A+-> aH+a->os u+u->U a+->n a+->Ev a+i->ce H+->rd +i->ni i+-> M+->Y a+a->cA w+->q a+a->o H+->rm
H+->z m+-> H+->rn t+->w i+a->yA a+u->no aH+u->o H+u->rU p+->b a+i->E A+->Ev a+U->o +u->nu +C->cC H+->rv O+a->A
+i->I aH+-> t+->dy n+->Y H+a-> A+i->E A+I->e +e->E aH+->or a+u->co a+i->ne m+a-> +i->e +saH->sa +A->a +t->st
n+->S +a->ra +f->r at+->""".split()
assert len(RULES_100) == 100, len(RULES_100)

VOWEL_SHORT = list('aiufx'); VOWEL = VOWEL_SHORT + list('AIUFX') + list('eEoO')
INSERTIONS = ([('n', v, 'nn' + v) for v in VOWEL] + [('R', v, 'RR' + v) for v in VOWEL] +
              [('N', v, 'NN' + v) for v in VOWEL] + [('n', v, 'ny' + v) for v in VOWEL] +
              [('', 'C', 'cC')] + [(v, 'C', v + 'cC') for v in VOWEL_SHORT])

def build_index(use_insertions=True):
    idx = defaultdict(list)
    for r in RULES_100:
        lr, out = r.split('->'); L, R = lr.split('+')
        idx[(L, R)].append(out)
    if use_insertions:
        for L, R, out in INSERTIONS:
            idx[(L, R)].append(out)
    return idx

def derive(pieces, surface, idx, max_span=6, budget=20000):
    """True if surface is exactly derivable from pieces; None if the search budget ran out."""
    n = len(pieces)
    if n == 1:
        if pieces[0] == surface: return True
    memo = set(); b = [budget]

    def final_ok(pos, rem):
        if surface[pos:] == rem: return True
        for tail in range(1, min(max_span, len(rem)) + 1):
            head, L = rem[:len(rem) - tail], rem[len(rem) - tail:]
            for out in idx.get((L, ''), []):
                if surface[pos:] == head + out: return True
        return False

    def rec(i, pos, rem):
        if b[0] <= 0: raise TimeoutError
        b[0] -= 1
        key = (i, pos, rem)
        if key in memo: return False
        memo.add(key)
        if i == n - 1: return final_ok(pos, rem)
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
                    if surface.startswith(out, jpos) and rec(i + 1, jpos + len(out), nxt[hlen:]):
                        return True
        return False
    try:
        return rec(0, 0, pieces[0])
    except TimeoutError:
        return None


# ---------------------------------------------------------------- independent, grammar-generated rule table
def build_vidyut_index(path=P.VIDYUT_RULES):
    """vidyut 0.4.0 sandhi/rules.csv (ambuda-org/vidyut): columns first, second, result, SLP1, one right char.
    Adapted to this engine: spaces dropped (compounds are written solid); underlying final s/r also written
    as visarga H (the Frontiers gold uses H); avagraha variants with and without the apostrophe; and a word-final
    rule (L,'') wherever the rule leaves the right-hand character untouched, so an item's final syllable can take
    the sandhi of the unseen next word (as allow_final does in sandhi.py). Nothing is harvested from Frontiers."""
    import csv
    idx = defaultdict(list)
    def add(L, R, out):
        if out not in idx[(L, R)]: idx[(L, R)].append(out)
    for row in csv.DictReader(open(path, encoding='utf-8')):
        first, second, res = row['first'], row['second'], row['result']
        outs = {res.replace(' ', '')}
        if "'" in res: outs.add(res.replace(' ', '').replace("'", ''))
        lefts = {first}
        if first[-1] in 'sr': lefts.add(first[:-1] + 'H')
        for L in lefts:
            for out in outs:
                add(L, second, out)
                if out.endswith(second):
                    add(L, '', out[:-len(second)])
    return idx

def derive_trace(pieces, surface, idx, max_span=6, budget=20000):
    """Like derive(), but returns the list of (L, R, out) rule applications used, [] for plain concatenation
    everywhere, or False / None."""
    n = len(pieces)
    if n == 1 and pieces[0] == surface: return []
    memo = set(); b = [budget]
    def final_ok(pos, rem):
        if surface[pos:] == rem: return []
        for tail in range(1, min(max_span, len(rem)) + 1):
            head, L = rem[:len(rem) - tail], rem[len(rem) - tail:]
            for out in idx.get((L, ''), []):
                if surface[pos:] == head + out: return [(L, '', out)]
        return False
    def rec(i, pos, rem):
        if b[0] <= 0: raise TimeoutError
        b[0] -= 1
        key = (i, pos, rem)
        if key in memo: return False
        memo.add(key)
        if i == n - 1: return final_ok(pos, rem)
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
                    if surface.startswith(out, jpos):
                        got = rec(i + 1, jpos + len(out), nxt[hlen:])
                        if got is not False:
                            return ([(L, R, out)] if (L or R) else []) + got
        return False
    try:
        return rec(0, 0, pieces[0])
    except TimeoutError:
        return None


def build_grammar_index():
    """vidyut rules + two optional rules vidyut's table omits, each cited:
    8.4.59 (vā padāntasya): pada-final m -> nasal homorganic with the following stop;
    8.4.46 (aco rahābhyāṃ dve): a consonant after r (itself after a vowel) may be doubled."""
    idx = build_vidyut_index()
    def add(L, R, out):
        idx.setdefault((L, R), [])
        if out not in idx[(L, R)]: idx[(L, R)].append(out)
    nasal = dict(zip('kKgGcCjJwWqQtTdDpPbB', 'NNNNYYYYRRRRnnnnmmmm'))
    for c, n in nasal.items():
        add('m', c, n + c)
    for c in 'kKgGcCjJwWqQtTdDpPbBmnyvlSzsh':
        for L, R, out in [(k[0], k[1], o) for k, outs in list(idx.items()) for o in outs if k[1] == c and o.endswith('r' + c)]:
            add(L, R, out[:-1] + c + c)          # ...r + c  ->  ...r c c
    return idx
