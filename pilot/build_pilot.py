"""Paper 2, H2c pilot: do boundaries where conventions disagree coincide with boundaries humans find contested?

Unit = one junction (cut point) in one written word; one junction per word.
Groups (hidden from annotators):
  D  convention disagreement: the <fr> and <dcs> settings of the same tagged model disagree here (one cuts, the other not)
  A  agreed cut: both settings cut here
  C  agreed no-cut: neither setting cuts, but the Heritage segmenter (parse mode, top answer) does -> a plausible
     boundary both conventions reject (control for 'no' answers)
Sources: Frontiers evaluation words (3,259) and held-out DCS words (3,000); C uses Frontiers words only (Heritage ran there).
v2: D 90 (nominal prefix/negation 25, compound 40, between words 25), A 60 (compound 35, between words 25), C 40, + 10 repeats = 200 rows.
Diversity: at most 2 junctions with the same piece on either side (left piece: 6 for prefix/negation junctions). Words over 40 characters excluded (readability).
Validity filter (all groups): both pieces at the junction must be attested DCS word forms (>=3 occurrences, ṃ->m),
so the sample tests convention choices between real words, not model or Heritage errors."""
import io, os, contextlib, json, pickle, random, sys, unicodedata, pandas as pd
from collections import Counter
from indic_transliteration import sanscript
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'analysis')); import paths as PATHS
with contextlib.redirect_stdout(io.StringIO()):
    from robust_ci import cuts_on_surface
    from kt4_suffix_cuts import label
SEED = 20261015; rng = random.Random(SEED)
U = PATHS.ROOT + '/'   # expects predictions/ and results/heritage_modes.csv
W = pickle.load(open(PATHS.CACHE + 'dcs_surface.pkl', 'rb'))['words']
bn = lambda s: unicodedata.normalize('NFC', s.strip().lower()).replace('ṃ', 'm')
real = lambda x: W.get(bn(x), 0) >= 3
deva = lambda s: sanscript.transliterate(s, 'iast', 'devanagari')

def junctions(pieces, surf):
    cs = cuts_on_surface(pieces, surf)
    return [(c, pieces[i], pieces[i + 1]) for i, c in enumerate(cs) if 0 < c < len(surf)]

def near(c, lst): return any(abs(c - x) <= 1 for x, _, _ in lst)

cands = []
for src, fr_f, dcs_f in (('frontiers', PATHS.RUNS + 'byt5_tagged/results_fr.csv', PATHS.RUNS + 'byt5_tagged/results_dcs.csv'),
                         ('dcs', PATHS.RUNS + 'tagged_on_dcs/results_fr.csv', PATHS.RUNS + 'tagged_on_dcs/results_dcs.csv'),
                         ('dcs_frequent', PATHS.RUNS + 'tagged_on_dcsfreq/results_fr.csv', PATHS.RUNS + 'tagged_on_dcsfreq/results_dcs.csv')):
    fr = pd.read_csv(fr_f).drop_duplicates('compound').set_index('compound').predicted
    dc = pd.read_csv(dcs_f).drop_duplicates('compound').set_index('compound').predicted
    for w in fr.index.intersection(dc.index):
        if not isinstance(fr[w], str) or not isinstance(dc[w], str) or len(w) > 40: continue
        jf, jd = junctions(fr[w].split(), w), junctions(dc[w].split(), w)
        for c, l, r in jf:
            cands.append(dict(src=src, word=w, cut=c, left=l, right=r, group='A' if near(c, jd) else 'D',
                              cut_by='both' if near(c, jd) else 'fr_only', fr=fr[w], dcs=dc[w]))
        for c, l, r in jd:
            if not near(c, jf): cands.append(dict(src=src, word=w, cut=c, left=l, right=r, group='D', cut_by='dcs_only', fr=fr[w], dcs=dc[w]))
h = pd.read_csv(PATHS.ROOT + '/results/heritage_modes.csv'); h = h[h['mode'] == 'p']
frp = pd.read_csv(PATHS.RUNS + 'byt5_tagged/results_fr.csv').drop_duplicates('compound').set_index('compound').predicted
dcp = pd.read_csv(PATHS.RUNS + 'byt5_tagged/results_dcs.csv').drop_duplicates('compound').set_index('compound').predicted
for x in h.itertuples():
    sols = json.loads(x.candidates_json or '[]'); w = x.compound
    if not sols or len(w) > 40 or w not in frp.index or not isinstance(frp[w], str) or not isinstance(dcp[w], str): continue
    jf, jd = junctions(frp[w].split(), w), junctions(dcp[w].split(), w)
    for c, l, r in junctions(sols[0], w):
        if not near(c, jf) and not near(c, jd) and len(l) > 1 and len(r) > 1:
            cands.append(dict(src='frontiers', word=w, cut=c, left=l, right=r, group='C', cut_by='heritage_only', fr=frp[w], dcs=dcp[w]))
C = pd.DataFrame(cands)
C = C[C.left.map(real) & C.right.map(real) & ~C.word.str.startswith("'")]   # avagraha-initial = fragment
# junction type from DCS: how often the LEFT piece occurs as a compound member (Case=Cpd)
CS = pickle.load(open(PATHS.CACHE + 'cpd_share.pkl', 'rb'))
def jtype(l):
    v = CS.get(bn(l))
    return 'unknown' if v is None else 'compound' if v >= .8 else 'between_words' if v <= .2 else 'mixed'
PRE = {'a', 'an', 'su', 'dur', 'dus', 'ku', 'sa', 'ati'}   # nominal prefixes/negation only (verb preverbs excluded: neither model convention disagrees on them)
C['junction'] = [('prefix_negation' if bn(l) in PRE else jtype(l)) for l in C.left]
print('candidate junctions:', C.groupby(['group', 'src', 'cut_by']).size().to_dict(), file=sys.stderr)

used_words, side = set(), Counter()
def take(df, n, lcap=2):
    out = []
    for r in df.sample(frac=1, random_state=rng.randint(0, 10**6)).itertuples():
        if r.word in used_words or side[('L', r.left)] >= lcap or side[('R', r.right)] >= 2: continue
        out.append(r._asdict()); used_words.add(r.word); side[('L', r.left)] += 1; side[('R', r.right)] += 1
        if len(out) == n: break
    assert len(out) == n, (len(out), n)
    return out
picked = []
# v2 design (after the corpus-proxy result): three junction types, disagreement (D) vs agreement (A) within each
QUOTA = {('D', 'prefix_negation'): 25, ('D', 'compound'): 40, ('D', 'between_words'): 25,
         ('A', 'compound'): 35, ('A', 'between_words'): 25}
for (g, jt), n in QUOTA.items():
    picked += take(C[(C.group == g) & (C.junction == jt)], n, lcap=6 if jt == 'prefix_negation' else 2)  # conventions disagree on few prefixes (a-, an-, su-, sa-, ati-)
picked += take(C[C.group == 'C'], 40)
# 190 items + 10 repeats = 200 rows
P = pd.DataFrame(picked).drop(columns='Index')
P['kind'] = P.right.map(lambda r: 'suffix' if label(r).startswith('-') else 'particle' if label(r) == 'particle/pronoun' else 'other')
codes = rng.sample(range(1000, 9999), len(P) + 10)
P['id'] = [f'J{c}' for c in codes[:len(P)]]; P['repeat_of'] = ''
rep = P.sample(10, random_state=SEED).copy(); rep['repeat_of'] = rep.id; rep['id'] = [f'J{c}' for c in codes[len(P):]]
P = pd.concat([P, rep], ignore_index=True)
P['marked_iast'] = [w[:c] + ' | ' + w[c:] for w, c in zip(P.word, P.cut)]
P['marked_deva'] = [deva(w[:c]) + ' | ' + deva(w[c:]) for w, c in zip(P.word, P.cut)]
P['pieces_iast'] = P.left + ' + ' + P.right
P['pieces_deva'] = P.left.map(deva) + ' + ' + P.right.map(deva)
P['word_deva'] = P.word.map(deva)
P.to_csv('pilot/pilot_key_DO_NOT_SEND.csv', index=False)
print(P.groupby(['group', 'src', 'cut_by']).size().to_string()); print(P[P.repeat_of == ''].groupby(['group', 'junction']).size().to_string()); print(P[P.repeat_of == ''].groupby(['group', 'src']).size().to_string()); print('kind:', P[P.repeat_of == ''].groupby('group').kind.value_counts().to_dict())
