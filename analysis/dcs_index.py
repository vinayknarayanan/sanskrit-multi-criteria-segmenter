"""Build a DCS adjacency index for the ByT5 extra cuts.

For every sentence in DCS CoNLL-U: take the word-level tokens (skip multiword range lines), normalise each
token's Unsandhied form, and record
  - form -> lemma counts (only for forms we need),
  - adjacent (lemmaId, lemmaId) pairs, split by whether the left token is a compound member (Case=Cpd),
  - adjacent exact (form, form) pairs for the cut pairs we need.
Output: dcs_index.pkl
"""
import paths as P
import os, re, sys, pickle, unicodedata
from collections import Counter, defaultdict
from multiprocessing import Pool
import pandas as pd

ROOT = P.DCS_CONLLU

def norm(s):
    s = unicodedata.normalize('NFC', s.strip().lower()).replace('ṃ', 'm')
    s = re.sub(r's$', 'ḥ', s); s = re.sub(r'r$', 'ḥ', s); s = re.sub(r'd$', 't', s)
    return s

cuts = pd.read_csv('kt4_extra_cuts.csv')
NEED = set(map(norm, cuts.left)) | set(map(norm, cuts.right))
PAIRS = set(zip(map(norm, cuts.left), map(norm, cuts.right)))

def parse(path):
    f2l = defaultdict(Counter); lemname = {}; adj_cpd = Counter(); adj_word = Counter(); fpairs = Counter()
    ntok = 0
    sent = []
    def flush():
        for (f1, l1, c1), (f2, l2, _) in zip(sent, sent[1:]):
            (adj_cpd if c1 else adj_word)[(l1, l2)] += 1
            if (f1, f2) in PAIRS: fpairs[(f1, f2, c1)] += 1
        sent.clear()
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():
                flush(); continue
            if line[0] == '#': continue
            p = line.rstrip('\n').split('\t')
            if len(p) < 10 or '-' in p[0] or '.' in p[0]: continue
            misc = dict(kv.split('=', 1) for kv in p[9].split('|') if '=' in kv)
            form = norm(misc.get('Unsandhied', p[1])); lid = misc.get('LemmaId', p[2])
            lemname[lid] = p[2]; ntok += 1
            if form in NEED: f2l[form][lid] += 1
            sent.append((form, lid, 'Case=Cpd' in p[5]))
    flush()
    return f2l, lemname, adj_cpd, adj_word, fpairs, ntok

if __name__ == '__main__':
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(ROOT) for f in fs if f.endswith('.conllu')]
    F2L = defaultdict(Counter); LN = {}; AC = Counter(); AW = Counter(); FP = Counter(); N = 0
    with Pool(os.cpu_count()) as pool:
        for i, (f2l, ln, ac, aw, fp, n) in enumerate(pool.imap_unordered(parse, files, chunksize=50)):
            for k, v in f2l.items(): F2L[k].update(v)
            LN.update(ln); AC.update(ac); AW.update(aw); FP.update(fp); N += n
            if i % 5000 == 0: print(i, len(files), flush=True)
    pickle.dump(dict(f2l=dict(F2L), lemname=LN, adj_cpd=AC, adj_word=AW, fpairs=FP, ntok=N, nfiles=len(files)),
                open(P.CACHE + P.CACHE + 'dcs_index.pkl', 'wb'))
    print('files', len(files), 'tokens', N, 'lemma pairs cpd', len(AC), 'word', len(AW), 'needed forms found', len(F2L), '/', len(NEED))
