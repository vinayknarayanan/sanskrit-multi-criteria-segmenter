"""Index DCS by written (surface) form -> how DCS divides it.

Every orthographic token in DCS is either a plain word line or a multiword range line ("3-6 surface") followed by
its members. We key each token by its surface form, lightly normalised so that a Frontiers gold piece (internal sandhi
applied, final form unsandhied) can find it: lowercase, ṃ->m, and one final ḥ/m/s/r/t/d/n/o/e/ā stripped
(final-sandhi variants of the same word then share a key). Value: Counter of member tuples (unsandhied forms),
plus the lemma tuple and Case=Cpd flags of the most frequent analysis.
Also indexes every unsandhied word form (for the reverse direction: does DCS keep A+B as one word?).
"""
import paths as P
import os, re, pickle, unicodedata
from collections import Counter, defaultdict
from multiprocessing import Pool

ROOT = P.DCS_CONLLU

def base(s):
    return unicodedata.normalize('NFC', s.strip().lower()).replace('ṃ', 'm')

def key(s):
    s = base(s)
    return re.sub(r'(?:[ḥmsrtdn]|o|e)$', '', s) if len(s) > 3 else s

def parse(path):
    idx = defaultdict(Counter); words = Counter()
    rng = None  # (end_id, surface, members)
    def close():
        nonlocal rng
        if rng:
            idx[key(rng[1])][tuple(rng[2])] += 1
        rng = None
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if not line.strip() or line[0] == '#':
                close(); continue
            p = line.rstrip('\n').split('\t')
            if len(p) < 10: continue
            if '-' in p[0]:
                close(); a, b = p[0].split('-'); rng = (int(b), p[1], []); continue
            if '.' in p[0]: continue
            misc = dict(kv.split('=', 1) for kv in p[9].split('|') if '=' in kv)
            uns = base(misc.get('Unsandhied', p[1]))
            member = (uns, p[2], 'Case=Cpd' in p[5])
            words[uns] += 1
            if rng and int(p[0]) <= rng[0]:
                rng[2].append(member)
                if int(p[0]) == rng[0]: close()
            else:
                close(); idx[key(p[1])][(member,)] += 1
    close()
    return idx, words

if __name__ == '__main__':
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(ROOT) for f in fs if f.endswith('.conllu')]
    IDX = defaultdict(Counter); W = Counter()
    with Pool(os.cpu_count()) as pool:
        for idx, w in pool.imap_unordered(parse, files, chunksize=50):
            for k, v in idx.items(): IDX[k].update(v)
            W.update(w)
    pickle.dump(dict(idx=dict(IDX), words=W), open(P.CACHE + P.CACHE + 'dcs_surface.pkl', 'wb'))
    print('surface keys', len(IDX), 'unsandhied word types', len(W))
