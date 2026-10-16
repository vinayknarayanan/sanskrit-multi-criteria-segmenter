"""DCS-convention training pairs in Frontiers format: orthographic token (range line) -> unsandhied members joined by '+'."""
import paths as P
import os, unicodedata, random, pandas as pd
from collections import Counter
from multiprocessing import Pool
ROOT = P.DCS_CONLLU
nf = lambda s: unicodedata.normalize('NFC', s.strip())
def parse(path):
    out = Counter(); rng = None
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if not line.strip() or line[0] == '#': rng = None; continue
            p = line.rstrip('\n').split('\t')
            if len(p) < 10: continue
            if '-' in p[0]:
                a, b = p[0].split('-'); rng = [int(b), nf(p[1]), []]; continue
            if '.' in p[0] or not rng: continue
            misc = dict(kv.split('=', 1) for kv in p[9].split('|') if '=' in kv)
            rng[2].append(nf(misc.get('Unsandhied', p[1])))
            if int(p[0]) == rng[0]:
                if len(rng[2]) >= 2 and ' ' not in rng[1] and all(rng[2]): out[(rng[1], '+'.join(rng[2]))] += 1
                rng = None
    return out
if __name__ == '__main__':
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(ROOT) for f in fs if f.endswith('.conllu')]
    C = Counter()
    with Pool(os.cpu_count()) as pool:
        for c in pool.imap_unordered(parse, files, chunksize=50): C.update(c)
    # one split per surface: the most frequent analysis
    best = {}
    for (s, sp), n in C.most_common():
        if s not in best: best[s] = sp
    d = pd.DataFrame(best.items(), columns=['compound', 'split'])
    d = d[d.compound.str.len().between(4, 120)]
    print('DCS multi-member tokens (unique surfaces):', len(d))
    print(d.split.str.count(r'\+').add(1).value_counts().sort_index().head(8).to_string())
    print(d.sample(8, random_state=1).to_string())
    d.to_csv('dcs_pairs_all.csv', index=False)
