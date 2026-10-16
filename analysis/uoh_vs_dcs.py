"""UoH compound annotation (SaCTI corpus, Sandhan et al. 2022; components joined by '-') vs DCS compound analysis.

Both annotate compounds as sequences of members in dictionary (unsandhied) form, so the concatenated members give
a shared key and boundaries are character offsets in the same string -> directly comparable, no sandhi alignment.
DCS compound word = a run of Case=Cpd tokens + the following head token (Unsandhied forms).
"""
import paths as P
import os, re, glob, pickle, unicodedata
from collections import Counter, defaultdict
from multiprocessing import Pool
import pandas as pd
from indic_transliteration import sanscript

ROOT = P.DCS_CONLLU
def base(s): return unicodedata.normalize('NFC', s.strip().lower()).replace('ṃ', 'm')

def parse(path):
    idx = defaultdict(Counter); sent = []
    def flush():
        run = []
        for form, lemma, cpd in sent:
            run.append((form, lemma))
            if not cpd:
                if len(run) > 1: idx[base(''.join(f for f, _ in run))][tuple(run)] += 1
                run = []
        sent.clear()
    for line in open(path, encoding='utf-8'):
        if not line.strip(): flush(); continue
        if line[0] == '#': continue
        p = line.rstrip('\n').split('\t')
        if len(p) < 10 or '-' in p[0] or '.' in p[0]: continue
        misc = dict(kv.split('=', 1) for kv in p[9].split('|') if '=' in kv)
        sent.append((base(misc.get('Unsandhied', p[1])), p[2], 'Case=Cpd' in p[5]))
    flush()
    return idx

def dcs_compounds():
    cache = P.CACHE + 'dcs_compounds.pkl'
    if os.path.exists(cache): return pickle.load(open(cache, 'rb'))
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(ROOT) for f in fs if f.endswith('.conllu')]
    IDX = defaultdict(Counter)
    with Pool(os.cpu_count()) as pool:
        for idx in pool.imap_unordered(parse, files, chunksize=50):
            for k, v in idx.items(): IDX[k].update(v)
    IDX = dict(IDX); pickle.dump(IDX, open(cache, 'wb')); return IDX

def offsets(parts):
    out, acc = [], 0
    for p in parts[:-1]:
        acc += len(p); out.append(acc)
    return out

if __name__ == '__main__':
    D = dcs_compounds()
    print('DCS compound types:', len(D))
    uoh = Counter()
    for f in glob.glob(P.SACTI + '*.conll'):
        for line in open(f, encoding='utf-8'):
            p = line.split('\t')
            if len(p) > 1 and '-' in p[1]: uoh[p[1].strip()] += 1
    iast = lambda s: sanscript.transliterate(s, 'devanagari', 'iast')
    rows = []
    for dev, n in uoh.items():
        parts = [base(iast(x)) for x in dev.split('-') if x.strip()]
        if len(parts) < 2: continue
        k = base(''.join(parts)); an = D.get(k)
        r = dict(uoh=' - '.join(parts), n_uoh=len(parts))
        if not an: r['status'] = 'not in DCS'; rows.append(r); continue
        top, c = an.most_common(1)[0]
        dparts = [m[0] for m in top]
        u, dd = set(offsets(parts)), set(offsets(dparts))
        r.update(dcs=' - '.join(dparts), n_dcs=len(dparts), dcs_freq=sum(an.values()),
                 dcs_variants=len({tuple(m[0] for m in t) for t in an}))
        if u == dd: r['status'] = 'identical'
        elif dd > u: r['status'] = 'DCS divides further'
        elif u > dd: r['status'] = 'UoH divides further'
        else: r['status'] = 'different boundaries'
        # the extra boundaries, labelled by the member to their right
        r['dcs_extra'] = '; '.join(k[j:j2] for j, j2 in [(b, min([x for x in dd | {len(k)} if x > b])) for b in sorted(dd - u)])
        r['uoh_extra'] = '; '.join(k[j:j2] for j, j2 in [(b, min([x for x in u | {len(k)} if x > b])) for b in sorted(u - dd)])
        rows.append(r)
    x = pd.DataFrame(rows); x.to_csv('corpus_uoh_vs_dcs.csv', index=False)
    print('UoH compound types:', len(x), '| found in DCS:', (x.status != 'not in DCS').sum(),
          f"({(x.status != 'not in DCS').mean()*100:.1f}%)")
    m = x[x.status != 'not in DCS']
    print(m.status.value_counts().to_string())
    print((m.status.value_counts(normalize=True) * 100).round(1).to_string())
    print('\nby UoH member count (share identical):')
    print((m.groupby('n_uoh').status.apply(lambda s: (s == 'identical').mean()) * 100).round(1).to_string())
    print('\nDCS extra boundaries, right-hand member (top):', Counter(t for s in m.dcs_extra for t in s.split('; ') if t).most_common(20))
    print('UoH extra boundaries, right-hand member (top):', Counter(t for s in m.uoh_extra for t in s.split('; ') if t).most_common(20))
    for st in ['DCS divides further', 'UoH divides further', 'different boundaries']:
        print(f'\n== {st}'); print(m[m.status == st].sample(min(8, (m.status == st).sum()), random_state=1)[['uoh', 'dcs', 'dcs_freq']].to_string())
