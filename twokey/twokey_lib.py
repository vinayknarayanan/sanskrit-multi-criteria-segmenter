import difflib, unicodedata
nf = lambda s: unicodedata.normalize('NFC', str(s).strip())
def cuts_on_surface(pieces, surf):
    c = ''.join(pieces); offs, acc = [], 0
    for p in pieces[:-1]: acc += len(p); offs.append(acc)
    ops = difflib.SequenceMatcher(None, c, surf, autojunk=False).get_opcodes()
    out = []
    for i in offs:
        for tag, i1, i2, j1, j2 in ops:
            if i1 <= i <= i2 and (i < i2 or tag == 'equal' or i2 == len(c)):
                out.append(j1 + (i - i1) if tag == 'equal' else j1 + round((i - i1) * (j2 - j1) / max(i2 - i1, 1))); break
    return out
def match(a, b, tol=1):
    used, tp = set(), 0
    for x in a:
        for k, y in enumerate(b):
            if k not in used and abs(x - y) <= tol: used.add(k); tp += 1; break
    return tp
def f1(tp, nc, ng):
    p = tp / nc if nc else 1.0; r = tp / ng if ng else 1.0
    return 2 * p * r / (p + r) if p + r else 0.0
def agn(gold, cand, surf):
    g, c = cuts_on_surface(gold, surf), cuts_on_surface(cand, surf)
    return f1(match(c, g), len(c), len(g))
