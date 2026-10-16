"""Frontiers eval items rewritten into DCS convention piece by piece (DCS's most frequent analysis of each Frontiers piece).
Items kept only if every piece is found in DCS (or is a short word). Score both tag settings against both golds."""
import paths as P
import io, contextlib, pandas as pd, numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    from robust_ci import agnostic, POP
U=P.RUNS + ''
f=pd.read_csv('corpus_frontiers_vs_dcs_forward.csv')
ev=pd.read_csv(P.DATA + 'shared_eval_sample.csv').drop_duplicates('compound').set_index('compound')
bad=set(f[f.status=='not in DCS'].compound)
conv={}
for c,g in f.groupby('compound', sort=False):
    if c in bad: continue
    pieces=[]
    for r in g.itertuples():
        pieces += [p.strip() for p in r.dcs.split('+')] if r.status=='DCS divides, Frontiers does not' else [r.piece]
    conv[c]=pieces
PRED={}
for s in ('byt5_tagged','byt5_tagged_s2'):
    for t in ('fr','dcs'):
        r=pd.read_csv(U+f'{s}/results_{t}.csv'); PRED[(s[-2:] if s.endswith('s2') else 's1',t)]=dict(zip(r.compound,r.predicted))
rows=[]
for c,dg in conv.items():
    fg=ev.loc[c,'gold_split'].split('+'); st=ev.loc[c,'split_count']
    for k,pr in PRED.items():
        p=pr.get(c)
        if not isinstance(p,str) or not p.strip(): continue
        cand=p.split()
        rows.append(dict(compound=c,stratum=st,seed=k[0],tag=k[1],differs=(len(dg)!=len(fg)),
          vs_fr=agnostic(fg,cand,c),vs_dcs=agnostic(dg,cand,c),ex_fr=float(len(cand)==len(fg) and agnostic(fg,cand,c)==1),
          ex_dcs=float(len(cand)==len(dg) and agnostic(dg,cand,c)==1)))
d=pd.DataFrame(rows); d.to_csv('converted_gold_scores.csv',index=False)
def wm(x,col):
    g=x.groupby('stratum')[col].mean(); w={k:v for k,v in POP.items() if k in g.index}; return sum(g[k]*w[k] for k in w)/sum(w.values())
def boot(x,col,reps=2000):
    # paired: tag dcs - tag fr on same items, stratified
    a=x[x.tag=='dcs'].set_index('compound')[col]; b=x[x.tag=='fr'].set_index('compound')[col]
    m=pd.DataFrame({'a':a,'b':b}).dropna().join(x.drop_duplicates('compound').set_index('compound').stratum)
    rng=np.random.default_rng(0); G={k:g for k,g in m.groupby('stratum')}; vals=[]
    for _ in range(reps):
        t=ws=0
        for k,g in G.items():
            i=rng.integers(0,len(g),len(g)); t+=POP[k]*(g.a.values[i].mean()-g.b.values[i].mean()); ws+=POP[k]
        vals.append(t/ws)
    return np.percentile(vals,[2.5,97.5])
print('items converted:',len(conv),'of',ev.shape[0],'| DCS conversion changes piece count on',d.drop_duplicates('compound').differs.mean().round(3))
for sub,lab in ((d,'ALL converted items'),(d[d.differs],'items where the two golds differ')):
    print(f'\n== {lab}: n={sub.compound.nunique()}')
    for sd in ('s1','s2'):
        x=sub[sub.seed==sd]
        for t in ('fr','dcs'):
            y=x[x.tag==t]; print(f'  {sd} <{t}>  vs Fr gold {wm(y,"vs_fr"):.3f} ex {wm(y,"ex_fr"):.3f} | vs DCS-converted gold {wm(y,"vs_dcs"):.3f} ex {wm(y,"ex_dcs"):.3f}')
        lo,hi=boot(x,'vs_dcs'); print(f'    <dcs>-<fr> on DCS-converted gold: {wm(x[x.tag=="dcs"],"vs_dcs")-wm(x[x.tag=="fr"],"vs_dcs"):+.3f} [{lo:+.3f},{hi:+.3f}]')

# ---- other systems on the same converted items (same-convention comparison) ----
O={'byt5_sanskrit':'results_byt5.csv','ft_frontiers':U+'byt5_frontiers_v2/results_byt5_frontiers.csv','rcnn':'results_rcnn.csv'}
R=[]
for name,path in O.items():
    r=pd.read_csv(path); pr=dict(zip(r.compound,r.predicted))
    for c,dg in conv.items():
        p=pr.get(c)
        if not isinstance(p,str) or not p.strip(): continue
        fg=ev.loc[c,'gold_split'].split('+'); cand=p.split()
        R.append(dict(compound=c,stratum=ev.loc[c,'split_count'],sys=name,differs=len(dg)!=len(fg),
            vs_fr=agnostic(fg,cand,c),vs_dcs=agnostic(dg,cand,c)))
R=pd.DataFrame(R)
mine=d[d.seed=='s1'].assign(sys=lambda x:'tag_'+x.tag)[['compound','stratum','sys','differs','vs_fr','vs_dcs']]
A=pd.concat([R,mine]); A.to_csv('converted_gold_systems.csv',index=False)
for lab,sub in (('all converted',A),('golds differ',A[A.differs])):
    print('\n==',lab)
    for s,g in sub.groupby('sys'): print(f'  {s:14s} n={g.compound.nunique():4d} vs Fr {wm(g,"vs_fr"):.3f}  vs DCS-conv {wm(g,"vs_dcs"):.3f}')
def bootpair(x,s1,s2,col,reps=2000):
    a=x[x.sys==s1].set_index('compound')[col]; b=x[x.sys==s2].set_index('compound')[col]
    m=pd.DataFrame({'a':a,'b':b}).dropna().join(x.drop_duplicates('compound').set_index('compound').stratum)
    rng=np.random.default_rng(0); G={k:g for k,g in m.groupby('stratum')}; vals=[]
    pt=sum(POP[k]*(g.a.mean()-g.b.mean()) for k,g in G.items())/sum(POP[k] for k in G)
    for _ in range(reps):
        t=ws=0
        for k,g in G.items():
            i=rng.integers(0,len(g),len(g)); t+=POP[k]*(g.a.values[i].mean()-g.b.values[i].mean()); ws+=POP[k]
        vals.append(t/ws)
    return pt,*np.percentile(vals,[2.5,97.5]),len(m)
for lab,sub in (('all converted',A),('golds differ',A[A.differs])):
    print(lab,'ByT5-Sanskrit - <dcs> on DCS-converted gold: %+.3f [%+.3f, %+.3f] n=%d'%bootpair(sub,'byt5_sanskrit','tag_dcs','vs_dcs'))
    print(lab,'ByT5-Sanskrit - <dcs> on Fr gold: %+.3f [%+.3f, %+.3f] n=%d'%bootpair(sub,'byt5_sanskrit','tag_dcs','vs_fr'))

# ---- unweighted (as reported in the paper: the 293 items are a targeted diagnostic subset, not a population sample) ----
def bt_unw(a, b, reps=2000):
    m = pd.DataFrame({'a': a, 'b': b}).dropna(); x = (m.a - m.b).values; rng = np.random.default_rng(0)
    v = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(reps)]
    return x.mean(), *np.percentile(v, [2.5, 97.5]), len(x)
print('\n== UNWEIGHTED, items whose key changes')
sub = d[d.differs]
print(sub.groupby(['seed', 'tag'])[['vs_fr', 'vs_dcs', 'ex_fr', 'ex_dcs']].mean().round(3))
for sd in ('s1', 's2'):
    x = sub[sub.seed == sd]; A = x[x.tag == 'dcs'].set_index('compound'); B = x[x.tag == 'fr'].set_index('compound')
    print(sd, 'DCS-style key, <dcs>-<fr>: %+.3f [%+.3f, %+.3f] n=%d' % bt_unw(A.vs_dcs, B.vs_dcs),
          '| Frontiers key, <fr>-<dcs>: %+.3f [%+.3f, %+.3f] n=%d' % bt_unw(B.vs_fr, A.vs_fr))
S_ = pd.read_csv('converted_gold_systems.csv')
for s_, lab in ((S_[S_.differs], '293'), (S_, 'all rewritten')):
    print(lab, s_.groupby('sys')[['vs_fr', 'vs_dcs']].mean().round(3).to_dict('index'))
    a = s_[s_.sys == 'byt5_sanskrit'].set_index('compound'); b = s_[s_.sys == 'tag_dcs'].set_index('compound')
    print('  ByT5-Sanskrit - <dcs>, DCS-style key: %+.3f [%+.3f, %+.3f] n=%d' % bt_unw(a.vs_dcs, b.vs_dcs))
