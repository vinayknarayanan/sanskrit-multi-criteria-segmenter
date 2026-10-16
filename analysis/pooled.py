import paths as P
import io,contextlib,pandas as pd,numpy as np,pickle,re,unicodedata
with contextlib.redirect_stdout(io.StringIO()):
    from robust_ci import cuts_on_surface, POP
U=P.RUNS + ''
def counts(gold,cand,surf):
    g,c=cuts_on_surface(gold,surf),cuts_on_surface(cand,surf); used=set(); tp=0
    for x in c:
        for k,y in enumerate(g):
            if k not in used and abs(x-y)<=1: used.add(k); tp+=1; break
    return tp,len(c),len(g)
ev=pd.read_csv(P.DATA + 'shared_eval_sample.csv').drop_duplicates('compound')
sysf={'tagfr_s1':U+'byt5_tagged/results_fr.csv','tagdcs_s1':U+'byt5_tagged/results_dcs.csv','tagfr_s2':U+'byt5_tagged_s2/results_fr.csv','tagdcs_s2':U+'byt5_tagged_s2/results_dcs.csv',
 'ft3':U+'byt5_frontiers_v2/results_byt5_frontiers.csv','byt5':'results_byt5.csv','rcnn':'results_rcnn.csv'}
R=[]
for k,f in sysf.items():
    pr=dict(zip(*pd.read_csv(f)[['compound','predicted']].values.T))
    for x in ev.itertuples():
        p=pr.get(x.compound)
        if not isinstance(p,str) or not p.strip(): continue
        tp,nc,ng=counts(x.gold_split.split('+'),p.split(),x.compound)
        R.append((k,x.compound,x.split_count,tp,nc,ng))
d=pd.DataFrame(R,columns=['sys','compound','stratum','tp','nc','ng'])
keep=set.intersection(*[set(g.compound) for _,g in d.groupby('sys')]); d=d[d.compound.isin(keep)]
def f(tp,nc,ng): p=tp/nc if nc else 1; r=tp/ng if ng else 1; return 2*p*r/(p+r) if p+r else 0
def item(g): return g.apply(lambda r:f(r.tp,r.nc,r.ng),axis=1)
d['f1']=item(d); d['p']=d.apply(lambda r:r.tp/r.nc if r.nc else 1,axis=1); d['r']=d.tp/d.ng
def wmean(g,c):
    m=g.groupby('stratum')[c].mean(); return sum(m[k]*POP[k] for k in m.index)/sum(POP[k] for k in m.index)
def pooled(g,weighted):
    if weighted:
        n=g.groupby('stratum').size(); w=g.stratum.map(lambda s:POP[s]/n[s])
    else: w=pd.Series(1.0,index=g.index)
    tp,nc,ng=(g.tp*w).sum(),(g.nc*w).sum(),(g.ng*w).sum(); return tp/nc,tp/ng,f(tp,nc,ng)
print('items',len(keep))
out=[]
for k,g in d.groupby('sys'):
    pw=pooled(g,True); pu=pooled(g,False)
    out.append((k,round(wmean(g,'f1'),3),round(wmean(g,'p'),3),round(wmean(g,'r'),3),*[round(v,3) for v in pw],*[round(v,3) for v in pu]))
print(pd.DataFrame(out,columns=['sys','itemF1_w','itemP_w','itemR_w','pooledP_w','pooledR_w','pooledF1_w','pooledP_u','pooledR_u','pooledF1_u']).to_string(index=False))
# bootstrap pooled weighted gap
def bgap(a,b,reps=2000):
    A=d[d.sys==a].set_index('compound'); B=d[d.sys==b].set_index('compound').loc[A.index]
    rng=np.random.default_rng(0); idx={s:np.where(A.stratum.values==s)[0] for s in POP}
    def val(ii):
        res=[]
        for X in (A,B):
            tp=nc=ng=0
            for s,ix in ii.items():
                w=POP[s]/len(ix); tp+=w*X.tp.values[ix].sum(); nc+=w*X.nc.values[ix].sum(); ng+=w*X.ng.values[ix].sum()
            res.append(f(tp,nc,ng))
        return res[0]-res[1]
    pt=val(idx); vals=[val({s:ix[rng.integers(0,len(ix),len(ix))] for s,ix in idx.items()}) for _ in range(reps)]
    return round(pt,3),*np.round(np.percentile(vals,[2.5,97.5]),3)
for a,b in (('tagfr_s1','tagdcs_s1'),('tagfr_s2','tagdcs_s2'),('tagfr_s1','ft3'),('tagfr_s2','ft3'),('byt5','tagdcs_s1'),('byt5','tagdcs_s2'),('tagdcs_s1','rcnn')):
    print(a,'-',b,'pooled weighted F1 gap',bgap(a,b))
d.to_csv('pooled_counts_frontiers.csv',index=False)
