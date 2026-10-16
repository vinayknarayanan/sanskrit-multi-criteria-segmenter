import paths as P
import pandas as pd,numpy as np,io,contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from pooled import counts,f,POP
U=P.RUNS + ''
hv=pd.read_csv(P.DATA + 'dcs_holdout_eval.csv')
R=[]
for k,f_ in {'fr_s1':'tagged_on_dcs/results_fr.csv','dcs_s1':'tagged_on_dcs/results_dcs.csv','fr_s2':'tagged_s2_on_dcs/results_fr.csv','dcs_s2':'tagged_s2_on_dcs/results_dcs.csv'}.items():
    pr=dict(zip(*pd.read_csv(U+f_)[['compound','predicted']].values.T))
    for x in hv.itertuples():
        p=pr.get(x.compound)
        if not isinstance(p,str) or not p.strip(): continue
        R.append((k,x.compound,x.split_count,*counts(x.gold_split.split('+'),p.split(),x.compound)))
d=pd.DataFrame(R,columns=['sys','compound','stratum','tp','nc','ng'])
def pw(g):
    n=g.groupby('stratum').size(); w=g.stratum.map(lambda s:POP[s]/n[s]); return f((g.tp*w).sum(),(g.nc*w).sum(),(g.ng*w).sum())
for s in ('s1','s2'):
    a=d[d.sys=='dcs_'+s]; b=d[d.sys=='fr_'+s]; print(s,'pooled weighted: dcs %.3f fr %.3f gap %+.3f'%(pw(a),pw(b),pw(a)-pw(b)))
