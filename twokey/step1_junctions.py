import difflib, unicodedata, pandas as pd, sys
nf=lambda s: unicodedata.normalize('NFC',str(s).strip())
def cuts(pieces,surf):
    c=''.join(pieces); offs=[];acc=0
    for p in pieces[:-1]: acc+=len(p); offs.append(acc)
    ops=difflib.SequenceMatcher(None,c,surf,autojunk=False).get_opcodes(); out=[]
    for i in offs:
        for t,i1,i2,j1,j2 in ops:
            if i1<=i<=i2 and (i<i2 or t=='equal' or i2==len(c)):
                out.append(j1+(i-i1) if t=='equal' else j1+round((i-i1)*(j2-j1)/max(i2-i1,1)));break
    return out
def has(x,L): return any(abs(x-y)<=1 for y in L)
P='/'.join(sys.argv[1:2]) or ''
R=sys.argv[1]
runs={'Frontiers':['byt5_tagged','byt5_tagged_s2','tagged_s7','tagged_s21','tagged_s99'],'DCS':['tagged_on_dcs','tagged_s2_on_dcs','tagged_s7_on_dcs','tagged_s21_on_dcs','tagged_s99_on_dcs']}
rows=[]
for test,L in runs.items():
  for r in L:
    a=pd.read_csv(f'{R}/{r}/results_fr.csv'); b=pd.read_csv(f'{R}/{r}/results_dcs.csv')
    m=a[['compound','predicted']].merge(b[['compound','predicted']],on='compound',suffixes=('_fr','_dcs')).dropna()
    C=dict(same=0,spelling=0,deeper=0,shallower=0,mixed=0); add=rem=tot_fr=0
    for s,pf,pd_ in zip(m.compound,m.predicted_fr,m.predicted_dcs):
        s=nf(s); f=[nf(x) for x in str(pf).split()]; d=[nf(x) for x in str(pd_).split()]
        cf,cd=cuts(f,s),cuts(d,s)
        A=[x for x in cd if not has(x,cf)]; Rm=[x for x in cf if not has(x,cd)]
        add+=len(A); rem+=len(Rm); tot_fr+=len(cf)
        if not A and not Rm: C['same' if f==d else 'spelling']+=1
        elif A and not Rm: C['deeper']+=1
        elif Rm and not A: C['shallower']+=1
        else: C['mixed']+=1
    n=len(m); rows.append(dict(test=test,run=r,n=n,**{k:v/n for k,v in C.items()},cuts_added=add,cuts_removed=rem,fr_cuts=tot_fr))
D=pd.DataFrame(rows); pd.set_option('display.width',200); print(D.round(3).to_string(index=False))
print(D.groupby('test')[['same','spelling','deeper','shallower','mixed']].agg(['mean','min','max']).round(3).to_string())
print(D.groupby('test')[['cuts_added','cuts_removed','fr_cuts']].sum())
