import paths as P
import pickle,re,unicodedata,io,contextlib,pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    from robust_ci import cuts_on_surface
b=lambda s: unicodedata.normalize('NFC',str(s).strip().lower()).replace('ṃ','m')
key=lambda s: re.sub(r'(?:[ḥmsrtdn]|o|e)$','',b(s)) if len(b(s))>3 else b(s)
D=pickle.load(open(P.CACHE + 'dcs_surface.pkl','rb'))['idx']
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'converted_gold.py'), encoding='utf-8').read().split('PRED={}')[0])  # builds conv, ev, f
ok=same=tot=0; diff_ok=diff_same=0; rows=[]
for c,dg in conv.items():
    cc=D.get(key(c))
    if not cc: continue
    full=[m[0] for m in cc.most_common(1)[0][0]]
    if len(full)<2 and len(dg)<2: continue
    a=sorted(cuts_on_surface(dg,c)); z=sorted(cuts_on_surface(full,c))
    m=len(a)==len(z) and all(abs(x-y)<=1 for x,y in zip(a,z))
    fg=ev.loc[c,'gold_split'].split('+'); changed=len(dg)!=len(fg)
    rows.append((c,changed,m,len(dg),len(full)))
r=pd.DataFrame(rows,columns=['c','changed','match','n_recon','n_full'])
print('converted items with a full-form DCS analysis:',len(r),'of',len(conv))
print(r.groupby('changed').match.agg(['size','mean']))
print(r[~r.match].head(8).to_string())
