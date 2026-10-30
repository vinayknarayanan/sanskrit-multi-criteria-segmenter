import pandas as pd, sys
sys.argv=[sys.argv[0]]; exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'step1_junctions.py')).read().split("P='/'")[0])
import os; R=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','predictions')
for r in ['tagged_on_dcs','tagged_s2_on_dcs','tagged_s7_on_dcs','tagged_s21_on_dcs','tagged_s99_on_dcs']:
    a=pd.read_csv(f'{R}/{r}/results_fr.csv'); b=pd.read_csv(f'{R}/{r}/results_dcs.csv')
    m=a[['compound','gold_split','predicted']].merge(b[['compound','predicted']],on='compound',suffixes=('_fr','_dcs')).dropna()
    ag=ok=rg=rok=0
    for s,g,pf,pd_ in zip(m.compound,m.gold_split,m.predicted_fr,m.predicted_dcs):
        s=nf(s); gc=cuts([nf(x) for x in g.split('+')],s); cf=cuts(str(pf).split(),s); cd=cuts(str(pd_).split(),s)
        for x in cd:
            if not has(x,cf): ag+=1; ok+=has(x,gc)
        for x in cf:
            if not has(x,cd): rg+=1; rok+=not has(x,gc)
    print(r,'cuts <dcs> adds:',ag,'correct vs DCS gold %.3f'%(ok/ag),'| cuts <dcs> removes:',rg,'removal correct %.3f'%(rok/rg))
