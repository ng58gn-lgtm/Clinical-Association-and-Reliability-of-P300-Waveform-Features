"""
Two further tests for RQ1 and RQ2. Post-hoc.

Whether any feature reaches at least moderate reliability, by the lower bound
of its confidence interval, and the between-cohort latency difference set
against the patient-control difference.

    python analysis_O_rq1_rq2.py
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

FEATS=["p300_latency_s","tgt_p300_latency_s","p300_prominence_uv","tgt_p300_prominence_uv","p300_amplitude_uv",
       "tgt_p300_amplitude_uv","p300_mean_uv","tgt_p300_mean_uv","p300_auc_uvs","tgt_p300_auc_uvs","diff_mean_uv"]
SEED=20260924

def icc21(Y, alpha=0.05):
    """ICC(2,1), two-way random, absolute agreement, single measurement, with McGraw & Wong (1996) CI."""
    Y=np.asarray(Y,float); n,k=Y.shape; gm=Y.mean()
    msr=k*((Y.mean(1)-gm)**2).sum()/(n-1); msc=n*((Y.mean(0)-gm)**2).sum()/(k-1)
    mse=((Y-Y.mean(1,keepdims=True)-Y.mean(0,keepdims=True)+gm)**2).sum()/((n-1)*(k-1))
    icc=(msr-mse)/(msr+(k-1)*mse+k*(msc-mse)/n)
    a=k*icc/(n*(1-icc)); b=1+k*icc*(n-1)/(n*(1-icc))
    v=(a*msc+b*mse)**2/((a*msc)**2/(k-1)+(b*mse)**2/((n-1)*(k-1)))
    Fl=stats.f.ppf(1-alpha/2,n-1,v); Fu=stats.f.ppf(1-alpha/2,v,n-1)
    lo=n*(msr-Fl*mse)/(Fl*(k*msc+(k*n-k-n)*mse)+n*msr); hi=n*(Fu*msr-mse)/(k*msc+(k*n-k-n)*mse+n*Fu*msr)
    return icc,lo,hi,n,k

def load(p, excl=True):
    d=pd.read_csv(p); d=d[d.qc_pass==1]
    if excl and 'group' in d: d=d[d.group.astype(str).str.upper()!='EXCLUDED']
    return d

def designs(S,P,O):
    out={}
    s=S.sort_values(['subject_uid','session']).groupby('subject_uid').head(2)
    s=s[s.groupby('subject_uid').subject_uid.transform('size')==2].copy()
    s['rep']=s.groupby('subject_uid').cumcount()
    out['Study-Siesta retest']=s
    p=P[P.group=='PD'].copy(); p['rep']=(p.med_state=='OFF').astype(int)
    out['Parkinson ON/OFF']=p
    o=O.sort_values(['subject_uid','session']).copy(); o['rep']=o.groupby('subject_uid').cumcount()
    out['ds003061 runs']=o
    return out

def mw_sep(a,b):
    u=stats.mannwhitneyu(a,b).statistic; auc=u/(len(a)*len(b)); return max(auc,1-auc)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--siesta'); ap.add_argument('--ds003061'); ap.add_argument('--parkinson'); ap.add_argument('--outdir',default='.')
    a=ap.parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    S,P,O=load(a.siesta),load(a.parkinson),load(a.ds003061)
    rows=[]
    for name,d in designs(S,P,O).items():
        for f in FEATS:
            w=d.pivot_table(index='subject_uid',columns='rep',values=f).dropna()
            icc,lo,hi,n,k=icc21(w.values)
            rows.append(dict(design=name,feature=f,n=n,k=k,icc=icc,ci_low=lo,ci_high=hi,at_least_moderate=lo>0.5))
    r1=pd.DataFrame(rows); r1.round(4).to_csv(out/'analysis_O_icc_ci.csv',index=False)
    # H0-2b
    rng=np.random.default_rng(SEED)
    ctl=P[P.group=='CTL'].groupby('subject_uid').p300_latency_s.mean().dropna().values
    off=P[(P.group=='PD')&(P.med_state=='OFF')].groupby('subject_uid').p300_latency_s.mean().dropna().values
    sie=S.groupby('subject_uid').p300_latency_s.mean().dropna().values
    obs=mw_sep(ctl,sie)-mw_sep(off,ctl); B=[]
    for _ in range(5000):
        c=rng.choice(ctl,len(ctl)); s=rng.choice(sie,len(sie)); o=rng.choice(off,len(off)); B.append(mw_sep(c,s)-mw_sep(o,c))
    r2=pd.DataFrame([dict(sep_dataset=mw_sep(ctl,sie),sep_disease=mw_sep(off,ctl),difference=obs,ci_low=np.percentile(B,2.5),ci_high=np.percentile(B,97.5),n_ctl=len(ctl),n_pd_off=len(off),n_siesta=len(sie))])
    r2.round(4).to_csv(out/'analysis_O_dataset_vs_disease.csv',index=False)
    # H0-2c
    C={k:v.groupby('subject_uid')[FEATS].mean().corr(method='spearman').values for k,v in [('Study-Siesta',S),('ds003061',O),('Parkinson-Oddball',P)]}
    iu=np.triu_indices(len(FEATS),1); rows=[]
    for x,y in [('Study-Siesta','Parkinson-Oddball'),('Study-Siesta','ds003061'),('ds003061','Parkinson-Oddball')]:
        A,Bm=C[x],C[y]; r0=stats.spearmanr(A[iu],Bm[iu])[0]; cnt=0
        for _ in range(10000):
            p=rng.permutation(len(FEATS))
            if stats.spearmanr(A[iu],Bm[np.ix_(p,p)][iu])[0]>=r0: cnt+=1
        rows.append(dict(pair=f'{x} vs {y}',mantel_r=r0,p=(cnt+1)/10001))
    r3=pd.DataFrame(rows); r3.round(4).to_csv(out/'analysis_O_mantel.csv',index=False)
    pd.set_option('display.width',200); print(r1.round(3).to_string(index=False)); print(r2.round(3).to_string(index=False)); print(r3.round(4).to_string(index=False))
if __name__=='__main__': main()
