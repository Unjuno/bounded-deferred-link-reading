import json, math
from functools import lru_cache
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

FIT_SEED=26082602; FIT_EPISODES=700; TEST_SEEDS=tuple(range(920001,920013)); NTEST=200
HORIZONS=(24,40); KVALS=(2,4,8,16); NBOOT=10000
MAX_DEPTH=3; N_ROOTS=6; MAX_NODES=18; RHO=.90; LATE_SCALE=1.0; NEED_NOISE=1.5

def sig(x): return 1/(1+math.exp(-x))

def gen(rng):
    nodes=[]; roots=[]; H=max(HORIZONS)
    def add(d,z,p,rel,need):
        i=len(nodes)
        nodes.append({'i':i,'d':d,'z':z,'r':float(math.exp(.8*z+rng.normal(0,.25))),
          'l':int(np.clip(np.rint(math.exp(rng.normal(1,.55))),1,8)),
          'c':float(z+rng.normal(0,1.35)),'p':p,'ch':[],'rel':rel,'need':need,
          'nc':int(np.clip(np.rint(need+rng.normal(0,NEED_NOISE)),0,H))})
        return i
    for rel in sorted(map(int,rng.integers(0,11,size=N_ROOTS))):
        z=float(rng.normal()); roots.append(add(0,z,None,rel,min(H,rel+int(rng.integers(2,11)))))
    def expd(i):
        if len(nodes)>=MAX_NODES or nodes[i]['d']>=MAX_DEPTH:return
        n=nodes[i]; m=min(3,int(rng.poisson(.35+1.45*sig(n['z']))))
        for _ in range(m):
            if len(nodes)>=MAX_NODES:break
            z=RHO*n['z']+math.sqrt(1-RHO*RHO)*float(rng.normal())
            j=add(n['d']+1,z,i,n['rel'],min(H,n['need']+int(rng.integers(1,7))))
            n['ch'].append(j); expd(j)
    for i in roots: expd(i)
    return nodes

def reward(n,t,h):
    end=t+n['l']
    if end>h:return 0.
    return n['r']*math.exp(-max(0,end-n['need'])/LATE_SCALE)

def disc(nodes,mask,t):
    return [n['i'] for n in nodes if not(mask>>n['i']&1) and ((n['p'] is None and n['rel']<=t) or (n['p'] is not None and mask>>n['p']&1))]

def nextrel(nodes,mask,t,h):
    a=[n['rel'] for n in nodes if n['p'] is None and not(mask>>n['i']&1) and t<n['rel']<=h]
    return min(a) if a else None

def oracle(nodes,h):
    @lru_cache(None)
    def V(mask,t):
        a=[i for i in disc(nodes,mask,t) if t+nodes[i]['l']<=h]
        if not a:
            q=nextrel(nodes,mask,t,h); return 0. if q is None else V(mask,q)
        return max(reward(nodes[i],t,h)+V(mask|1<<i,t+nodes[i]['l']) for i in a)
    return V(0,0)

def branch(nodes,root,t,h):
    allow=set(); st=[root]
    while st:
        i=st.pop()
        if i in allow:continue
        allow.add(i);st.extend(nodes[i]['ch'])
    @lru_cache(None)
    def V(mask,tt):
        a=[]
        for i in allow:
            if mask>>i&1:continue
            p=nodes[i]['p']
            if (i==root and tt==t) or (p is not None and mask>>p&1):a.append(i)
        a=[i for i in a if tt+nodes[i]['l']<=h]
        return 0. if not a else max(reward(nodes[i],tt,h)+V(mask|1<<i,tt+nodes[i]['l']) for i in a)
    return V(0,t)

def rank(nodes,i,f): return (1+sum(nodes[j]['c']>nodes[i]['c'] for j in f))/max(1,len(f))

def feat(nodes,i,t,h,f,mask,slack):
    n=nodes[i]; pr=nodes[n['p']]['r'] if n['p'] is not None and mask>>n['p']&1 else 1.
    x=[n['c'],math.log1p(n['l']),n['d']/MAX_DEPTH,math.log1p(pr),(h-t)/h,math.log1p(len(f)),n['c']/n['l'],rank(nodes,i,f)]
    if slack:
        s=n['nc']-t; e=max(0,n['l']-s); x += [s/h,e/max(1,n['l']),(n['nc']-t-n['l'])/h]
    return np.array(x,float)

def rows(nodes,h,slack):
    mask=t=0; X=[]; yi=[]; yr=[]
    while t<h:
        f=disc(nodes,mask,t); a=[i for i in f if t+nodes[i]['l']<=h]
        if not a:
            q=nextrel(nodes,mask,t,h)
            if q is None:break
            t=q;continue
        for i in a:X.append(feat(nodes,i,t,h,f,mask,slack));yi.append(reward(nodes[i],t,h));yr.append(branch(nodes,i,t,h))
        i=max(a,key=lambda j:(nodes[j]['c']/nodes[j]['l']+.08*max(0,nodes[j]['l']-(nodes[j]['nc']-t)),nodes[j]['c']))
        t+=nodes[i]['l'];mask|=1<<i
    return X,yi,yr

def train():
    rng=np.random.default_rng(FIT_SEED); xr=[];xs=[];yi=[];yr=[]
    for ep in range(FIT_EPISODES):
        n=gen(rng)
        for h in HORIZONS:
            a,ia,ra=rows(n,h,False);b,ib,rb=rows(n,h,True)
            if not(np.allclose(ia,ib) and np.allclose(ra,rb)):raise RuntimeError('matched rows diverged')
            xr+=a;xs+=b;yi+=ia;yr+=ra
        if ep%175==0:print('fit_episode',ep,'rows',len(yr))
    def fit(X,y):
        m=make_pipeline(StandardScaler(),Ridge(alpha=10.));m.fit(np.vstack(X),np.asarray(y));return m
    return fit(xr,yr),fit(xs,yr),fit(xs,yi),len(yr)

def pred(m,n,f,t,h,mask,a,slack):return m.predict(np.vstack([feat(n,i,t,h,f,mask,slack) for i in a]))

def rollout(n,h,m=None,slack=False,k=None,adaptive=False,kind='learned'):
    mask=t=0;tot=0.;store=set();drop=set();sizes=[]
    while t<h:
        for i in disc(n,mask,t):
            if i not in drop:store.add(i)
        f=list(store)
        if m is not None and f:
            p=pred(m,n,f,t,h,mask,f,slack);o=np.argsort(-p)
            keep=len(f)
            if adaptive:
                keep=min(8,len(f));v=np.maximum(0,p[o[:keep]])
                if keep>4 and v.sum()>0 and v[:4].sum()/v.sum()>=.90:keep=4
            elif k is not None:keep=min(k,len(f))
            if keep<len(f):
                kept=[f[int(j)] for j in o[:keep]];drop.update(set(f)-set(kept));store=set(kept);f=kept
        sizes.append(len(f));a=[i for i in f if t+n[i]['l']<=h]
        if not a:
            q=nextrel(n,mask,t,h)
            if q is None:break
            t=q;continue
        if kind=='rawgreedy':i=max(a,key=lambda j:(n[j]['c']/n[j]['l'],n[j]['c']))
        elif kind=='effgreedy':i=max(a,key=lambda j:(n[j]['c']/(1+max(0,n[j]['l']-(n[j]['nc']-t))),n[j]['c']))
        elif kind=='dfs':i=max(a,key=lambda j:(n[j]['d'],n[j]['c']/n[j]['l']))
        else:i=a[int(np.argmax(pred(m,n,f,t,h,mask,a,slack)))]
        store.discard(i);tot+=reward(n[i],t,h);t+=n[i]['l'];mask|=1<<i
    return tot,float(np.mean(sizes)) if sizes else 0.

def boot(d,rng):
    z=[np.mean(d[rng.integers(0,len(d),len(d))]) for _ in range(NBOOT)]
    return [float(x) for x in np.quantile(z,[.025,.975])]

def main():
    raw,rec,imm,nrows=train(); vals={h:{} for h in HORIZONS}; mem={h:{} for h in HORIZONS}
    pol=['dfs','rawgreedy','effgreedy','rawrec','rec','imm']+[f'k{k}' for k in KVALS]+['adapt']
    for h in HORIZONS:
        for p in pol:vals[h][p]=[];mem[h][p]=[]
    for seed in TEST_SEEDS:
        rng=np.random.default_rng(seed);sv={h:{p:[] for p in pol} for h in HORIZONS};sm={h:{p:[] for p in pol} for h in HORIZONS}
        for _ in range(NTEST):
            n=gen(rng)
            for h in HORIZONS:
                o=oracle(n,h)
                calls={'dfs':(None,False,None,False,'dfs'),'rawgreedy':(None,False,None,False,'rawgreedy'),'effgreedy':(None,False,None,False,'effgreedy'),
                  'rawrec':(raw,False,None,False,'learned'),'rec':(rec,True,None,False,'learned'),'imm':(imm,True,None,False,'learned'),'adapt':(rec,True,None,True,'learned')}
                for k in KVALS:calls[f'k{k}']=(rec,True,k,False,'learned')
                for p,a in calls.items():
                    r,mm=rollout(n,h,*a);sv[h][p].append(r/o);sm[h][p].append(mm)
        for h in HORIZONS:
            for p in pol:vals[h][p].append(float(np.mean(sv[h][p])));mem[h][p].append(float(np.mean(sm[h][p])))
        print('test_seed_done',seed)
    rng=np.random.default_rng(880022)
    def delta(a,b,h):
        d=np.asarray(vals[h][a])-np.asarray(vals[h][b]);return {'mean':float(d.mean()),'ci95':boot(d,rng),'positive_seeds':int((d>0).sum()),'seed_deltas':d.tolist()}
    h1=delta('rec','rawrec',24);h1l=delta('rec','rawrec',40);h2=delta('rec','imm',24);h2l=delta('rec','imm',40);ve=delta('rec','effgreedy',24)
    cap={};mink=None
    for k in KVALS:
        cap[str(k)]={};ok=True
        for h in HORIZONS:
            d=np.asarray(vals[h]['rec'])-np.asarray(vals[h][f'k{k}']);cap[str(k)][str(h)]={'mean_full_minus_k':float(d.mean()),'ci95':boot(d,rng)}
            if d.mean()>.01:ok=False
        if mink is None and ok:mink=k
    adapt={};ads=True
    for h in HORIZONS:
        pg=delta('k8','adapt',h);m8=float(np.mean(mem[h]['k8']));ma=float(np.mean(mem[h]['adapt']));mr=0 if m8<=0 else 1-ma/m8
        adapt[str(h)]={'k8_minus_adaptive_performance':pg,'k8_mean_retained':m8,'adaptive_mean_retained':ma,'memory_reduction_fraction':mr}
        if pg['mean']>.005 or mr<.10:ads=False
    agg={str(h):{p:{'mean_oracle_fraction':float(np.mean(vals[h][p])),'mean_retained':float(np.mean(mem[h][p]))} for p in pol} for h in HORIZONS}
    ph1=h1['mean']>=.01 and h1['ci95'][0]>0 and h1['positive_seeds']>=9 and h1l['mean']>=0
    ph2=h2['mean']>=.01 and h2['ci95'][0]>0 and h2['positive_seeds']>=9 and h2l['mean']>=-.01
    dec='PASS' if ph1 and ph2 else ('PARTIAL' if ph1 else 'FAIL')
    out={'name':'AP-LM2 asynchronous slack-aware recursive query scheduling','fit_seed':FIT_SEED,'fit_episodes':FIT_EPISODES,'teacher_rows':nrows,'test_seeds':list(TEST_SEEDS),'episodes_per_seed':NTEST,'horizons':list(HORIZONS),'aggregate':agg,
      'h1_slack_vs_raw_h24':h1,'h1_long_h40':h1l,'h2_recursive_vs_immediate_h24':h2,'h2_long_h40':h2l,'slack_recursive_vs_effective_greedy_h24':ve,
      'capacity_full_minus_k':cap,'minimal_k_within_1pp_of_full_at_both_horizons':mink,'adaptive':adapt,'pass_h1':ph1,'pass_h2':ph2,'adaptive_resource_success':ads,'decision':dec,
      'boundaries':['hallucination=0','noisy synthetic slack cue','one query server','non-idling policy class','not human comprehension','development seeds 280001..280005 excluded']}
    d=Path('artifacts/ap_lm2');d.mkdir(parents=True,exist_ok=True);(d/'AP_LM2_RESULTS.json').write_text(json.dumps(out,indent=2))
    summary=f"# AP-LM2 Summary\n\nDecision: **{dec}**\n\nH1 H=24: {100*h1['mean']:+.3f} pp, CI [{100*h1['ci95'][0]:+.3f},{100*h1['ci95'][1]:+.3f}], positive {h1['positive_seeds']}/12; H=40 {100*h1l['mean']:+.3f} pp.\n\nH2 H=24: {100*h2['mean']:+.3f} pp, CI [{100*h2['ci95'][0]:+.3f},{100*h2['ci95'][1]:+.3f}], positive {h2['positive_seeds']}/12; H=40 {100*h2l['mean']:+.3f} pp.\n\nMinimum K within 1 pp: **{mink}**. Adaptive resource success: **{ads}**.\n"
    (d/'AP_LM2_SUMMARY.md').write_text(summary)
    print('AP_LM2_DECISION',dec);print('AP_LM2_H1',json.dumps(h1));print('AP_LM2_H1_LONG',json.dumps(h1l));print('AP_LM2_H2',json.dumps(h2));print('AP_LM2_H2_LONG',json.dumps(h2l));print('AP_LM2_MIN_K',mink);print('AP_LM2_ADAPTIVE',ads,json.dumps(adapt))
if __name__=='__main__':main()
