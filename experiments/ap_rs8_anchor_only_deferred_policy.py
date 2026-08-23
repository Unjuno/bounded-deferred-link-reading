#!/usr/bin/env python3
"""AP-RS8: visible anchor-only local vs bounded deferred-link policy.

Motivated by the preregistered AP-RS5 result that anchor-only local navigation was
substantially stronger than equal anchor+paragraph scoring on goal-directed Wikispeedia.
RS8 tests whether bounded one-shot reconsideration adds material value even when the
local semantic channel is already strong. No unvisited candidate metadata is used.
"""
from __future__ import annotations
import csv, hashlib, json, math, os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from experiments.ap_rs5_real_anchor_context_policy import GRAPH_URL, HTML_URL, download, extract_graph, build_visible_link_corpus

SEED=20260827
K=4
FIT_N=700
TUNE_N=500
TEST_N=1200
N_BOOT=2000
MARGINS=[-0.20,-0.10,-0.05,0.0,0.02,0.05,0.10,0.20,0.40,999.0]
PRIMARY_BUDGET=16


def choose_tasks(missions,target_set,n,rng):
    xs=[x for x in missions if x[1] in target_set]; rng.shuffle(xs); return xs[:min(n,len(xs))]

def target_ci(rows,key,offset=0):
    by=defaultdict(list)
    for r in rows: by[r['target']].append(r[key])
    groups=[(sum(v),len(v)) for v in by.values()]; rng=np.random.default_rng(SEED+offset); reps=np.empty(N_BOOT)
    for b in range(N_BOOT):
        ii=rng.integers(0,len(groups),len(groups)); reps[b]=sum(groups[i][0] for i in ii)/sum(groups[i][1] for i in ii)
    return [float(np.quantile(reps,.025)),float(np.quantile(reps,.975))]

def buckets(rows,key):
    vv=[[] for _ in range(8)]
    for r in rows: vv[int(hashlib.md5(r['target'].encode()).hexdigest()[:8],16)%8].append(r[key])
    means=[float(np.mean(v)) if v else float('nan') for v in vv]; z=np.asarray([x for x in means if np.isfinite(x)])
    uc=float(z.std(ddof=1)/math.sqrt(len(z))) if len(z)>1 else float('nan')
    return {'means':means,'positive':int(sum(x>0 for x in means if np.isfinite(x))),'u_c':uc,'U_k1.96':float(1.96*uc) if np.isfinite(uc) else None}

def main():
    out=Path(os.environ.get('AP_RS8_OUT','artifacts/ap_rs8')); out.mkdir(parents=True,exist_ok=True); raw=out/'raw'; raw.mkdir(exist_ok=True)
    gt=raw/'graph.tar.gz'; ht=raw/'html.tar.gz'; download(GRAPH_URL,gt); download(HTML_URL,ht)
    articles,links,missions=extract_graph(gt,raw/'graph'); idx={a:i for i,a in enumerate(articles)}
    occ,anchors,contexts,cov=build_visible_link_corpus(ht,articles,links)
    if cov<.85: raise RuntimeError(f'visible edge coverage {cov:.3f}')
    enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    te=enc.encode([a.replace('_',' ') for a in articles],batch_size=128,show_progress_bar=True,normalize_embeddings=True)
    ae=enc.encode(anchors,batch_size=128,show_progress_bar=True,normalize_embeddings=True)
    @lru_cache(maxsize=30000)
    def scored(src,goal):
        gv=te[goal]; best={}
        for v,ai,ci in occ[src]:
            s=float(np.dot(ae[ai],gv))
            if v not in best or s>best[v]: best[v]=s
        return tuple(sorted(best.items(),key=lambda x:(-x[1],x[0])))
    def avail(src,goal,seen): return [(v,s) for v,s in scored(src,goal) if v not in seen]
    def rollout(start,goal,steps,budget,seen):
        cur=start; ss=set(seen)
        if cur==goal: return True
        while steps<budget:
            xs=avail(cur,goal,ss)
            if not xs: return False
            cur=xs[0][0]; steps+=1
            if cur==goal: return True
            ss.add(cur)
        return False
    rng=np.random.default_rng(SEED); targets=sorted({t for s,t in missions if s in idx and t in idx}); rng.shuffle(targets); nt=len(targets)
    ft=set(targets[:int(.40*nt)]); tt=set(targets[int(.40*nt):int(.60*nt)]); et=set(targets[int(.60*nt):])
    fit=choose_tasks(missions,ft,FIT_N,rng); tune=choose_tasks(missions,tt,TUNE_N,rng); test=choose_tasks(missions,et,TEST_N,rng)
    teacher=[]
    for qi,(source,target) in enumerate(fit):
        cur=idx[source]; goal=idx[target]; seen={cur}; steps=0; prev=[]
        while steps<PRIMARY_BUDGET and cur!=goal:
            opts=avail(cur,goal,seen); cb=opts[0][1] if opts else 0.0
            if prev:
                cont=rollout(cur,goal,steps,PRIMARY_BUDGET,seen)
                for v,s,count,rank in prev:
                    if v in seen or steps+2>PRIMARY_BUDGET: continue
                    back=rollout(v,goal,steps+2,PRIMARY_BUDGET,seen|{v})
                    teacher.append(([math.log1p(count),s-cb,float(rank)],float(back)-float(cont)))
            if not opts: break
            prev=[(v,s,len(opts),rank) for rank,(v,s) in enumerate(opts[1:1+K],start=2)]
            cur=opts[0][0]; steps+=1
            if cur==goal: break
            seen.add(cur)
        if qi%100==0: print('teacher',qi,len(teacher))
    X=np.asarray([x for x,y in teacher],float); y=np.asarray([y for x,y in teacher],float)
    reg=make_pipeline(StandardScaler(),Ridge(alpha=1.0)); reg.fit(X,y); pred=reg.predict(X); m=y!=0; auc=None
    if m.sum() and len(np.unique(y[m]>0))==2: auc=float(roc_auc_score((y[m]>0).astype(int),pred[m]))
    def run(task,budget,margin=None):
        cur=idx[task[0]]; goal=idx[task[1]]; seen={cur}; steps=0; prev=[]; used=False
        while steps<budget and cur!=goal:
            opts=avail(cur,goal,seen); cb=opts[0][1] if opts else 0.0
            if margin is not None and not used and steps<PRIMARY_BUDGET and prev and steps+2<=budget:
                feats=[]; valid=[]
                for v,s,count,rank in prev:
                    if v in seen: continue
                    feats.append([math.log1p(count),s-cb,float(rank)]); valid.append(v)
                if feats:
                    pp=reg.predict(np.asarray(feats,float)); j=int(np.argmax(pp))
                    if float(pp[j])>margin:
                        cur=valid[j]; steps+=2; used=True; prev=[]
                        if cur==goal: return True,steps,True
                        seen.add(cur); continue
            if not opts: return False,steps,used
            prev=[(v,s,len(opts),rank) for rank,(v,s) in enumerate(opts[1:1+K],start=2)]
            cur=opts[0][0]; steps+=1
            if cur==goal: return True,steps,used
            seen.add(cur)
        return cur==goal,steps,used
    def evaluate(tasks,margin=None):
        rr=[]
        for task in tasks:
            l16=run(task,16)[0]; l32=run(task,32)[0]
            if margin is None: p16,p32,u=l16,l32,False
            else:
                p16,_,u1=run(task,16,margin); p32,_,u2=run(task,32,margin); u=u1 or u2
            rr.append({'source':task[0],'target':task[1],'local16':int(l16),'local32':int(l32),'policy16':int(p16),'policy32':int(p32),'d16':int(p16)-int(l16),'d32':int(p32)-int(l32),'intervened':int(u)})
        return rr
    grid=[]; selected=None
    for margin in MARGINS:
        rr=evaluate(tune,margin); d16=float(np.mean([r['d16'] for r in rr])); d32=float(np.mean([r['d32'] for r in rr])); intr=float(np.mean([r['intervened'] for r in rr])); z={'margin':margin,'d16':d16,'d32':d32,'intervention_rate':intr}; grid.append(z)
        if d32>=-.005 and (selected is None or d16>selected['d16'] or (d16==selected['d16'] and d32>selected['d32'])): selected=z
    assert selected is not None; margin=selected['margin']; rr=evaluate(test,margin)
    d16=float(np.mean([r['d16'] for r in rr])); d32=float(np.mean([r['d32'] for r in rr])); ci16=target_ci(rr,'d16',1); ci32=target_ci(rr,'d32',37); b16=buckets(rr,'d16'); b32=buckets(rr,'d32')
    l16=float(np.mean([r['local16'] for r in rr])); l32=float(np.mean([r['local32'] for r in rr])); p16=float(np.mean([r['policy16'] for r in rr])); p32=float(np.mean([r['policy32'] for r in rr])); intr=float(np.mean([r['intervened'] for r in rr]))
    cond={'test_n_ge_400':len(rr)>=400,'visible_edge_coverage_ge_0.85':cov>=.85,'S16_gain_ge_2pp':d16>=.02,'S16_target_CI_lower_gt_0':ci16[0]>0,'S16_positive_target_buckets_ge_6_of_8':b16['positive']>=6,'S32_mean_noninferior_minus_0_5pp':d32>=-.005,'S32_target_CI_lower_ge_minus_1pp':ci32[0]>=-.01}
    decision='PASS' if all(cond.values()) else 'FAIL'
    result={'phase':'AP-RS8','name':'visible anchor-only bounded deferred-link policy','decision':decision,'preregistered_conditions':cond,
            'construct':'real Wikispeedia visible anchor text, real graph and human mission distribution; simulated policy','data':{'articles':len(articles),'graph_links':len(links),'missions':len(missions),'visible_edge_coverage':cov},
            'split':{'target_disjoint':True,'fit_tasks':len(fit),'tune_tasks':len(tune),'test_tasks':len(test),'fit_targets':len(ft),'tune_targets':len(tt),'test_targets':len(et)},
            'teacher':{'rows':len(y),'non_tie_rows':int(m.sum()),'non_tie_sign_auc_fit':auc,'features':['log1p(origin_candidate_count)','relative_anchor_score','origin_rank']},
            'tuning':{'grid':grid,'selected':selected,'safety_constraint':'mean S@32 harm <= 0.5pp'},
            'test':{'local':{'S16':l16,'S32':l32},'policy':{'S16':p16,'S32':p32,'delta_S16':d16,'delta_S32':d32,'target_cluster_CI95_S16':ci16,'target_cluster_CI95_S32':ci32,'target_bucket_S16':b16,'target_bucket_S32':b32,'intervention_rate':intr}},
            'boundary':['Motivated by AP-RS5 anchor-only local superiority; this is a new post-RS5 hypothesis.','All gate features are available at the current or previous page.','Navigation simulation is not human comprehension.']}
    (out/'AP_RS8_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'AP_RS8_SUMMARY.md').write_text(f"# AP-RS8 — Anchor-only bounded deferred policy\n\n**Decision: {decision}**\n\n- local S@16/S@32: {l16:.4f}/{l32:.4f}\n- policy S@16/S@32: {p16:.4f}/{p32:.4f}\n- delta S@16: {100*d16:+.3f} pp, CI {ci16}\n- delta S@32: {100*d32:+.3f} pp, CI {ci32}\n- positive S@16 buckets: {b16['positive']}/8\n- intervention rate: {intr:.4f}\n",encoding='utf-8')
    print('AP_RS8_DECISION',decision); print('AP_RS8_S16',d16,ci16,b16); print('AP_RS8_S32',d32,ci32,b32)
if __name__=='__main__': main()
