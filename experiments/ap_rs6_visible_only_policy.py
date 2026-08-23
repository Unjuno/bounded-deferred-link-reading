#!/usr/bin/env python3
"""AP-RS6: visible-only bounded deferred-link policy on real Wikispeedia HTML.

Preregistered before reading AP-RS5 results. This phase removes the one non-human-visible
feature in RS5: unvisited candidate out-degree. Only information available on the current
or immediately previous page is allowed.

Two small feature sets are fit on fit targets and compared only on tune targets:
  V3 = log1p(origin candidate count), relative equal semantic score, origin rank
  V4 = V3 + |anchor score - context score|
The feature set + threshold are selected on tune targets under the same S@32 safety
constraint as RS5. The selected policy is evaluated once on target-disjoint test targets.
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

from experiments.ap_rs5_real_anchor_context_policy import (
    GRAPH_URL, HTML_URL, download, extract_graph, build_visible_link_corpus,
)

SEED=20260825
K=4
FIT_N=700
TUNE_N=500
TEST_N=1200
N_BOOT=2000
MARGINS=[-0.20,-0.10,-0.05,0.0,0.02,0.05,0.10,0.20,0.40,999.0]
PRIMARY_BUDGET=16
SAFETY_BUDGET=32
MODELS=("V3","V4")


def choose_tasks(missions,target_set,n,rng):
    xs=[x for x in missions if x[1] in target_set]
    rng.shuffle(xs)
    return xs[:min(n,len(xs))]


def target_cluster_ci(rows,key,offset=0):
    by=defaultdict(list)
    for r in rows: by[r['target']].append(r[key])
    groups=[(sum(v),len(v)) for v in by.values()]
    rng=np.random.default_rng(SEED+offset); reps=np.empty(N_BOOT)
    for b in range(N_BOOT):
        ii=rng.integers(0,len(groups),len(groups))
        reps[b]=sum(groups[i][0] for i in ii)/sum(groups[i][1] for i in ii)
    return [float(np.quantile(reps,.025)),float(np.quantile(reps,.975))]


def bucket_stats(rows,key):
    vals=[[] for _ in range(8)]
    for r in rows:
        b=int(hashlib.md5(r['target'].encode()).hexdigest()[:8],16)%8
        vals[b].append(r[key])
    means=[float(np.mean(v)) if v else float('nan') for v in vals]
    finite=np.asarray([x for x in means if np.isfinite(x)],float)
    uc=float(finite.std(ddof=1)/math.sqrt(len(finite))) if len(finite)>1 else float('nan')
    return {'means':means,'positive':int(sum(x>0 for x in means if np.isfinite(x))),
            'u_c':uc,'U_k1.96':float(1.96*uc) if np.isfinite(uc) else None}


def main():
    out=Path(os.environ.get('AP_RS6_OUT','artifacts/ap_rs6')); out.mkdir(parents=True,exist_ok=True)
    raw=out/'raw'; raw.mkdir(exist_ok=True)
    graph_tar=raw/'wikispeedia_paths-and-graph.tar.gz'; html_tar=raw/'wikispeedia_articles_html.tar.gz'
    download(GRAPH_URL,graph_tar); download(HTML_URL,html_tar)
    articles, graph_links, missions=extract_graph(graph_tar,raw/'graph')
    article_idx={a:i for i,a in enumerate(articles)}
    edge_occ,anchors,contexts,coverage=build_visible_link_corpus(html_tar,articles,graph_links)
    if coverage<.85: raise RuntimeError(f'visible edge coverage too low: {coverage:.3f}')

    enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    title_emb=enc.encode([a.replace('_',' ') for a in articles],batch_size=128,show_progress_bar=True,normalize_embeddings=True)
    anchor_emb=enc.encode(anchors,batch_size=128,show_progress_bar=True,normalize_embeddings=True)
    context_emb=enc.encode(contexts,batch_size=64,show_progress_bar=True,normalize_embeddings=True)

    @lru_cache(maxsize=30000)
    def scored(src_i,goal_i):
        gv=title_emb[goal_i]; best={}
        for tgt_i,ai,ci in edge_occ[src_i]:
            sa=float(np.dot(anchor_emb[ai],gv)); sc=float(np.dot(context_emb[ci],gv)); s=.5*(sa+sc)
            if tgt_i not in best or s>best[tgt_i][0]: best[tgt_i]=(s,sa,sc)
        return tuple((v,*z) for v,z in sorted(best.items(),key=lambda kv:(-kv[1][0],kv[0])))

    def available(src_i,goal_i,seen): return [x for x in scored(src_i,goal_i) if x[0] not in seen]

    def local_rollout(start_i,goal_i,steps,budget,seen):
        cur=start_i; ss=set(seen)
        if cur==goal_i: return True
        while steps<budget:
            xs=available(cur,goal_i,ss)
            if not xs: return False
            cur=xs[0][0]; steps+=1
            if cur==goal_i: return True
            ss.add(cur)
        return False

    rng=np.random.default_rng(SEED)
    targets=sorted({t for s,t in missions if s in article_idx and t in article_idx}); rng.shuffle(targets)
    nt=len(targets); fit_targets=set(targets[:int(.40*nt)]); tune_targets=set(targets[int(.40*nt):int(.60*nt)]); test_targets=set(targets[int(.60*nt):])
    fit_tasks=choose_tasks(missions,fit_targets,FIT_N,rng); tune_tasks=choose_tasks(missions,tune_targets,TUNE_N,rng); test_tasks=choose_tasks(missions,test_targets,TEST_N,rng)
    print('split',len(fit_tasks),len(tune_tasks),len(test_tasks),'targets',len(fit_targets),len(tune_targets),len(test_targets))

    teacher=[]
    for ti,(source,target) in enumerate(fit_tasks):
        cur=article_idx[source]; goal=article_idx[target]; seen={cur}; steps=0; prev=[]
        while steps<PRIMARY_BUDGET and cur!=goal:
            opts=available(cur,goal,seen); current_best=opts[0][1] if opts else 0.0
            if prev:
                cont=local_rollout(cur,goal,steps,PRIMARY_BUDGET,seen)
                for alt_i,alt_s,sa,sc,origin_count,rank in prev:
                    if steps+2>PRIMARY_BUDGET or alt_i in seen: continue
                    back=local_rollout(alt_i,goal,steps+2,PRIMARY_BUDGET,seen|{alt_i})
                    teacher.append({'origin_count':origin_count,'rel':alt_s-current_best,'rank':rank,
                                    'disagree':abs(sa-sc),'y':float(back)-float(cont)})
            if not opts: break
            chosen=opts[0]; prev=[(v,s,sa,sc,len(opts),rank) for rank,(v,s,sa,sc) in enumerate(opts[1:1+K],start=2)]
            cur=chosen[0]; steps+=1
            if cur==goal: break
            seen.add(cur)
        if ti%100==0: print('teacher tasks',ti,'rows',len(teacher))

    def fvec(r,name):
        x=[math.log1p(r['origin_count']),r['rel'],float(r['rank'])]
        if name=='V4': x.append(r['disagree'])
        return x

    yy=np.asarray([r['y'] for r in teacher],float); non=yy!=0
    regs={}; fit_auc={}
    for name in MODELS:
        XX=np.asarray([fvec(r,name) for r in teacher],float)
        reg=make_pipeline(StandardScaler(),Ridge(alpha=1.0)); reg.fit(XX,yy); regs[name]=reg
        pp=reg.predict(XX); auc=None
        if non.sum() and len(np.unique(yy[non]>0))==2: auc=float(roc_auc_score((yy[non]>0).astype(int),pp[non]))
        fit_auc[name]=auc
        print(name,'rows',len(yy),'non-tie',int(non.sum()),'auc',auc)

    def run_policy(task,budget,name=None,margin=None):
        source,target=task; cur=article_idx[source]; goal=article_idx[target]; seen={cur}; steps=0; prev=[]; used=False
        if cur==goal: return True,steps,False
        while steps<budget:
            opts=available(cur,goal,seen); current_best=opts[0][1] if opts else 0.0
            if name is not None and margin is not None and (not used) and steps<PRIMARY_BUDGET and prev and steps+2<=budget:
                rr=[]; valid=[]
                for alt_i,alt_s,sa,sc,origin_count,rank in prev:
                    if alt_i in seen: continue
                    rr.append({'origin_count':origin_count,'rel':alt_s-current_best,'rank':rank,'disagree':abs(sa-sc)})
                    valid.append(alt_i)
                if rr:
                    pp=regs[name].predict(np.asarray([fvec(r,name) for r in rr],float)); j=int(np.argmax(pp))
                    if float(pp[j])>margin:
                        cur=valid[j]; steps+=2; used=True; prev=[]
                        if cur==goal: return True,steps,True
                        seen.add(cur); continue
            if not opts: return False,steps,used
            chosen=opts[0]
            prev=[(v,s,sa,sc,len(opts),rank) for rank,(v,s,sa,sc) in enumerate(opts[1:1+K],start=2)]
            cur=chosen[0]; steps+=1
            if cur==goal: return True,steps,used
            seen.add(cur)
        return False,steps,used

    def eval_tasks(tasks,name=None,margin=None):
        rows=[]
        for task in tasks:
            l16=run_policy(task,16)[0]; l32=run_policy(task,32)[0]
            if name is None: p16,p32,used=l16,l32,False
            else:
                p16,_,u16=run_policy(task,16,name,margin); p32,_,u32=run_policy(task,32,name,margin); used=u16 or u32
            rows.append({'source':task[0],'target':task[1],'local16':int(l16),'local32':int(l32),
                         'policy16':int(p16),'policy32':int(p32),'d16':int(p16)-int(l16),'d32':int(p32)-int(l32),'intervened':int(used)})
        return rows

    grid=[]; selected=None
    for name in MODELS:
        for margin in MARGINS:
            rr=eval_tasks(tune_tasks,name,margin); d16=float(np.mean([r['d16'] for r in rr])); d32=float(np.mean([r['d32'] for r in rr])); intr=float(np.mean([r['intervened'] for r in rr]))
            row={'model':name,'margin':margin,'d16':d16,'d32':d32,'intervention_rate':intr}; grid.append(row)
            if d32>=-.005:
                if selected is None or d16>selected['d16'] or (d16==selected['d16'] and d32>selected['d32']) or (d16==selected['d16'] and d32==selected['d32'] and name=='V3' and selected['model']=='V4'):
                    selected=row
    assert selected is not None
    print('selected',selected)
    name=selected['model']; margin=selected['margin']; test=eval_tasks(test_tasks,name,margin)
    d16=float(np.mean([r['d16'] for r in test])); d32=float(np.mean([r['d32'] for r in test])); ci16=target_cluster_ci(test,'d16',1); ci32=target_cluster_ci(test,'d32',37); b16=bucket_stats(test,'d16'); b32=bucket_stats(test,'d32')
    l16=float(np.mean([r['local16'] for r in test])); l32=float(np.mean([r['local32'] for r in test])); p16=float(np.mean([r['policy16'] for r in test])); p32=float(np.mean([r['policy32'] for r in test])); intr=float(np.mean([r['intervened'] for r in test]))
    conditions={'test_n_ge_400':len(test)>=400,'visible_edge_coverage_ge_0.85':coverage>=.85,'S16_gain_ge_2pp':d16>=.02,
                'S16_target_CI_lower_gt_0':ci16[0]>0,'S16_positive_target_buckets_ge_6_of_8':b16['positive']>=6,
                'S32_mean_noninferior_minus_0_5pp':d32>=-.005,'S32_target_CI_lower_ge_minus_1pp':ci32[0]>=-.01}
    decision='PASS' if all(conditions.values()) else 'FAIL'
    result={'phase':'AP-RS6','name':'visible-only real anchor/context bounded-deferred-link policy','decision':decision,
            'preregistered_conditions':conditions,'construct':'real Wikispeedia visible anchor + containing paragraph only; no unvisited candidate metadata',
            'data':{'articles':len(articles),'graph_links':len(graph_links),'unique_missions':len(missions),'visible_edge_coverage':coverage,
                    'unique_anchor_strings':len(anchors),'unique_context_strings':len(contexts)},
            'split':{'target_disjoint':True,'fit_tasks':len(fit_tasks),'tune_tasks':len(tune_tasks),'test_tasks':len(test_tasks),
                     'fit_targets':len(fit_targets),'tune_targets':len(tune_targets),'test_targets':len(test_targets)},
            'scorer':{'encoder':'sentence-transformers/all-MiniLM-L6-v2','anchor_weight':.5,'context_weight':.5,'weight_retuned':False},
            'teacher':{'rows':len(teacher),'non_tie_rows':int(non.sum()),'fit_auc':fit_auc,
                       'candidate_models':{'V3':['log1p(origin_candidate_count)','relative_semantic_score','origin_rank'],
                                           'V4':['log1p(origin_candidate_count)','relative_semantic_score','origin_rank','abs(anchor_context_score_difference)']}},
            'tuning':{'selection_on_tune_only':True,'safety_constraint':'mean S@32 harm <= 0.5pp','grid':grid,'selected':selected},
            'test':{'local':{'S16':l16,'S32':l32},'policy':{'model':name,'margin':margin,'S16':p16,'S32':p32,'delta_S16':d16,'delta_S32':d32,
                    'target_cluster_CI95_S16':ci16,'target_cluster_CI95_S32':ci32,'target_bucket_S16':b16,'target_bucket_S32':b32,'intervention_rate':intr}},
            'boundary':['All decision features are available from the current or immediately previous page.','Policy trajectories remain simulated, not human comprehension outcomes.','This phase was preregistered before reading AP-RS5 results.']}
    (out/'AP_RS6_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'test_rows.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=test[0].keys()); w.writeheader(); w.writerows(test)
    (out/'AP_RS6_SUMMARY.md').write_text(f"# AP-RS6 — Visible-only real anchor/context policy\n\n**Decision: {decision}**\n\n- selected: {name}, margin {margin}\n- test tasks: {len(test)}\n- local S@16/S@32: {l16:.4f}/{l32:.4f}\n- policy S@16/S@32: {p16:.4f}/{p32:.4f}\n- delta S@16: {100*d16:+.3f} pp, CI {ci16}\n- delta S@32: {100*d32:+.3f} pp, CI {ci32}\n- positive S@16 buckets: {b16['positive']}/8\n- intervention rate: {intr:.4f}\n",encoding='utf-8')
    print('AP_RS6_DECISION',decision); print('AP_RS6_S16',d16,ci16,b16); print('AP_RS6_S32',d32,ci32,b32)

if __name__=='__main__': main()
