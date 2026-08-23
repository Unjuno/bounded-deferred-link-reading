#!/usr/bin/env python3
"""AP-RS3: complete-coverage MiniLM title-semantic replication on Wikispeedia.

Recomputes the same SentenceTransformer family used by the ADASpeedia source
(`all-MiniLM-L6-v2`) for every title appearing in the unique finished paths,
removing the pair-coverage limitation of AP-RS0/RS2.

RS3A: nonterminal semantic progress along real human paths.
RS3B: first nonterminal BACK replacement vs abandoned page.
"""
from __future__ import annotations
import json, math, os, urllib.parse, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

REF='69052d52bbbfe57ed25e9bccbd36a5acbc0f988d'
PATHS_URL=f'https://raw.githubusercontent.com/epfl-ada/ada-2024-project-adaspeedia/{REF}/data/paths_finished_unique.tsv'
N_BOOT=2000
SEED=20260825
MIN_PROGRESS=400
MIN_BACK=300

def dec(x): return urllib.parse.unquote(str(x))
def dl(url,p):
    if not p.exists() or p.stat().st_size==0: urllib.request.urlretrieve(url,p)

def reconstruct(tokens):
    stack=[]; states=[]
    for raw in tokens:
        t=dec(raw)
        if t=='<':
            if len(stack)>1:
                stack.pop(); states.append(stack[-1])
        else:
            stack.append(t); states.append(t)
    return states

def back_episodes(tokens):
    stack=[]; active=None; out=[]
    for raw in tokens:
        t=dec(raw)
        if t=='<':
            if not stack: continue
            if active is None: active={'abandoned':stack[-1], 'n_back':0}
            if len(stack)>1: stack.pop(); active['n_back']+=1
            continue
        if active is not None:
            out.append((active['abandoned'], stack[-1] if stack else None, t, active['n_back']))
            active=None
        stack.append(t)
    return out

def cboot(df,col,cluster,off=0):
    g=df.groupby(cluster,dropna=False)[col].agg(['sum','count'])
    sums=g['sum'].to_numpy(float); counts=g['count'].to_numpy(float)
    rng=np.random.default_rng(SEED+off); reps=np.empty(N_BOOT)
    for b in range(N_BOOT):
        ii=rng.integers(0,len(g),len(g))
        reps[b]=sums[ii].sum()/counts[ii].sum()
    return [float(np.quantile(reps,.025)),float(np.quantile(reps,.975))]

def stat(df,col):
    x=df[col].to_numpy(float)
    return {'n':int(len(x)),'mean':float(x.mean()),'median':float(np.median(x)),
            'sd':float(x.std(ddof=1)),'u_c':float(x.std(ddof=1)/math.sqrt(len(x))),
            'positive_fraction':float(np.mean(x>0)),'zero_fraction':float(np.mean(x==0)),
            'user_clusters':int(df.user.nunique()),'target_clusters':int(df.target.nunique()),
            'user_cluster_ci95':cboot(df,col,'user',0),'target_cluster_ci95':cboot(df,col,'target',31)}

def main():
    out=Path(os.environ.get('AP_RS3_OUT','artifacts/ap_rs3')); out.mkdir(parents=True,exist_ok=True)
    raw=out/'raw'; raw.mkdir(exist_ok=True); pf=raw/'paths.tsv'; dl(PATHS_URL,pf)
    paths=pd.read_csv(pf,sep='\t')
    parsed=[]; titles=set()
    for _,r in paths.iterrows():
        toks=str(r.path).split(';'); arts=[dec(t) for t in toks if dec(t)!='<']
        if len(arts)<2: continue
        parsed.append((r,toks,arts)); titles.update(arts)
    titles=sorted(titles); print('unique titles',len(titles),'paths',len(parsed))
    model=SentenceTransformer('all-MiniLM-L6-v2')
    emb=model.encode(titles,batch_size=128,show_progress_bar=True,normalize_embeddings=True)
    idx={t:i for i,t in enumerate(titles)}
    def sim(a,b): return float(np.dot(emb[idx[a]],emb[idx[b]]))

    prog=[]; backs=[]
    for r,toks,arts in parsed:
        target=arts[-1]; source=arts[0]; user=str(r.get('hashedIpAddress','NA')); pid=r.get('path_id',0)
        states=reconstruct(toks)
        if len(states)>=3:
            ds=[sim(b,target)-sim(a,target) for a,b in zip(states[:-2],states[1:-1])]
            if ds:
                prog.append({'path_id':pid,'user':user,'source':source,'target':target,
                             'mean_delta':float(np.mean(ds)),'positive_step_fraction':float(np.mean(np.array(ds)>0)),
                             'n_transitions':len(ds),'had_backtrack':'<' in [dec(t) for t in toks]})
        # first eligible nonterminal replacement episode only
        for ei,(abandoned,returned,replacement,nback) in enumerate(back_episodes(toks)):
            if returned is None or replacement==target: continue
            backs.append({'path_id':pid,'user':user,'source':source,'target':target,'episode_index':ei,
                          'return_delta':sim(returned,target)-sim(abandoned,target),
                          'replacement_delta':sim(replacement,target)-sim(abandoned,target),
                          'replacement_vs_return_delta':sim(replacement,target)-sim(returned,target),
                          'n_back':nback})
            break
    prog=pd.DataFrame(prog); backs=pd.DataFrame(backs)
    prog.to_csv(out/'path_progress.csv',index=False); backs.to_csv(out/'first_backtrack_replacement.csv',index=False)
    ps=stat(prog,'mean_delta')
    pc={'n_ge_400':ps['n']>=MIN_PROGRESS,'mean_gt_0':ps['mean']>0,
        'user_ci_lower_gt_0':ps['user_cluster_ci95'][0]>0,'target_ci_lower_gt_0':ps['target_cluster_ci95'][0]>0,
        'positive_paths_gt_half':ps['positive_fraction']>0.5}
    if len(backs):
        bs=stat(backs,'replacement_delta'); rs=stat(backs,'return_delta'); br=stat(backs,'replacement_vs_return_delta')
        bc={'n_ge_300':bs['n']>=MIN_BACK,'mean_gt_0':bs['mean']>0,
            'user_ci_lower_gt_0':bs['user_cluster_ci95'][0]>0,'target_ci_lower_gt_0':bs['target_cluster_ci95'][0]>0,
            'positive_paths_gt_half':bs['positive_fraction']>0.5}
    else:
        bs=rs=br={}; bc={'n_ge_300':False,'mean_gt_0':False,'user_ci_lower_gt_0':False,'target_ci_lower_gt_0':False,'positive_paths_gt_half':False}
    result={'phase':'AP-RS3','encoder':'sentence-transformers/all-MiniLM-L6-v2','complete_title_coverage':True,
            'construct':'real human Wikispeedia navigation + article-title semantics; not body/anchor/context semantics',
            'RS3A_progress':{'decision':'PASS' if all(pc.values()) else 'FAIL','conditions':pc,'stats':ps,
                             'terminal_transition_excluded':True},
            'RS3B_backtrack_replacement':{'decision':'PASS' if all(bc.values()) else 'FAIL','conditions':bc,
                                           'replacement_vs_abandoned':bs,'return_vs_abandoned':rs,
                                           'replacement_vs_returned_ancestor':br,
                                           'terminal_replacement_excluded':True,'first_eligible_episode_only':True},
            'boundary':['Complete title-semantic coverage removes AP-RS0/RS2 pair-table missingness.',
                        'A PASS still would not validate article-body anchor/context semantics or human comprehension.']}
    (out/'AP_RS3_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    md=f"""# AP-RS3 — Complete-coverage real-title semantic replication\n\n## RS3A path progress: {result['RS3A_progress']['decision']}\n- n: {ps['n']}\n- mean: {ps['mean']:.6f}\n- positive paths: {ps['positive_fraction']:.4f}\n- user CI: {ps['user_cluster_ci95']}\n- target CI: {ps['target_cluster_ci95']}\n\n## RS3B first nonterminal BACK replacement: {result['RS3B_backtrack_replacement']['decision']}\n- n: {bs.get('n')}\n- replacement-abandoned mean: {bs.get('mean')}\n- positive paths: {bs.get('positive_fraction')}\n- user CI: {bs.get('user_cluster_ci95')}\n- target CI: {bs.get('target_cluster_ci95')}\n\n## Boundary\nReal human paths and full MiniLM title embeddings, but still title semantics only.\n"""
    (out/'AP_RS3_SUMMARY.md').write_text(md,encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
