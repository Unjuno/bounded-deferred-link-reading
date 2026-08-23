#!/usr/bin/env python3
"""AP-RS1/RS2: semantic structure of real human Wikispeedia BACK episodes.

AP-RS1 (diagnostic): does the BACK action itself return to a title-semantically
better state relative to the abandoned page?

AP-RS2 (primary mechanism gate): after a BACK episode, is the first replacement
article title-semantically closer to the target than the abandoned page?

To avoid endpoint leakage, episodes whose replacement is the target are excluded
from the primary analysis. Only the first eligible BACK-replacement episode per
unique source-target path is used as the primary independent observation.
"""
from __future__ import annotations
import ast, json, math, os, urllib.parse, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

REF="69052d52bbbfe57ed25e9bccbd36a5acbc0f988d"
BASE=f"https://raw.githubusercontent.com/epfl-ada/ada-2024-project-adaspeedia/{REF}/data"
PATHS_URL=f"{BASE}/paths_finished_unique.tsv"
SIM_URL=f"{BASE}/article_similarities.csv"
N_BOOT=2000
BOOT_SEED=20260824
MIN_BACKTRACK_PATHS=300

def dl(url,p):
    if not p.exists() or p.stat().st_size==0:
        urllib.request.urlretrieve(url,p)

def dec(x): return urllib.parse.unquote(str(x))

def pair(x):
    try:
        p=ast.literal_eval(x)
        if isinstance(p,(tuple,list)) and len(p)==2: return dec(p[0]),dec(p[1])
    except Exception: pass
    return None

def episodes(tokens):
    """Yield (abandoned, returned_ancestor, replacement, n_back) episodes."""
    stack=[]
    active=None
    out=[]
    for raw in tokens:
        t=dec(raw)
        if t=='<':
            if not stack: continue
            if active is None:
                active={'abandoned':stack[-1], 'n_back':0}
            if len(stack)>1:
                stack.pop()
                active['n_back'] += 1
            continue
        # article token
        if active is not None:
            returned=stack[-1] if stack else None
            out.append((active['abandoned'], returned, t, active['n_back']))
            active=None
        stack.append(t)
    return out

def cluster_ci(df,col,cluster,seedoff=0):
    groups=[g[col].to_numpy(float) for _,g in df.groupby(cluster,dropna=False)]
    rng=np.random.default_rng(BOOT_SEED+seedoff)
    reps=[]
    for _ in range(N_BOOT):
        ids=rng.integers(0,len(groups),len(groups))
        reps.append(float(np.mean(np.concatenate([groups[i] for i in ids]))))
    return [float(np.quantile(reps,.025)),float(np.quantile(reps,.975))]

def stats(df,col):
    x=df[col].to_numpy(float)
    return {
        'n':int(len(x)), 'mean':float(x.mean()), 'median':float(np.median(x)),
        'sd':float(x.std(ddof=1)), 'u_c':float(x.std(ddof=1)/math.sqrt(len(x))),
        'positive_fraction':float(np.mean(x>0)), 'zero_fraction':float(np.mean(x==0)),
        'user_clusters':int(df.user.nunique()), 'target_clusters':int(df.target.nunique()),
        'user_cluster_ci95':cluster_ci(df,col,'user',0),
        'target_cluster_ci95':cluster_ci(df,col,'target',23),
    }

def main():
    out=Path(os.environ.get('AP_RS12_OUT','artifacts/ap_rs1_rs2')); out.mkdir(parents=True,exist_ok=True)
    raw=out/'raw'; raw.mkdir(exist_ok=True)
    pf=raw/'paths.tsv'; sf=raw/'sim.csv'; dl(PATHS_URL,pf); dl(SIM_URL,sf)
    paths=pd.read_csv(pf,sep='\t'); sims=pd.read_csv(sf)
    sims['pp']=sims['pair'].map(pair); sims=sims[sims.pp.notna()].copy()
    maps={
      'sbert':dict(zip(sims.pp,pd.to_numeric(sims.sbert_cosine_similarity,errors='coerce'))),
      'bert':dict(zip(sims.pp,pd.to_numeric(sims.cosine_similarity,errors='coerce'))),
    }
    rows=[]
    for _,r in paths.iterrows():
        toks=str(r.path).split(';')
        arts=[dec(t) for t in toks if dec(t)!='<']
        if len(arts)<2: continue
        target=arts[-1]; source=arts[0]
        eps=episodes(toks)
        for ei,(abandoned,returned,replacement,nback) in enumerate(eps):
            if returned is None: continue
            for ch,sm in maps.items():
                va=sm.get((target,abandoned)); vr=sm.get((target,returned)); vn=sm.get((target,replacement))
                if any(v is None or not np.isfinite(v) for v in [va,vr,vn]): continue
                rows.append({
                  'path_id':r.get('path_id',0),'user':str(r.get('hashedIpAddress','NA')),
                  'source':source,'target':target,'episode_index':ei,'channel':ch,
                  'n_back':nback,'abandoned':abandoned,'returned':returned,'replacement':replacement,
                  'replacement_is_target':replacement==target,
                  'return_delta':float(vr-va),
                  'replacement_delta':float(vn-va),
                  'replacement_vs_return_delta':float(vn-vr),
                })
    ev=pd.DataFrame(rows); ev.to_csv(out/'backtrack_episodes.csv',index=False)

    results={}
    for ch in ['sbert','bert']:
        d=ev[(ev.channel==ch)&(~ev.replacement_is_target)].copy()
        # Primary independence: first eligible episode per unique source-target path.
        first=d.sort_values(['path_id','episode_index']).drop_duplicates('path_id',keep='first')
        if len(first)==0: continue
        rs1=stats(first,'return_delta')
        rs2=stats(first,'replacement_delta')
        rs2_conditions={
          'n_ge_300':rs2['n']>=MIN_BACKTRACK_PATHS,
          'mean_gt_0':rs2['mean']>0,
          'user_ci_lower_gt_0':rs2['user_cluster_ci95'][0]>0,
          'target_ci_lower_gt_0':rs2['target_cluster_ci95'][0]>0,
          'positive_paths_gt_half':rs2['positive_fraction']>0.5,
        }
        results[ch]={
          'first_nonterminal_backtrack_episode':{
            'AP_RS1_return_action':rs1,
            'AP_RS2_replacement_vs_abandoned':rs2,
            'AP_RS2_conditions':rs2_conditions,
            'AP_RS2_decision':'PASS' if all(rs2_conditions.values()) else 'FAIL',
            'replacement_vs_return':stats(first,'replacement_vs_return_delta'),
            'mean_back_steps':float(first.n_back.mean()),
          },
          'all_nonterminal_episodes_descriptive':{
             'n_episodes':int(len(d)),
             'n_paths':int(d.path_id.nunique()),
             'replacement_delta_mean':float(d.replacement_delta.mean()),
             'return_delta_mean':float(d.return_delta.mean()),
          }
        }
    primary=results.get('sbert',{}).get('first_nonterminal_backtrack_episode',{})
    result={
      'phase':'AP-RS1/RS2',
      'name':'real human Wikispeedia BACK semantic correction gate',
      'primary_channel':'SBERT article-title cosine',
      'primary_AP_RS2_decision':primary.get('AP_RS2_decision','NO_DATA'),
      'construct':'real human BACK behavior + article-title semantics; NOT body/anchor/context semantics',
      'terminal_replacement_excluded':True,
      'one_episode_per_unique_source_target_path_primary':True,
      'results':results,
      'interpretation_boundary':[
        'RS1 tests whether the BACK action itself moves toward a title-semantically closer page.',
        'RS2 tests whether the replacement chosen after BACK is title-semantically closer to target than the abandoned page.',
        'Neither establishes causal benefit of deferred-link reading or human comprehension.'
      ]
    }
    (out/'AP_RS1_RS2_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    p=primary
    rs1=p.get('AP_RS1_return_action',{}); rs2=p.get('AP_RS2_replacement_vs_abandoned',{})
    md=f"""# AP-RS1/RS2 — Human Wikispeedia BACK semantics\n\n**AP-RS2 primary decision: {result['primary_AP_RS2_decision']}**\n\nPrimary: SBERT title cosine; first nonterminal BACK-replacement episode per unique source-target path.\n\n## AP-RS1: BACK return vs abandoned page\n- n: {rs1.get('n')}\n- mean delta: {rs1.get('mean')}\n- positive fraction: {rs1.get('positive_fraction')}\n- user-cluster CI: {rs1.get('user_cluster_ci95')}\n- target-cluster CI: {rs1.get('target_cluster_ci95')}\n\n## AP-RS2: replacement after BACK vs abandoned page\n- n: {rs2.get('n')}\n- mean delta: {rs2.get('mean')}\n- positive fraction: {rs2.get('positive_fraction')}\n- user-cluster CI: {rs2.get('user_cluster_ci95')}\n- target-cluster CI: {rs2.get('target_cluster_ci95')}\n\n## Boundary\nReal human navigation, but title semantics only. This is not the final anchor/context or human-comprehension validation.\n"""
    (out/'AP_RS1_RS2_SUMMARY.md').write_text(md,encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
