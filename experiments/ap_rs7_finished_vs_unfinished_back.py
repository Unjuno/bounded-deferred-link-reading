#!/usr/bin/env python3
"""AP-RS7: compare human BACK semantic correction in finished vs unfinished Wikispeedia paths.

Uses actual article-body semantics. Primary analysis is target-matched: for each target
with eligible BACK episodes in both outcome groups, compare the mean replacement-vs-
abandoned semantic correction. Target-level bootstrap is the primary uncertainty unit.
Observational only; it tests whether branch correction is associated with eventual
navigation success, not whether BACK causally produces success.
"""
from __future__ import annotations
import json, os, tarfile, math
from pathlib import Path
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from experiments.ap_rs5_real_anchor_context_policy import GRAPH_URL, download, noncomment_rows, find_file
from experiments.ap_rs4_article_body_semantics import SNAP_TEXT_URL, load_article_texts, norm_title, dec

SEED=20260826
N_BOOT=4000
MIN_EPISODES=300
MIN_COMMON_TARGETS=100
MAX_CHARS=6000


def first_back(tokens,target):
    stack=[]; active=None
    for raw in tokens:
        t=dec(raw)
        if t=='<':
            if not stack: continue
            if active is None: active={'abandoned':stack[-1],'n_back':0}
            if len(stack)>1: stack.pop(); active['n_back']+=1
            continue
        if active is not None:
            returned=stack[-1] if stack else None
            replacement=t
            if returned is not None and replacement!=target:
                return active['abandoned'],returned,replacement,active['n_back']
            active=None
        stack.append(t)
    return None


def boot_mean(vals,seed):
    vals=np.asarray(vals,float); rng=np.random.default_rng(seed); reps=np.empty(N_BOOT)
    for b in range(N_BOOT): reps[b]=rng.choice(vals,size=len(vals),replace=True).mean()
    return [float(np.quantile(reps,.025)),float(np.quantile(reps,.975))],float(reps.std(ddof=1))


def main():
    out=Path(os.environ.get('AP_RS7_OUT','artifacts/ap_rs7')); out.mkdir(parents=True,exist_ok=True)
    raw=out/'raw'; raw.mkdir(exist_ok=True)
    graph_tar=raw/'wikispeedia_paths-and-graph.tar.gz'; text_tar=raw/'wikispeedia_articles_plaintext.tar.gz'
    download(GRAPH_URL,graph_tar); download(SNAP_TEXT_URL,text_tar)
    graph_root=raw/'graph'
    if not graph_root.exists() or not any(graph_root.iterdir()):
        graph_root.mkdir(exist_ok=True)
        with tarfile.open(graph_tar,'r:gz') as tf: tf.extractall(graph_root)
    articles=[dec(r[0]) for r in noncomment_rows(find_file(graph_root,'articles.tsv')) if r]
    finished_file=find_file(graph_root,'paths_finished.tsv'); unfinished_file=find_file(graph_root,'paths_unfinished.tsv')
    text_root=raw/'plaintext'
    if not text_root.exists() or not any(text_root.iterdir()):
        text_root.mkdir(exist_ok=True)
        with tarfile.open(text_tar,'r:gz') as tf: tf.extractall(text_root)
    texts=load_article_texts(text_root,set(articles)); coverage=len(texts)/len(articles)
    if coverage<.95: raise RuntimeError(f'body coverage too low {coverage:.3f}')
    titles=sorted(texts)
    enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    emb=enc.encode([texts[t][:MAX_CHARS] for t in titles],batch_size=64,show_progress_bar=True,normalize_embeddings=True)
    idx={t:i for i,t in enumerate(titles)}
    def sim(a,b): return float(np.dot(emb[idx[a]],emb[idx[b]]))

    rows=[]
    def add_group(pathfile,outcome):
        for r in noncomment_rows(pathfile):
            if outcome=='finished':
                if len(r)<4: continue
                user=r[0]; toks=[dec(x) for x in r[3].split(';')]; arts=[x for x in toks if x!='<']
                if len(arts)<2: continue
                target=arts[-1]
            else:
                if len(r)<5: continue
                user=r[0]; toks=[dec(x) for x in r[3].split(';')]; target=dec(r[4])
            ep=first_back(toks,target)
            if ep is None: continue
            abandoned,returned,replacement,n_back=ep
            if any(x not in idx for x in (abandoned,returned,replacement,target)): continue
            rows.append({'outcome':outcome,'user':user,'target':target,'n_back':n_back,
                         'return_delta':sim(returned,target)-sim(abandoned,target),
                         'replacement_delta':sim(replacement,target)-sim(abandoned,target),
                         'replacement_vs_return':sim(replacement,target)-sim(returned,target)})
    add_group(finished_file,'finished'); add_group(unfinished_file,'unfinished')
    print('episodes',len(rows))
    by=defaultdict(lambda:{'finished':[],'unfinished':[]})
    by_back=defaultdict(lambda:{'finished':[],'unfinished':[]})
    for r in rows:
        by[r['target']][r['outcome']].append(r['replacement_delta'])
        by_back[r['target']][r['outcome']].append(float(r['n_back']==1))
    diffs=[]; backdiffs=[]; target_counts=[]
    for t,g in by.items():
        if g['finished'] and g['unfinished']:
            diffs.append(float(np.mean(g['finished'])-np.mean(g['unfinished'])))
            backdiffs.append(float(np.mean(by_back[t]['finished'])-np.mean(by_back[t]['unfinished'])))
            target_counts.append({'target':t,'n_finished':len(g['finished']),'n_unfinished':len(g['unfinished']),
                                  'correction_diff':diffs[-1],'one_step_rate_diff':backdiffs[-1]})
    fin=[r for r in rows if r['outcome']=='finished']; unf=[r for r in rows if r['outcome']=='unfinished']
    ci,sd=boot_mean(diffs,SEED); ci_b,sd_b=boot_mean(backdiffs,SEED+41)
    effect=float(np.mean(diffs)); back_effect=float(np.mean(backdiffs))
    def desc(xs):
        v=np.asarray([r['replacement_delta'] for r in xs],float)
        return {'n':len(xs),'mean':float(v.mean()),'median':float(np.median(v)),'positive_fraction':float(np.mean(v>0)),
                'one_step_fraction':float(np.mean([r['n_back']==1 for r in xs]))}
    conditions={'finished_n_ge_300':len(fin)>=MIN_EPISODES,'unfinished_n_ge_300':len(unf)>=MIN_EPISODES,
                'common_targets_ge_100':len(diffs)>=MIN_COMMON_TARGETS,'target_matched_correction_diff_gt_0':effect>0,
                'target_bootstrap_CI_lower_gt_0':ci[0]>0}
    decision='PASS' if all(conditions.values()) else 'FAIL'
    result={'phase':'AP-RS7','name':'finished vs unfinished human BACK body-semantic correction','decision':decision,
            'preregistered_conditions':conditions,'data':{'article_body_coverage':coverage,'finished_episodes':len(fin),
            'unfinished_episodes':len(unf),'common_targets':len(diffs)},'finished':desc(fin),'unfinished':desc(unf),
            'target_matched':{'mean_finished_minus_unfinished_replacement_delta':effect,'target_bootstrap_CI95':ci,
                              'u_target_bootstrap':sd,'mean_one_step_rate_difference':back_effect,
                              'one_step_target_bootstrap_CI95':ci_b,'u_one_step_target_bootstrap':sd_b},
            'boundary':['Observational comparison; success status is not randomized.','Primary inference uses targets represented in both finished and unfinished BACK episodes.','Article-body semantics are stronger than titles but still a navigation proxy, not comprehension.']}
    (out/'AP_RS7_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'AP_RS7_SUMMARY.md').write_text(f"# AP-RS7 — Finished vs unfinished human BACK\n\n**Decision: {decision}**\n\n- finished episodes: {len(fin)}\n- unfinished episodes: {len(unf)}\n- common targets: {len(diffs)}\n- target-matched correction difference: {effect:+.6f}, CI {ci}\n- finished one-step fraction: {result['finished']['one_step_fraction']:.4f}\n- unfinished one-step fraction: {result['unfinished']['one_step_fraction']:.4f}\n- target-matched one-step-rate difference: {back_effect:+.6f}, CI {ci_b}\n",encoding='utf-8')
    print('AP_RS7_DECISION',decision); print('CORRECTION_DIFF',effect,ci); print('ONE_STEP_DIFF',back_effect,ci_b)

if __name__=='__main__': main()
