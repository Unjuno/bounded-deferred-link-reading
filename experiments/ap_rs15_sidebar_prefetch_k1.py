#!/usr/bin/env python3
"""AP-RS15: sidebar-agent K=1 deferred-destination prefetch test.

Preregistered in docs/PREREG_AP_RS15.md before this file was added.
The local policy remains anchor-only. The trigger gets exactly two additional
features derived from the unvisited rank-2 destination's prefetched article body.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tarfile
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
    GRAPH_URL, HTML_URL, build_visible_link_corpus, download, extract_graph,
)
from experiments.ap_rs4_article_body_semantics import SNAP_TEXT_URL, load_article_texts

SEED = 20260903
FIT_N = 1600
CAL_N = 1000
TEST_N = 2600
PRIMARY_BUDGET = 16
N_BOOT = 2000
MAX_CHARS = 6000
MARGINS = [-0.05, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20, 999.0]


def choose_tasks(missions, target_set, n, rng):
    xs = [x for x in missions if x[1] in target_set]
    rng.shuffle(xs)
    return xs[:min(n, len(xs))]


def target_cluster_ci(rows, key, offset=0):
    by = defaultdict(list)
    for r in rows:
        by[r["target"]].append(r[key])
    groups = [(sum(v), len(v)) for v in by.values()]
    rng = np.random.default_rng(SEED + offset)
    reps = np.empty(N_BOOT)
    for b in range(N_BOOT):
        ii = rng.integers(0, len(groups), len(groups))
        reps[b] = sum(groups[i][0] for i in ii) / sum(groups[i][1] for i in ii)
    return [float(np.quantile(reps, .025)), float(np.quantile(reps, .975))]


def bucket_stats(rows, key):
    vals = [[] for _ in range(8)]
    for r in rows:
        b = int(hashlib.md5(r["target"].encode()).hexdigest()[:8], 16) % 8
        vals[b].append(r[key])
    means = [float(np.mean(v)) if v else float("nan") for v in vals]
    finite = np.asarray([x for x in means if np.isfinite(x)], float)
    uc = float(finite.std(ddof=1) / math.sqrt(len(finite))) if len(finite) > 1 else float("nan")
    return {"means": means, "positive": int(sum(x > 0 for x in finite)),
            "u_c": uc, "U_k1.96": float(1.96 * uc) if np.isfinite(uc) else None}


def fold_id(target):
    return int(hashlib.sha256(target.encode()).hexdigest()[:8], 16) % 4


def main():
    out = Path(os.environ.get("AP_RS15_OUT", "artifacts/ap_rs15"))
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"; raw.mkdir(exist_ok=True)
    graph_tar = raw / "graph.tar.gz"
    html_tar = raw / "html.tar.gz"
    text_tar = raw / "plaintext.tar.gz"
    download(GRAPH_URL, graph_tar)
    download(HTML_URL, html_tar)
    download(SNAP_TEXT_URL, text_tar)

    articles, links, missions = extract_graph(graph_tar, raw / "graph")
    idx = {a: i for i, a in enumerate(articles)}
    occ, anchors, _contexts, edge_coverage = build_visible_link_corpus(html_tar, articles, links)
    if edge_coverage < .85:
        raise RuntimeError(f"visible edge coverage too low: {edge_coverage:.3f}")

    text_root = raw / "plaintext"
    if not text_root.exists() or not any(text_root.iterdir()):
        text_root.mkdir(exist_ok=True)
        with tarfile.open(text_tar, "r:gz") as tf:
            tf.extractall(text_root)
    texts = load_article_texts(text_root, set(articles))
    body_coverage = len(texts) / max(1, len(articles))
    if body_coverage < .90:
        raise RuntimeError(f"article-body coverage too low: {body_coverage:.3f}")

    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    title_emb = enc.encode([a.replace("_", " ") for a in articles], batch_size=128,
                           show_progress_bar=True, normalize_embeddings=True)
    anchor_emb = enc.encode(anchors, batch_size=128, show_progress_bar=True,
                            normalize_embeddings=True)
    body_texts = [texts.get(a, a.replace("_", " "))[:MAX_CHARS] for a in articles]
    body_emb = enc.encode(body_texts, batch_size=64, show_progress_bar=True,
                          normalize_embeddings=True)

    @lru_cache(maxsize=60000)
    def scored(src, goal):
        gv = title_emb[goal]
        best = {}
        for v, ai, _ci in occ[src]:
            sa = float(np.dot(anchor_emb[ai], gv))
            if v not in best or sa > best[v]:
                best[v] = sa
        return tuple(sorted(best.items(), key=lambda z: (-z[1], z[0])))

    def available(src, goal, seen):
        return [(v, s) for v, s in scored(src, goal) if v not in seen]

    @lru_cache(maxsize=60000)
    def body_sim(page, goal):
        return float(np.dot(body_emb[page], title_emb[goal]))

    def local_rollout(start, goal, steps, budget, seen):
        cur = start
        ss = set(seen)
        if cur == goal:
            return True
        while steps < budget:
            xs = available(cur, goal, ss)
            if not xs:
                return False
            cur = xs[0][0]
            steps += 1
            if cur == goal:
                return True
            ss.add(cur)
        return False

    def state_record(current, goal, steps, seen, origin_opts, alt):
        cur_opts = available(current, goal, seen)
        cur_top1 = cur_opts[0][1] if cur_opts else 0.0
        alt_body = body_sim(alt[0], goal)  # sidebar prefetch: unvisited deferred destination
        current_body = body_sim(current, goal)  # current page has already been visited
        return {
            "origin_count": len(origin_opts),
            "rel_alt_cur": alt[1] - cur_top1,
            "step_frac": steps / PRIMARY_BUDGET,
            "alt_body": alt_body,
            "alt_body_minus_current": alt_body - current_body,
        }

    def f_visible(r):
        return [math.log1p(r["origin_count"]), r["rel_alt_cur"], r["step_frac"]]

    def f_prefetch(r):
        return f_visible(r) + [r["alt_body"], r["alt_body_minus_current"]]

    def build_teacher(tasks):
        rows = []
        for qi, (source, target) in enumerate(tasks):
            cur = idx[source]; goal = idx[target]; seen = {cur}; steps = 0
            while steps < PRIMARY_BUDGET and cur != goal:
                opts = available(cur, goal, seen)
                if not opts: break
                chosen = opts[0]
                alt = opts[1] if len(opts) >= 2 else None
                cur = chosen[0]; steps += 1
                if cur == goal: break
                seen.add(cur)
                if alt is not None and alt[0] not in seen and steps + 2 <= PRIMARY_BUDGET:
                    rec = state_record(cur, goal, steps, seen, opts, alt)
                    cont = local_rollout(cur, goal, steps, PRIMARY_BUDGET, seen)
                    back = local_rollout(alt[0], goal, steps + 2, PRIMARY_BUDGET, seen | {alt[0]})
                    rec.update({"y": float(back)-float(cont), "target": target})
                    rows.append(rec)
            if qi % 400 == 0: print("teacher", qi, "rows", len(rows))
        return rows

    rng = np.random.default_rng(SEED)
    targets = sorted({t for s, t in missions if s in idx and t in idx})
    rng.shuffle(targets); nt = len(targets)
    fit_targets = set(targets[:int(.40*nt)])
    cal_targets = set(targets[int(.40*nt):int(.60*nt)])
    test_targets = set(targets[int(.60*nt):])
    fit_tasks = choose_tasks(missions, fit_targets, FIT_N, rng)
    cal_tasks = choose_tasks(missions, cal_targets, CAL_N, rng)
    test_tasks = choose_tasks(missions, test_targets, TEST_N, rng)
    print("split", len(fit_tasks), len(cal_tasks), len(test_tasks),
          "targets", len(fit_targets), len(cal_targets), len(test_targets))

    fit_teacher = build_teacher(fit_tasks)
    y = np.asarray([r["y"] for r in fit_teacher], float)
    Xp = np.asarray([f_prefetch(r) for r in fit_teacher], float)
    Xv = np.asarray([f_visible(r) for r in fit_teacher], float)
    prefetch_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xp, y)
    visible_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xv, y)
    non = y != 0
    fit_auc_prefetch = float(roc_auc_score((y[non] > 0).astype(int), prefetch_model.predict(Xp)[non])) if non.sum() else None
    fit_auc_visible = float(roc_auc_score((y[non] > 0).astype(int), visible_model.predict(Xv)[non])) if non.sum() else None

    def run_policy(task, budget, margin=None, diagnostic=False):
        source, target = task
        cur = idx[source]; goal = idx[target]; seen = {cur}; steps = 0; used = False
        trig_y = None; trig_pred = None
        if cur == goal: return True, used, trig_y, trig_pred
        while steps < budget:
            opts = available(cur, goal, seen)
            if not opts: return False, used, trig_y, trig_pred
            chosen = opts[0]; alt = opts[1] if len(opts) >= 2 else None
            cur = chosen[0]; steps += 1
            if cur == goal: return True, used, trig_y, trig_pred
            seen.add(cur)
            if margin is not None and not used and alt is not None and alt[0] not in seen \
                    and steps < PRIMARY_BUDGET and steps + 2 <= budget:
                rec = state_record(cur, goal, steps, seen, opts, alt)
                p = float(prefetch_model.predict(np.asarray([f_prefetch(rec)], float))[0])
                if p > margin:
                    if diagnostic and budget == PRIMARY_BUDGET:
                        cont = local_rollout(cur, goal, steps, PRIMARY_BUDGET, seen)
                        back = local_rollout(alt[0], goal, steps+2, PRIMARY_BUDGET, seen | {alt[0]})
                        trig_y = float(back)-float(cont); trig_pred = p
                    cur = alt[0]; steps += 2; used = True
                    if cur == goal: return True, used, trig_y, trig_pred
                    seen.add(cur)
        return False, used, trig_y, trig_pred

    def eval_tasks(tasks, margin, diagnostic=False):
        rows = []
        for source, target in tasks:
            task = (source, target)
            l16 = run_policy(task, 16)[0]; l32 = run_policy(task, 32)[0]
            p16, u16, ty, tp = run_policy(task, 16, margin, diagnostic)
            p32, u32, _, _ = run_policy(task, 32, margin, False)
            rows.append({"source": source, "target": target,
                         "local16": int(l16), "local32": int(l32),
                         "policy16": int(p16), "policy32": int(p32),
                         "d16": int(p16)-int(l16), "d32": int(p32)-int(l32),
                         "intervened": int(u16 or u32), "trigger_y16": ty, "trigger_pred": tp})
        return rows

    grid = []; eligible = []
    for margin in MARGINS:
        rr = eval_tasks(cal_tasks, margin)
        d16 = float(np.mean([r["d16"] for r in rr])); d32 = float(np.mean([r["d32"] for r in rr]))
        intr = float(np.mean([r["intervened"] for r in rr]))
        folds = []
        for f in range(4):
            ff = [r for r in rr if fold_id(r["target"]) == f]
            folds.append({"fold": f, "n": len(ff),
                          "d16": float(np.mean([r["d16"] for r in ff])) if ff else None,
                          "d32": float(np.mean([r["d32"] for r in ff])) if ff else None})
        pos16 = sum(x["d16"] is not None and x["d16"] > 0 for x in folds)
        safe32 = all(x["d32"] is not None and x["d32"] >= -.005 for x in folds)
        ok = intr <= .25 and d16 > 0 and pos16 >= 3 and safe32
        row = {"margin": margin, "d16": d16, "d32": d32, "intervention_rate": intr,
               "folds": folds, "eligible": ok}
        grid.append(row)
        if ok: eligible.append(row)
    selected = max(eligible, key=lambda r:(r["d16"], -r["intervention_rate"], r["margin"])) \
        if eligible else next(r for r in grid if r["margin"] == 999.0)
    margin = selected["margin"]
    print("SELECTED", json.dumps(selected))

    test_rows = eval_tasks(test_tasks, margin, diagnostic=True)
    d16 = float(np.mean([r["d16"] for r in test_rows])); d32 = float(np.mean([r["d32"] for r in test_rows]))
    ci16 = target_cluster_ci(test_rows, "d16", 1); ci32 = target_cluster_ci(test_rows, "d32", 37)
    b16 = bucket_stats(test_rows, "d16"); b32 = bucket_stats(test_rows, "d32")
    l16 = float(np.mean([r["local16"] for r in test_rows])); l32 = float(np.mean([r["local32"] for r in test_rows]))
    p16 = float(np.mean([r["policy16"] for r in test_rows])); p32 = float(np.mean([r["policy32"] for r in test_rows]))
    intr = float(np.mean([r["intervened"] for r in test_rows]))

    def run_oracle(task, budget):
        source, target = task
        cur = idx[source]; goal = idx[target]; seen = {cur}; steps = 0
        if cur == goal: return True, False
        while steps < budget:
            opts = available(cur, goal, seen)
            if not opts: return False, False
            chosen = opts[0]; alt = opts[1] if len(opts) >= 2 else None
            cur = chosen[0]; steps += 1
            if cur == goal: return True, False
            seen.add(cur)
            if alt is not None and alt[0] not in seen and steps + 2 <= budget:
                cont = local_rollout(cur, goal, steps, budget, seen)
                if cont: return True, False
                if local_rollout(alt[0], goal, steps+2, budget, seen | {alt[0]}): return True, True
        return False, False

    oracle_rows = []
    for source, target in test_tasks:
        task=(source,target); l_16=run_policy(task,16)[0]; l_32=run_policy(task,32)[0]
        o16,u16=run_oracle(task,16); o32,u32=run_oracle(task,32)
        oracle_rows.append({"source":source,"target":target,"d16":int(o16)-int(l_16),
                            "d32":int(o32)-int(l_32),"used16":int(u16),"used32":int(u32)})
    od16=float(np.mean([r["d16"] for r in oracle_rows])); od32=float(np.mean([r["d32"] for r in oracle_rows]))
    oci16=target_cluster_ci(oracle_rows,"d16",101); oci32=target_cluster_ci(oracle_rows,"d32",137)

    test_teacher = build_teacher(test_tasks)
    yt=np.asarray([r["y"] for r in test_teacher],float); ntmask=yt!=0
    ptp=prefetch_model.predict(np.asarray([f_prefetch(r) for r in test_teacher],float))
    ptv=visible_model.predict(np.asarray([f_visible(r) for r in test_teacher],float))
    auc_prefetch=float(roc_auc_score((yt[ntmask]>0).astype(int),ptp[ntmask])) if ntmask.sum() else None
    auc_visible=float(roc_auc_score((yt[ntmask]>0).astype(int),ptv[ntmask])) if ntmask.sum() else None

    triggered=[r for r in test_rows if r["trigger_y16"] is not None]
    trig_diag={"n":len(triggered),
               "help_fraction":float(np.mean([r["trigger_y16"]>0 for r in triggered])) if triggered else None,
               "tie_fraction":float(np.mean([r["trigger_y16"]==0 for r in triggered])) if triggered else None,
               "harm_fraction":float(np.mean([r["trigger_y16"]<0 for r in triggered])) if triggered else None,
               "mean_teacher_utility":float(np.mean([r["trigger_y16"] for r in triggered])) if triggered else None,
               "mean_prediction":float(np.mean([r["trigger_pred"] for r in triggered])) if triggered else None}

    recovery=float(d16/od16) if od16>0 else None
    conditions={"test_n_ge_400":len(test_rows)>=400,"S16_gain_ge_2pp":d16>=.02,
                "S16_target_CI_lower_gt_0":ci16[0]>0,"S16_positive_target_buckets_ge_6_of_8":b16["positive"]>=6,
                "S32_mean_noninferior_minus_0_5pp":d32>=-.005,"S32_target_CI_lower_ge_minus_1pp":ci32[0]>=-.01}
    decision="PASS" if all(conditions.values()) else "FAIL"

    result={"phase":"AP-RS15","name":"sidebar-agent K1 destination-body prefetch trigger","decision":decision,
            "preregistered_conditions":conditions,
            "construct":"real Wikispeedia anchor-local navigation plus one unvisited rank2 destination body preview",
            "data":{"articles":len(articles),"graph_links":len(links),"missions":len(missions),
                    "visible_edge_coverage":edge_coverage,"body_coverage":body_coverage},
            "split":{"seed":SEED,"target_disjoint":True,"fit_tasks":len(fit_tasks),"calibration_tasks":len(cal_tasks),
                     "test_tasks":len(test_tasks),"fit_targets":len(fit_targets),"calibration_targets":len(cal_targets),
                     "test_targets":len(test_targets)},
            "model":{"prefetch_features":["log1p(origin_count)","rank2_anchor_minus_current_top1","step_fraction",
                                           "prefetched_rank2_body_similarity","prefetched_rank2_body_minus_current_body"],
                     "fit_rows":len(fit_teacher),"fit_non_tie":int(non.sum()),
                     "fit_auc_prefetch":fit_auc_prefetch,"fit_auc_visible_V0":fit_auc_visible},
            "calibration":{"grid":grid,"selected":selected,
                           "rule":"intervention<=25%, mean d16>0, >=3/4 d16-positive folds, every-fold d32>=-0.5pp"},
            "test":{"local":{"S16":l16,"S32":l32},
                    "policy":{"margin":margin,"S16":p16,"S32":p32,"delta_S16":d16,"delta_S32":d32,
                              "target_cluster_CI95_S16":ci16,"target_cluster_CI95_S32":ci32,
                              "target_bucket_S16":b16,"target_bucket_S32":b32,"intervention_rate":intr},
                    "oracle_K1":{"delta_S16":od16,"delta_S32":od32,"target_cluster_CI95_S16":oci16,
                                 "target_cluster_CI95_S32":oci32,"recovery_fraction_S16":recovery},
                    "diagnostic":{"heldout_non_tie_auc_prefetch":auc_prefetch,"heldout_non_tie_auc_visible_V0":auc_visible,
                                  "auc_gain_prefetch_minus_visible":None if auc_prefetch is None or auc_visible is None else auc_prefetch-auc_visible,
                                  "triggered_state_teacher":trig_diag}},
            "boundary":["This phase deliberately grants a sidebar agent the unvisited rank-2 destination body prefix.",
                        "No graph distance or rollout oracle is available to the deployed trigger.",
                        "The result is simulated goal-directed navigation, not human comprehension or retention."]}
    (out/"AP_RS15_RESULTS.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    md=f"""# AP-RS15 — Sidebar K1 destination prefetch\n\n**Decision: {decision}**\n\n- selected margin: {margin}\n- test missions: {len(test_rows)}\n- intervention rate: {intr:.4f}\n- S@16 delta: {100*d16:+.3f} pp, CI {ci16}, + buckets {b16['positive']}/8\n- S@32 delta: {100*d32:+.3f} pp, CI {ci32}\n- K1 oracle S@16: {100*od16:+.3f} pp, CI {oci16}\n- held-out AUC visible / prefetch: {auc_visible} / {auc_prefetch}\n- triggered-state diagnostic: {trig_diag}\n- oracle recovery S@16: {recovery}\n"""
    (out/"AP_RS15_SUMMARY.md").write_text(md,encoding="utf-8")
    print("AP_RS15_DECISION",decision)
    print("AP_RS15_SELECTED",json.dumps(selected))
    print("AP_RS15_S16",d16,ci16,b16)
    print("AP_RS15_S32",d32,ci32,b32)
    print("AP_RS15_ORACLE",od16,oci16)
    print("AP_RS15_AUC",auc_visible,auc_prefetch)
    print("AP_RS15_TRIGGER_DIAG",trig_diag)


if __name__ == "__main__":
    main()
