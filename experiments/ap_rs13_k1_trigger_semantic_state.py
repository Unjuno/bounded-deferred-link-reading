#!/usr/bin/env python3
"""AP-RS13: K=1 trigger-only visible semantic-state experiment.

AP-RS12 showed that the immediately previous page's rank-2 anchor candidate has
large oracle rescue value, while AP-RS8's learned K=4 visible gate was near-null.
RS13 therefore removes candidate selection from the problem: K=1 is fixed and the
model learns only *when* to BACK to that rank-2 option.

V0 uses minimal anchor/state features, V1 adds current visible navigation state,
and V2 adds semantic features from pages/paragraphs already seen by the agent or
reader. No unvisited destination page body, degree, or prefetched content is used.
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
    GRAPH_URL,
    HTML_URL,
    build_visible_link_corpus,
    download,
    extract_graph,
)
from experiments.ap_rs4_article_body_semantics import SNAP_TEXT_URL, load_article_texts

SEED = 20260901
FIT_N = 1500
TUNE_N = 800
TEST_N = 2400
N_BOOT = 2000
PRIMARY_BUDGET = 16
SAFETY_BUDGET = 32
MAX_CHARS = 6000
MARGINS = [-0.20, -0.10, -0.05, -0.02, 0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 999.0]
VARIANTS = ("V0", "V1", "V2")


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
    return {
        "means": means,
        "positive": int(sum(x > 0 for x in means if np.isfinite(x))),
        "u_c": uc,
        "U_k1.96": float(1.96 * uc) if np.isfinite(uc) else None,
    }


def main():
    out = Path(os.environ.get("AP_RS13_OUT", "artifacts/ap_rs13"))
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    raw.mkdir(exist_ok=True)

    graph_tar = raw / "graph.tar.gz"
    html_tar = raw / "html.tar.gz"
    text_tar = raw / "plaintext.tar.gz"
    download(GRAPH_URL, graph_tar)
    download(HTML_URL, html_tar)
    download(SNAP_TEXT_URL, text_tar)

    articles, links, missions = extract_graph(graph_tar, raw / "graph")
    idx = {a: i for i, a in enumerate(articles)}
    occ, anchors, contexts, edge_coverage = build_visible_link_corpus(html_tar, articles, links)
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
    title_emb = enc.encode(
        [a.replace("_", " ") for a in articles], batch_size=128,
        show_progress_bar=True, normalize_embeddings=True,
    )
    anchor_emb = enc.encode(anchors, batch_size=128, show_progress_bar=True,
                            normalize_embeddings=True)
    context_emb = enc.encode(contexts, batch_size=64, show_progress_bar=True,
                             normalize_embeddings=True)
    body_texts = [texts.get(a, a.replace("_", " "))[:MAX_CHARS] for a in articles]
    body_emb = enc.encode(body_texts, batch_size=64, show_progress_bar=True,
                          normalize_embeddings=True)

    @lru_cache(maxsize=50000)
    def scored(src, goal):
        """Anchor-only destination ordering, with context score retained as a visible feature."""
        gv = title_emb[goal]
        best = {}
        for v, ai, ci in occ[src]:
            sa = float(np.dot(anchor_emb[ai], gv))
            sc = float(np.dot(context_emb[ci], gv))
            if v not in best or sa > best[v][0]:
                best[v] = (sa, sc)
        return tuple(sorted(((v, sa, sc) for v, (sa, sc) in best.items()),
                            key=lambda z: (-z[1], z[0])))

    def available(src, goal, seen):
        return [z for z in scored(src, goal) if z[0] not in seen]

    @lru_cache(maxsize=50000)
    def body_sim(page, goal):
        return float(np.dot(body_emb[page], title_emb[goal]))

    @lru_cache(maxsize=50000)
    def title_sim(page, goal):
        return float(np.dot(title_emb[page], title_emb[goal]))

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

    def state_record(parent, current, goal, steps, seen, origin_opts, chosen, alt):
        cur_opts = available(current, goal, seen)
        cur_count = len(cur_opts)
        cur_top1 = cur_opts[0][1] if cur_count else 0.0
        cur_top2 = cur_opts[1][1] if cur_count >= 2 else cur_top1
        cur_gap = cur_top1 - cur_top2 if cur_count >= 2 else 0.0
        chosen_sa = chosen[1]
        alt_sa, alt_sc = alt[1], alt[2]
        bcur = body_sim(current, goal)
        bpar = body_sim(parent, goal)
        tcur = title_sim(current, goal)
        tpar = title_sim(parent, goal)
        return {
            "origin_count": len(origin_opts),
            "rel_alt_cur": alt_sa - cur_top1,
            "step_frac": steps / PRIMARY_BUDGET,
            "cur_count": cur_count,
            "cur_top1": cur_top1,
            "cur_gap": cur_gap,
            "alt_parent_margin": alt_sa - chosen_sa,
            "body_current": bcur,
            "body_parent": bpar,
            "body_progress": bcur - bpar,
            "title_current": tcur,
            "title_parent": tpar,
            "title_progress": tcur - tpar,
            "alt_context": alt_sc,
            "alt_anchor_context_gap": alt_sa - alt_sc,
        }

    def fvec(r, variant):
        v = [math.log1p(r["origin_count"]), r["rel_alt_cur"], r["step_frac"]]
        if variant in ("V1", "V2"):
            v += [math.log1p(r["cur_count"]), r["cur_top1"], r["cur_gap"], r["alt_parent_margin"]]
        if variant == "V2":
            v += [
                r["body_current"], r["body_parent"], r["body_progress"],
                r["title_current"], r["title_parent"], r["title_progress"],
                r["alt_context"], r["alt_anchor_context_gap"],
            ]
        return v

    def build_teacher(tasks):
        rows = []
        for qi, (source, target) in enumerate(tasks):
            cur = idx[source]
            goal = idx[target]
            seen = {cur}
            steps = 0
            while steps < PRIMARY_BUDGET and cur != goal:
                parent = cur
                opts = available(parent, goal, seen)
                if not opts:
                    break
                chosen = opts[0]
                alt = opts[1] if len(opts) >= 2 else None
                cur = chosen[0]
                steps += 1
                if cur == goal:
                    break
                seen.add(cur)
                if alt is not None and alt[0] not in seen and steps + 2 <= PRIMARY_BUDGET:
                    rec = state_record(parent, cur, goal, steps, seen, opts, chosen, alt)
                    cont = local_rollout(cur, goal, steps, PRIMARY_BUDGET, seen)
                    back = local_rollout(alt[0], goal, steps + 2, PRIMARY_BUDGET, seen | {alt[0]})
                    rec["y"] = float(back) - float(cont)
                    rows.append(rec)
            if qi % 300 == 0:
                print("teacher task", qi, "rows", len(rows))
        return rows

    rng = np.random.default_rng(SEED)
    targets = sorted({t for s, t in missions if s in idx and t in idx})
    rng.shuffle(targets)
    nt = len(targets)
    fit_targets = set(targets[:int(.40 * nt)])
    tune_targets = set(targets[int(.40 * nt):int(.60 * nt)])
    test_targets = set(targets[int(.60 * nt):])
    fit_tasks = choose_tasks(missions, fit_targets, FIT_N, rng)
    tune_tasks = choose_tasks(missions, tune_targets, TUNE_N, rng)
    test_tasks = choose_tasks(missions, test_targets, TEST_N, rng)
    print("split", len(fit_tasks), len(tune_tasks), len(test_tasks),
          "targets", len(fit_targets), len(tune_targets), len(test_targets))

    fit_teacher = build_teacher(fit_tasks)
    y = np.asarray([r["y"] for r in fit_teacher], float)
    models = {}
    fit_auc = {}
    for variant in VARIANTS:
        X = np.asarray([fvec(r, variant) for r in fit_teacher], float)
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(X, y)
        pred = model.predict(X)
        non = y != 0
        auc = None
        if non.sum() and len(np.unique(y[non] > 0)) == 2:
            auc = float(roc_auc_score((y[non] > 0).astype(int), pred[non]))
        models[variant] = model
        fit_auc[variant] = auc
        print("FIT", variant, "rows", len(y), "non-tie", int(non.sum()), "auc", auc)

    def run_policy(task, budget, variant=None, margin=None):
        source, target = task
        cur = idx[source]
        goal = idx[target]
        seen = {cur}
        steps = 0
        used = False
        if cur == goal:
            return True, used
        while steps < budget:
            parent = cur
            opts = available(parent, goal, seen)
            if not opts:
                return False, used
            chosen = opts[0]
            alt = opts[1] if len(opts) >= 2 else None
            cur = chosen[0]
            steps += 1
            if cur == goal:
                return True, used
            seen.add(cur)
            if (variant is not None and not used and alt is not None and alt[0] not in seen
                    and steps < PRIMARY_BUDGET and steps + 2 <= budget):
                rec = state_record(parent, cur, goal, steps, seen, opts, chosen, alt)
                pred = float(models[variant].predict(np.asarray([fvec(rec, variant)], float))[0])
                if pred > margin:
                    cur = alt[0]
                    steps += 2
                    used = True
                    if cur == goal:
                        return True, used
                    seen.add(cur)
        return False, used

    def eval_tasks(tasks, variant, margin):
        rows = []
        for source, target in tasks:
            task = (source, target)
            l16 = run_policy(task, 16)[0]
            l32 = run_policy(task, 32)[0]
            p16, u16 = run_policy(task, 16, variant, margin)
            p32, u32 = run_policy(task, 32, variant, margin)
            rows.append({
                "source": source, "target": target,
                "local16": int(l16), "local32": int(l32),
                "policy16": int(p16), "policy32": int(p32),
                "d16": int(p16) - int(l16), "d32": int(p32) - int(l32),
                "intervened": int(u16 or u32),
            })
        return rows

    # Tune model block and threshold together using only tune targets.
    grid = []
    selected = None
    variant_order = {v: i for i, v in enumerate(VARIANTS)}
    for variant in VARIANTS:
        for margin in MARGINS:
            rr = eval_tasks(tune_tasks, variant, margin)
            d16 = float(np.mean([r["d16"] for r in rr]))
            d32 = float(np.mean([r["d32"] for r in rr]))
            intr = float(np.mean([r["intervened"] for r in rr]))
            row = {"variant": variant, "margin": margin, "d16": d16, "d32": d32,
                   "intervention_rate": intr}
            grid.append(row)
            if d32 >= -.005:
                key = (d16, d32, -variant_order[variant], margin)
                if selected is None or key > selected[0]:
                    selected = (key, row)
    assert selected is not None
    chosen = selected[1]
    variant = chosen["variant"]
    margin = chosen["margin"]
    print("SELECTED", chosen)

    test_rows = eval_tasks(test_tasks, variant, margin)
    d16 = float(np.mean([r["d16"] for r in test_rows]))
    d32 = float(np.mean([r["d32"] for r in test_rows]))
    ci16 = target_cluster_ci(test_rows, "d16", 1)
    ci32 = target_cluster_ci(test_rows, "d32", 37)
    b16 = bucket_stats(test_rows, "d16")
    b32 = bucket_stats(test_rows, "d32")
    l16 = float(np.mean([r["local16"] for r in test_rows]))
    l32 = float(np.mean([r["local32"] for r in test_rows]))
    p16 = float(np.mean([r["policy16"] for r in test_rows]))
    p32 = float(np.mean([r["policy32"] for r in test_rows]))
    intr = float(np.mean([r["intervened"] for r in test_rows]))

    # Same-test K=1 oracle ceiling for recovery accounting.
    def run_oracle(task, budget):
        source, target = task
        cur = idx[source]
        goal = idx[target]
        seen = {cur}
        steps = 0
        if cur == goal:
            return True, False
        while steps < budget:
            parent = cur
            opts = available(parent, goal, seen)
            if not opts:
                return False, False
            chosen = opts[0]
            alt = opts[1] if len(opts) >= 2 else None
            cur = chosen[0]
            steps += 1
            if cur == goal:
                return True, False
            seen.add(cur)
            if alt is not None and alt[0] not in seen and steps + 2 <= budget:
                cont = local_rollout(cur, goal, steps, budget, seen)
                if cont:
                    return True, False
                back = local_rollout(alt[0], goal, steps + 2, budget, seen | {alt[0]})
                if back:
                    return True, True
        return False, False

    oracle_rows = []
    for source, target in test_tasks:
        task = (source, target)
        l_16 = run_policy(task, 16)[0]
        l_32 = run_policy(task, 32)[0]
        o16, u16 = run_oracle(task, 16)
        o32, u32 = run_oracle(task, 32)
        oracle_rows.append({
            "source": source, "target": target,
            "d16": int(o16) - int(l_16), "d32": int(o32) - int(l_32),
            "used16": int(u16), "used32": int(u32),
        })
    od16 = float(np.mean([r["d16"] for r in oracle_rows]))
    od32 = float(np.mean([r["d32"] for r in oracle_rows]))
    oci16 = target_cluster_ci(oracle_rows, "d16", 101)
    oci32 = target_cluster_ci(oracle_rows, "d32", 137)

    # Held-out state-level identifiability diagnostic, never used for selection.
    test_teacher = build_teacher(test_tasks)
    yt = np.asarray([r["y"] for r in test_teacher], float)
    heldout_auc = {}
    for v in VARIANTS:
        Xt = np.asarray([fvec(r, v) for r in test_teacher], float)
        pr = models[v].predict(Xt)
        non = yt != 0
        auc = None
        if non.sum() and len(np.unique(yt[non] > 0)) == 2:
            auc = float(roc_auc_score((yt[non] > 0).astype(int), pr[non]))
        heldout_auc[v] = auc

    recovery = float(d16 / od16) if od16 > 0 else None
    conditions = {
        "test_n_ge_400": len(test_rows) >= 400,
        "S16_gain_ge_2pp": d16 >= .02,
        "S16_target_CI_lower_gt_0": ci16[0] > 0,
        "S16_positive_target_buckets_ge_6_of_8": b16["positive"] >= 6,
        "S32_mean_noninferior_minus_0_5pp": d32 >= -.005,
        "S32_target_CI_lower_ge_minus_1pp": ci32[0] >= -.01,
    }
    decision = "PASS" if all(conditions.values()) else "FAIL"
    result = {
        "phase": "AP-RS13",
        "name": "K1 trigger-only visible semantic-state direct utility",
        "decision": decision,
        "preregistered_conditions": conditions,
        "construct": "real Wikispeedia visible anchor/paragraph plus visited-page semantics; K1 simulated navigation trigger",
        "data": {
            "articles": len(articles), "graph_links": len(links), "missions": len(missions),
            "visible_edge_coverage": edge_coverage, "body_coverage": body_coverage,
        },
        "split": {
            "target_disjoint": True, "seed": SEED,
            "fit_tasks": len(fit_tasks), "tune_tasks": len(tune_tasks), "test_tasks": len(test_tasks),
            "fit_targets": len(fit_targets), "tune_targets": len(tune_targets), "test_targets": len(test_targets),
        },
        "teacher": {
            "fit_rows": len(fit_teacher),
            "fit_non_tie": int(np.sum(y != 0)),
            "fit_positive": int(np.sum(y > 0)),
            "fit_negative": int(np.sum(y < 0)),
            "fit_auc": fit_auc,
            "heldout_test_rows": len(test_teacher),
            "heldout_non_tie": int(np.sum(yt != 0)),
            "heldout_auc": heldout_auc,
        },
        "variants": {
            "V0": "log1p(origin count), rank2 anchor relative to current top1, step fraction",
            "V1": "V0 + current candidate count/top1/gap + previous rank2-vs-rank1 margin",
            "V2": "V1 + visited current/parent body and title semantics + remembered rank2 paragraph and anchor-context disagreement",
        },
        "tuning": {"grid": grid, "selected": chosen, "safety_constraint": "mean S@32 harm <= 0.5pp"},
        "test": {
            "local": {"S16": l16, "S32": l32},
            "policy": {
                "variant": variant, "margin": margin,
                "S16": p16, "S32": p32,
                "delta_S16": d16, "delta_S32": d32,
                "target_cluster_CI95_S16": ci16,
                "target_cluster_CI95_S32": ci32,
                "target_bucket_S16": b16,
                "target_bucket_S32": b32,
                "intervention_rate": intr,
            },
            "oracle_K1": {
                "delta_S16": od16, "delta_S32": od32,
                "target_cluster_CI95_S16": oci16,
                "target_cluster_CI95_S32": oci32,
                "learned_oracle_recovery_fraction_S16": recovery,
            },
        },
        "boundary": [
            "V2 uses only pages and link context already visited/seen; it does not prefetch the deferred destination.",
            "Semantic scoring computation is sidebar-agent feasible but human cognitive cost is not modeled.",
            "Variant and threshold selection occurs only on tune targets; test targets are touched once for deployment evaluation.",
            "This is simulated goal-directed navigation, not human comprehension or retention.",
        ],
    }
    (out / "AP_RS13_RESULTS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md = f"""# AP-RS13 — K1 trigger-only visible semantic state\n\n**Decision: {decision}**\n\n- selected: {variant}, margin={margin}\n- test missions: {len(test_rows)}\n- local S@16 / S@32: {l16:.4f} / {l32:.4f}\n- policy S@16 delta: {100*d16:+.3f} pp, CI {ci16}, buckets + {b16['positive']}/8\n- policy S@32 delta: {100*d32:+.3f} pp, CI {ci32}\n- intervention rate: {intr:.4f}\n- same-test oracle K1 S@16: {100*od16:+.3f} pp, CI {oci16}\n- learned/oracle recovery: {recovery}\n- fit AUC: {fit_auc}\n- held-out state AUC: {heldout_auc}\n\n## Boundary\nNo deferred destination prefetch. V2 uses only already-visible semantic state; navigation is simulated and is not a comprehension study.\n"""
    (out / "AP_RS13_SUMMARY.md").write_text(md, encoding="utf-8")
    print("AP_RS13_DECISION", decision)
    print("SELECTED", chosen)
    print("TEST", d16, ci16, b16, d32, ci32, "intr", intr)
    print("ORACLE_K1", od16, oci16, od32, oci32, "recovery", recovery)
    print("HELDOUT_AUC", heldout_auc)


if __name__ == "__main__":
    main()
