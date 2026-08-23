#!/usr/bin/env python3
"""AP-RS9: preregistered wrong-turn-triggered branch-switch policy.

Motivation
----------
AP-RS4e was an exploratory human-path analysis: one-step BACK was strongly
corrective when the immediately preceding click had decreased target-body
semantic similarity, but not when that click had already made semantic progress.
AP-RS9 turns that observation into a new causal policy test on target-disjoint
Wikispeedia missions.

Information regime
------------------
* Link choice uses only real visible anchor text + containing paragraph.
* After visiting a page, its article-body prefix is considered readable and may
  be compared with the known Wikispeedia target title.
* No unvisited destination body or out-degree is used.
* At most one discretionary return is allowed.

The semantic-regression threshold is selected on tune targets only under the
same S@32 safety constraint used in RS5/RS6. The test target set is touched once.
This is simulated navigation, not a human comprehension experiment; reading time
for obtaining the current-page body representation is not modeled.
"""
from __future__ import annotations

import csv
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

from experiments.ap_rs5_real_anchor_context_policy import (
    GRAPH_URL,
    HTML_URL,
    build_visible_link_corpus,
    download,
    extract_graph,
)
from experiments.ap_rs4_article_body_semantics import SNAP_TEXT_URL, load_article_texts

SEED = 20260826
K = 4
TUNE_N = 600
TEST_N = 1600
N_BOOT = 2000
PRIMARY_BUDGET = 16
SAFETY_BUDGET = 32
# Trigger when body_progress = sim(current,target)-sim(parent,target) < -tau.
TAUS = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 999.0]
MAX_CHARS = 6000


def choose_tasks(missions, target_set, n, rng):
    xs = [x for x in missions if x[1] in target_set]
    rng.shuffle(xs)
    return xs[: min(n, len(xs))]


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
    return [float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))]


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
    out = Path(os.environ.get("AP_RS9_OUT", "artifacts/ap_rs9"))
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    raw.mkdir(exist_ok=True)

    graph_tar = raw / "wikispeedia_paths-and-graph.tar.gz"
    html_tar = raw / "wikispeedia_articles_html.tar.gz"
    text_tar = raw / "wikispeedia_articles_plaintext.tar.gz"
    download(GRAPH_URL, graph_tar)
    download(HTML_URL, html_tar)
    download(SNAP_TEXT_URL, text_tar)

    articles, graph_links, missions = extract_graph(graph_tar, raw / "graph")
    article_idx = {a: i for i, a in enumerate(articles)}
    edge_occ, anchors, contexts, edge_coverage = build_visible_link_corpus(html_tar, articles, graph_links)
    if edge_coverage < 0.85:
        raise RuntimeError(f"visible edge coverage too low: {edge_coverage:.3f}")

    text_root = raw / "plaintext"
    if not text_root.exists() or not any(text_root.iterdir()):
        text_root.mkdir(exist_ok=True)
        with tarfile.open(text_tar, "r:gz") as tf:
            tf.extractall(text_root)
    texts = load_article_texts(text_root, set(articles))
    body_coverage = len(texts) / max(1, len(articles))
    if body_coverage < 0.90:
        raise RuntimeError(f"body coverage too low: {body_coverage:.3f}")

    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    title_emb = enc.encode(
        [a.replace("_", " ") for a in articles], batch_size=128,
        show_progress_bar=True, normalize_embeddings=True,
    )
    anchor_emb = enc.encode(anchors, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    context_emb = enc.encode(contexts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    body_texts = [texts.get(a, a.replace("_", " "))[:MAX_CHARS] for a in articles]
    body_emb = enc.encode(body_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    @lru_cache(maxsize=40000)
    def scored(src_i, goal_i):
        gv = title_emb[goal_i]
        best = {}
        for tgt_i, ai, ci in edge_occ[src_i]:
            sa = float(np.dot(anchor_emb[ai], gv))
            sc = float(np.dot(context_emb[ci], gv))
            s = 0.5 * (sa + sc)
            if tgt_i not in best or s > best[tgt_i]:
                best[tgt_i] = s
        return tuple(sorted(best.items(), key=lambda x: (-x[1], x[0])))

    def available(src_i, goal_i, seen):
        return [(v, s) for v, s in scored(src_i, goal_i) if v not in seen]

    @lru_cache(maxsize=50000)
    def body_sim(page_i, goal_i):
        return float(np.dot(body_emb[page_i], title_emb[goal_i]))

    def run_policy(task, budget, tau=None):
        source, target = task
        cur = article_idx[source]
        goal = article_idx[target]
        seen = {cur}
        steps = 0
        used = False
        trigger_delta = None
        if cur == goal:
            return True, steps, used, trigger_delta

        while steps < budget:
            parent = cur
            opts = available(parent, goal, seen)
            if not opts:
                return False, steps, used, trigger_delta
            chosen_i, _ = opts[0]
            deferred = opts[1 : 1 + K]

            # Forward click.
            cur = chosen_i
            steps += 1
            if cur == goal:
                return True, steps, used, trigger_delta
            seen.add(cur)

            # After reading the visited page, detect a semantic wrong turn.
            if tau is not None and (not used) and deferred and steps < PRIMARY_BUDGET and steps + 2 <= budget:
                delta = body_sim(cur, goal) - body_sim(parent, goal)
                if delta < -tau:
                    alt_i = deferred[0][0]  # best visible alternative from the parent page
                    if alt_i not in seen:
                        trigger_delta = float(delta)
                        cur = alt_i
                        steps += 2  # browser Back + click alternative
                        used = True
                        if cur == goal:
                            return True, steps, used, trigger_delta
                        seen.add(cur)
        return False, steps, used, trigger_delta

    def eval_tasks(tasks, tau=None):
        rows = []
        for source, target in tasks:
            task = (source, target)
            l16 = run_policy(task, 16, None)[0]
            l32 = run_policy(task, 32, None)[0]
            if tau is None:
                p16, p32, u16, u32, td = l16, l32, False, False, None
            else:
                p16, _, u16, td16 = run_policy(task, 16, tau)
                p32, _, u32, td32 = run_policy(task, 32, tau)
                td = td16 if td16 is not None else td32
            rows.append({
                "source": source, "target": target,
                "local16": int(l16), "local32": int(l32),
                "policy16": int(p16), "policy32": int(p32),
                "d16": int(p16) - int(l16), "d32": int(p32) - int(l32),
                "intervened": int(u16 or u32),
                "trigger_delta": td if td is not None else "",
            })
        return rows

    rng = np.random.default_rng(SEED)
    targets = sorted({t for s, t in missions if s in article_idx and t in article_idx})
    rng.shuffle(targets)
    cut = int(0.30 * len(targets))
    tune_targets = set(targets[:cut])
    test_targets = set(targets[cut:])
    tune_tasks = choose_tasks(missions, tune_targets, TUNE_N, rng)
    test_tasks = choose_tasks(missions, test_targets, TEST_N, rng)
    print("split", len(tune_tasks), len(test_tasks), "targets", len(tune_targets), len(test_targets))

    grid = []
    selected = None
    for tau in TAUS:
        rr = eval_tasks(tune_tasks, tau)
        d16 = float(np.mean([r["d16"] for r in rr]))
        d32 = float(np.mean([r["d32"] for r in rr]))
        intr = float(np.mean([r["intervened"] for r in rr]))
        row = {"tau": tau, "d16": d16, "d32": d32, "intervention_rate": intr}
        grid.append(row)
        if d32 >= -0.005:
            if selected is None or d16 > selected["d16"] or (d16 == selected["d16"] and d32 > selected["d32"]):
                selected = row
    assert selected is not None
    tau = selected["tau"]
    print("selected", selected)

    test = eval_tasks(test_tasks, tau)
    d16 = float(np.mean([r["d16"] for r in test]))
    d32 = float(np.mean([r["d32"] for r in test]))
    ci16 = target_cluster_ci(test, "d16", 1)
    ci32 = target_cluster_ci(test, "d32", 37)
    b16 = bucket_stats(test, "d16")
    b32 = bucket_stats(test, "d32")
    l16 = float(np.mean([r["local16"] for r in test]))
    l32 = float(np.mean([r["local32"] for r in test]))
    p16 = float(np.mean([r["policy16"] for r in test]))
    p32 = float(np.mean([r["policy32"] for r in test]))
    intr = float(np.mean([r["intervened"] for r in test]))

    conditions = {
        "test_n_ge_400": len(test) >= 400,
        "visible_edge_coverage_ge_0.85": edge_coverage >= 0.85,
        "body_coverage_ge_0.90": body_coverage >= 0.90,
        "S16_gain_ge_2pp": d16 >= 0.02,
        "S16_target_CI_lower_gt_0": ci16[0] > 0,
        "S16_positive_target_buckets_ge_6_of_8": b16["positive"] >= 6,
        "S32_mean_noninferior_minus_0_5pp": d32 >= -0.005,
        "S32_target_CI_lower_ge_minus_1pp": ci32[0] >= -0.01,
    }
    decision = "PASS" if all(conditions.values()) else "FAIL"

    result = {
        "phase": "AP-RS9",
        "name": "wrong-turn-triggered visible branch-switch policy",
        "decision": decision,
        "preregistered_conditions": conditions,
        "construct": "real Wikispeedia visible anchor+paragraph local scoring plus visited-page body semantic regression trigger; simulated navigation",
        "data": {
            "articles": len(articles), "graph_links": len(graph_links), "unique_missions": len(missions),
            "visible_edge_coverage": edge_coverage, "body_coverage": body_coverage,
        },
        "split": {
            "target_disjoint": True, "tune_tasks": len(tune_tasks), "test_tasks": len(test_tasks),
            "tune_targets": len(tune_targets), "test_targets": len(test_targets), "seed": SEED,
        },
        "scoring": {
            "link_score": "0.5 anchor MiniLM cosine + 0.5 containing-paragraph MiniLM cosine to target title",
            "wrong_turn_signal": "visited-current body cosine to target title minus visited-parent body cosine",
            "unvisited_destination_metadata_used": False,
            "K_deferred": K,
            "max_discretionary_returns": 1,
        },
        "tuning": {
            "trigger_rule": "body_progress < -tau",
            "tau_grid": TAUS,
            "safety_constraint": "mean S@32 harm <= 0.5pp on tune targets",
            "grid": grid,
            "selected": selected,
        },
        "test": {
            "local": {"S16": l16, "S32": l32},
            "policy": {
                "S16": p16, "S32": p32, "delta_S16": d16, "delta_S32": d32,
                "target_cluster_CI95_S16": ci16, "target_cluster_CI95_S32": ci32,
                "target_bucket_S16": b16, "target_bucket_S32": b32,
                "intervention_rate": intr,
            },
        },
        "boundary": [
            "RS4e motivated the trigger but was exploratory; AP-RS9 uses new target-disjoint tune/test missions and a fixed rule family.",
            "Current-page body content is only used after visiting that page; no unvisited page body or degree is used.",
            "Reading time required to form a body-semantic representation is not modeled.",
            "Outcome is simulated goal-directed navigation, not human comprehension or retention.",
        ],
    }
    (out / "AP_RS9_RESULTS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "test_rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=test[0].keys())
        w.writeheader()
        w.writerows(test)
    (out / "AP_RS9_SUMMARY.md").write_text(
        f"# AP-RS9 — Wrong-turn-triggered visible branch switch\n\n"
        f"**Decision: {decision}**\n\n"
        f"- selected tau: {tau}\n"
        f"- test tasks: {len(test)}\n"
        f"- local S@16/S@32: {l16:.4f}/{l32:.4f}\n"
        f"- policy S@16/S@32: {p16:.4f}/{p32:.4f}\n"
        f"- delta S@16: {100*d16:+.3f} pp, CI {ci16}\n"
        f"- delta S@32: {100*d32:+.3f} pp, CI {ci32}\n"
        f"- positive S@16 buckets: {b16['positive']}/8\n"
        f"- intervention rate: {intr:.4f}\n",
        encoding="utf-8",
    )
    print("AP_RS9_DECISION", decision)
    print("AP_RS9_S16", d16, ci16, b16)
    print("AP_RS9_S32", d32, ci32, b32)


if __name__ == "__main__":
    main()
