#!/usr/bin/env python3
"""Preregistered AP-LM3B confirmation on the untouched final 40% target split."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tarfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.ap_lm3b_visible_query_cost import (
    Node,
    compute_node_dps,
    forest_oracle,
    visible_query_cost,
)
from experiments.ap_rs5_real_anchor_context_policy import (
    GRAPH_URL,
    HTML_URL,
    build_visible_link_corpus,
    choose_tasks,
    download,
    extract_graph,
)
from experiments.ap_rs4_article_body_semantics import SNAP_TEXT_URL, load_article_texts

SEED = 20260912
FIT_N = 700
TEST_N = 1200
PRIMARY_BUDGET = 12
SAFETY_BUDGET = 20
BUDGETS = (PRIMARY_BUDGET, SAFETY_BUDGET)
MAX_DEPTH = 3
N_ROOTS = 6
MAX_CHILDREN = 3
MAX_NODES = 18
MAX_BODY_CHARS = 1800
RIDGE_ALPHA = 10.0
N_BOOT = 10000
K_VALUES = (2, 4, 8)


def target_cluster_compare(rows, a, b, budget, offset):
    by = {}
    for r in rows:
        if r["budget"] != budget or r["policy"] not in (a, b):
            continue
        by.setdefault(r["target"], {}).setdefault(r["policy"], []).append(r["oracle_fraction"])
    deltas = []
    for d in by.values():
        if a in d and b in d:
            deltas.append(float(np.mean(d[a]) - np.mean(d[b])))
    x = np.asarray(deltas, dtype=float)
    if not len(x):
        raise RuntimeError("no paired target clusters")
    rng = np.random.default_rng(SEED + 5000 + offset)
    reps = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        reps[i] = float(np.mean(x[rng.integers(0, len(x), len(x))]))
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return {
        "mean": float(np.mean(x)),
        "ci95": [float(lo), float(hi)],
        "positive_targets": int(np.sum(x > 0)),
        "n_targets": int(len(x)),
    }


def bucket_compare(rows, a, b, budget):
    by = {}
    for r in rows:
        if r["budget"] != budget or r["policy"] not in (a, b):
            continue
        by.setdefault((r["source"], r["target"]), {})[r["policy"]] = r["oracle_fraction"]
    buckets = [[] for _ in range(8)]
    for (_source, target), d in by.items():
        if a not in d or b not in d:
            continue
        j = int(hashlib.sha256(target.encode()).hexdigest()[:8], 16) % 8
        buckets[j].append(d[a] - d[b])
    means = [float(np.mean(v)) if v else float("nan") for v in buckets]
    return {"means": means, "positive": int(sum(np.isfinite(x) and x > 0 for x in means))}


def main():
    out = Path(os.environ.get("AP_LM3B_CONFIRM_OUT", "artifacts/ap_lm3b_confirm"))
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    graph_tar = raw / "graph.tar.gz"
    html_tar = raw / "html.tar.gz"
    text_tar = raw / "plaintext.tar.gz"
    download(GRAPH_URL, graph_tar)
    download(HTML_URL, html_tar)
    download(SNAP_TEXT_URL, text_tar)

    articles, graph_links, missions = extract_graph(graph_tar, raw / "graph")
    idx = {a: i for i, a in enumerate(articles)}
    occ, anchors, _contexts, edge_coverage = build_visible_link_corpus(html_tar, articles, graph_links)
    if edge_coverage < 0.85:
        raise RuntimeError(f"visible coverage too low: {edge_coverage:.3f}")

    text_root = raw / "plaintext"
    if not text_root.exists() or not any(text_root.iterdir()):
        text_root.mkdir(exist_ok=True)
        with tarfile.open(text_tar, "r:gz") as tf:
            tf.extractall(text_root)
    texts = load_article_texts(text_root, set(articles))
    body_coverage = len(texts) / max(1, len(articles))
    if body_coverage < 0.90:
        raise RuntimeError(f"body coverage too low: {body_coverage:.3f}")

    answer_texts = [texts.get(a, a.replace("_", " "))[:MAX_BODY_CHARS] for a in articles]
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    title_emb = enc.encode(
        [a.replace("_", " ") for a in articles], batch_size=128,
        show_progress_bar=True, normalize_embeddings=True,
    )
    anchor_emb = enc.encode(
        anchors, batch_size=128, show_progress_bar=True, normalize_embeddings=True,
    )
    body_emb = enc.encode(
        answer_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True,
    )
    anchor_costs = np.asarray([visible_query_cost(a) for a in anchors], dtype=int)

    @lru_cache(maxsize=120000)
    def visible_edges(src, goal):
        gv = title_emb[goal]
        best = {}
        for v, ai, _ci in occ[src]:
            if v == src:
                continue
            sa = float(np.dot(anchor_emb[ai], gv))
            st = float(np.dot(title_emb[v], gv))
            vis = 0.80 * sa + 0.20 * st
            qcost = int(anchor_costs[ai])
            cur = best.get(v)
            if cur is None or vis > cur[0]:
                best[v] = (vis, sa, st, qcost)
        return tuple(sorted(((v,) + x for v, x in best.items()), key=lambda z: (-z[1], z[4], z[0])))

    @lru_cache(maxsize=120000)
    def true_reward(article, goal):
        sim = float(np.dot(body_emb[article], body_emb[goal]))
        return max(0.0, sim) ** 2

    def build_tree(source, goal):
        nodes = {}
        roots = []
        used_articles = {source}
        next_id = 0

        def add_node(article, depth, parent_id, e):
            nonlocal next_id
            _article, vis, sa, st, qcost = e
            nid = next_id
            next_id += 1
            nodes[nid] = Node(
                nid, article, depth, parent_id, float(sa), float(st), float(vis),
                int(qcost), float(true_reward(article, goal)), []
            )
            return nid

        for e in [e for e in visible_edges(source, goal) if e[0] not in used_articles][:N_ROOTS]:
            used_articles.add(e[0])
            roots.append(add_node(e[0], 0, None, e))
        queue = list(roots)
        while queue and len(nodes) < MAX_NODES:
            nid = queue.pop(0)
            node = nodes[nid]
            if node.depth >= MAX_DEPTH:
                continue
            added = 0
            for e in visible_edges(node.article, goal):
                if e[0] in used_articles:
                    continue
                used_articles.add(e[0])
                cid = add_node(e[0], node.depth + 1, nid, e)
                node.children.append(cid)
                queue.append(cid)
                added += 1
                if added >= MAX_CHILDREN or len(nodes) >= MAX_NODES:
                    break
        return nodes, roots

    def feature(nodes, nid, remaining, budget, frontier):
        n = nodes[nid]
        parent_reward = nodes[n.parent_id].reward if n.parent_id is not None else 0.0
        rank = 1 + sum(nodes[x].visible_score > n.visible_score for x in frontier)
        return np.asarray([
            n.anchor_score,
            n.title_score,
            n.visible_score,
            math.log1p(n.query_cost),
            n.depth / MAX_DEPTH,
            math.log1p(parent_reward),
            remaining / budget,
            math.log1p(len(frontier)),
            n.visible_score / n.query_cost,
            rank / max(1, len(frontier)),
        ], dtype=float)

    def feasible(frontier, nodes, remaining):
        return [nid for nid in frontier if nodes[nid].query_cost <= remaining]

    def greedy_choice(nodes, frontier, remaining):
        xs = feasible(frontier, nodes, remaining)
        if not xs:
            return None
        return max(xs, key=lambda nid: (
            nodes[nid].visible_score / nodes[nid].query_cost,
            nodes[nid].visible_score,
            -nodes[nid].query_cost,
        ))

    def collect_teacher(task, budget):
        source, target = task
        nodes, roots = build_tree(idx[source], idx[target])
        if len(roots) < 2:
            return [], [], []
        dps = compute_node_dps(nodes, max(BUDGETS))
        remaining = budget
        frontier = list(roots)
        X, yr, yi = [], [], []
        while True:
            xs = feasible(frontier, nodes, remaining)
            if not xs:
                break
            for nid in xs:
                X.append(feature(nodes, nid, remaining, budget, frontier))
                yr.append(float(np.max(dps[nid][: remaining + 1])))
                yi.append(float(nodes[nid].reward))
            chosen = greedy_choice(nodes, frontier, remaining)
            frontier.remove(chosen)
            n = nodes[chosen]
            remaining -= n.query_cost
            frontier.extend(n.children)
        return X, yr, yi

    rng = np.random.default_rng(SEED)
    targets = sorted({t for s, t in missions if s in idx and t in idx})
    rng.shuffle(targets)
    nt = len(targets)
    fit_targets = set(targets[: int(0.40 * nt)])
    confirm_targets = set(targets[int(0.60 * nt):])
    fit_tasks = choose_tasks(missions, fit_targets, FIT_N, np.random.default_rng(SEED + 1))
    test_tasks = choose_tasks(missions, confirm_targets, TEST_N, np.random.default_rng(SEED + 3))
    print("confirm_split", len(fit_tasks), len(test_tasks), "targets", len(fit_targets), len(confirm_targets))

    X, yr, yi = [], [], []
    for i, task in enumerate(fit_tasks):
        for budget in BUDGETS:
            xx, rr, ii = collect_teacher(task, budget)
            X.extend(xx); yr.extend(rr); yi.extend(ii)
        if i % 150 == 0:
            print("fit_task", i, "rows", len(yr))
    Xn = np.vstack(X)
    rec_model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)).fit(Xn, np.asarray(yr))
    imm_model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)).fit(Xn, np.asarray(yi))

    def predict(model, nodes, frontier, remaining, budget, candidates):
        return model.predict(np.vstack([feature(nodes, nid, remaining, budget, frontier) for nid in candidates]))

    def compress_rec(nodes, frontier, remaining, budget, k):
        # Frozen pre-confirm correction: permanently infeasible candidates cannot consume K slots.
        frontier = feasible(frontier, nodes, remaining)
        if k is None or len(frontier) <= k:
            return frontier
        pred = predict(rec_model, nodes, frontier, remaining, budget, frontier)
        order = np.argsort(-pred)[:k]
        return [frontier[int(i)] for i in order]

    def rollout(task, budget, policy, k=None):
        source, target = task
        nodes, roots = build_tree(idx[source], idx[target])
        if len(roots) < 2:
            return None
        dps = compute_node_dps(nodes, max(BUDGETS))
        oracle = forest_oracle(nodes, roots, budget, dps)
        if oracle <= 0:
            return None
        remaining = budget
        frontier = list(roots)
        total = 0.0
        retained = []
        if policy == "rec" and k is not None:
            frontier = compress_rec(nodes, frontier, remaining, budget, k)
        while True:
            xs = feasible(frontier, nodes, remaining)
            if not xs:
                break
            retained.append(len(frontier))
            if policy == "greedy":
                chosen = greedy_choice(nodes, frontier, remaining)
            elif policy == "imm":
                pred = predict(imm_model, nodes, frontier, remaining, budget, xs)
                chosen = xs[int(np.argmax(pred))]
            elif policy == "rec":
                pred = predict(rec_model, nodes, frontier, remaining, budget, xs)
                chosen = xs[int(np.argmax(pred))]
            else:
                raise ValueError(policy)
            frontier.remove(chosen)
            n = nodes[chosen]
            remaining -= n.query_cost
            total += n.reward
            frontier.extend(n.children)
            if policy == "rec" and k is not None:
                frontier = compress_rec(nodes, frontier, remaining, budget, k)
        return total, oracle, float(np.mean(retained)) if retained else 0.0

    policy_specs = [
        ("greedy", "greedy", None),
        ("imm", "imm", None),
        ("rec_full", "rec", None),
        ("rec_k2", "rec", 2),
        ("rec_k4", "rec", 4),
        ("rec_k8", "rec", 8),
    ]
    rows = []
    for i, task in enumerate(test_tasks):
        for budget in BUDGETS:
            for name, policy, k in policy_specs:
                z = rollout(task, budget, policy, k)
                if z is None:
                    continue
                reward, oracle, retained = z
                rows.append({
                    "source": task[0], "target": task[1], "budget": budget,
                    "policy": name, "reward": reward, "oracle": oracle,
                    "oracle_fraction": reward / oracle, "mean_retained": retained,
                })
        if i % 200 == 0:
            print("confirm_task", i, "rows", len(rows))

    agg = {}
    for budget in BUDGETS:
        agg[str(budget)] = {}
        for name, _policy, _k in policy_specs:
            rr = [r for r in rows if r["budget"] == budget and r["policy"] == name]
            agg[str(budget)][name] = {
                "n": len(rr),
                "mean_oracle_fraction": float(np.mean([r["oracle_fraction"] for r in rr])) if rr else None,
                "mean_retained": float(np.mean([r["mean_retained"] for r in rr])) if rr else None,
            }

    h1 = target_cluster_compare(rows, "rec_full", "imm", PRIMARY_BUDGET, 1)
    h2 = target_cluster_compare(rows, "rec_full", "greedy", PRIMARY_BUDGET, 2)
    safety = target_cluster_compare(rows, "rec_full", "imm", SAFETY_BUDGET, 3)
    h1_buckets = bucket_compare(rows, "rec_full", "imm", PRIMARY_BUDGET)
    h2_buckets = bucket_compare(rows, "rec_full", "greedy", PRIMARY_BUDGET)

    primary_n = agg[str(PRIMARY_BUDGET)]["rec_full"]["n"]
    pass_h1 = bool(
        primary_n >= 800
        and h1["mean"] >= 0.01
        and h1["ci95"][0] > 0
        and h1_buckets["positive"] >= 6
    )
    support_h2 = bool(
        h2["mean"] > 0
        and h2["ci95"][0] > 0
        and h2_buckets["positive"] >= 5
    )
    pass_safety = bool(safety["mean"] >= -0.0025 and safety["ci95"][0] >= -0.005)

    capacity = {}
    for k in K_VALUES:
        capacity[str(k)] = float(
            agg[str(PRIMARY_BUDGET)]["rec_full"]["mean_oracle_fraction"]
            - agg[str(PRIMARY_BUDGET)][f"rec_k{k}"]["mean_oracle_fraction"]
        )
    resource_success = bool(capacity["8"] <= 0.01)
    decision = "PASS" if pass_h1 and pass_safety else "FAIL"

    result = {
        "name": "AP-LM3B grounded natural-language visible-query-cost confirmation",
        "decision": decision,
        "primary_n": primary_n,
        "h1_recursive_vs_immediate_b12": h1,
        "h1_buckets": h1_buckets,
        "pass_h1": pass_h1,
        "h2_recursive_vs_greedy_b12": h2,
        "h2_buckets": h2_buckets,
        "support_h2": support_h2,
        "safety_recursive_vs_immediate_b20": safety,
        "pass_safety": pass_safety,
        "aggregate": agg,
        "capacity_full_minus_k_b12": capacity,
        "resource_success_k8": resource_success,
        "fit_n": len(fit_tasks),
        "confirm_n_requested": TEST_N,
        "confirm_target_count": len(confirm_targets),
        "edge_coverage": edge_coverage,
        "body_coverage": body_coverage,
        "teacher_rows": len(yr),
        "boundaries": [
            "retrieval-backed grounded answer surrogate, not generative LM",
            "hallucination absent",
            "visible anchor-derived query cost, not measured wall-clock latency",
            "semantic utility is not human comprehension",
        ],
    }
    (out / "AP_LM3B_CONFIRM_RESULTS.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = (
        "# AP-LM3B confirmatory summary\n\n"
        f"Decision: **{decision}**\n\n"
        f"H1 B=12 recursive-immediate: {100*h1['mean']:+.3f} pp, "
        f"CI [{100*h1['ci95'][0]:+.3f},{100*h1['ci95'][1]:+.3f}], "
        f"buckets {h1_buckets['positive']}/8, PASS={pass_h1}.\n\n"
        f"H2 B=12 recursive-greedy: {100*h2['mean']:+.3f} pp, "
        f"CI [{100*h2['ci95'][0]:+.3f},{100*h2['ci95'][1]:+.3f}], "
        f"buckets {h2_buckets['positive']}/8, supported={support_h2}.\n\n"
        f"Safety B=20 recursive-immediate: {100*safety['mean']:+.3f} pp, "
        f"CI [{100*safety['ci95'][0]:+.3f},{100*safety['ci95'][1]:+.3f}], PASS={pass_safety}.\n\n"
        f"K8 full gap B=12: {100*capacity['8']:+.3f} pp; resource success={resource_success}.\n"
    )
    (out / "AP_LM3B_CONFIRM_SUMMARY.md").write_text(summary, encoding="utf-8")

    print("AP_LM3B_CONFIRM_DECISION", decision)
    print("AP_LM3B_H1", json.dumps(h1), "buckets", h1_buckets["positive"], "PASS", pass_h1)
    print("AP_LM3B_H2", json.dumps(h2), "buckets", h2_buckets["positive"], "SUPPORTED", support_h2)
    print("AP_LM3B_SAFETY", json.dumps(safety), "PASS", pass_safety)
    print("AP_LM3B_CAPACITY_B12", json.dumps(capacity), "K8_SUCCESS", resource_success)


if __name__ == "__main__":
    main()
