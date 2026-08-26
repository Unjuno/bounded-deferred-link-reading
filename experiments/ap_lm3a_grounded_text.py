#!/usr/bin/env python3
"""AP-LM3A development/confirmatory bridge: grounded natural-language recursive answer trees.

This experiment moves AP-LM1/2 from latent synthetic nodes to real Wikispeedia
strings. A query is a visible linked concept. Issuing the query reveals that
concept's real article body as a grounded answer and exposes further visible
linked concepts inside that answer. No generative hallucination is present.

The experiment has two phases:
- pilot: development targets only, for feasibility and preregistration design.
- confirm: held-out target-disjoint evaluation after criteria are frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tarfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.ap_rs5_real_anchor_context_policy import (
    GRAPH_URL,
    HTML_URL,
    build_visible_link_corpus,
    choose_tasks,
    download,
    extract_graph,
)
from experiments.ap_rs4_article_body_semantics import SNAP_TEXT_URL, load_article_texts

SEED = 20260910
FIT_N = 700
PILOT_N = 280
TEST_N = 1200
BUDGETS = (12, 20)
PRIMARY_BUDGET = 12
LONG_BUDGET = 20
MAX_DEPTH = 3
N_ROOTS = 6
MAX_CHILDREN = 3
MAX_NODES = 18
MAX_BODY_CHARS = 1800
K_VALUES = (2, 4, 8)
N_BOOT = 4000
RIDGE_ALPHA = 10.0


@dataclass
class Node:
    node_id: int
    article: int
    depth: int
    parent_id: Optional[int]
    anchor_score: float
    context_score: float
    title_score: float
    visible_score: float
    reward: float
    latency: int
    children: List[int]


def latency_from_text(text: str) -> int:
    words = max(1, len(text.split()))
    return int(np.clip(1 + math.ceil(words / 90.0), 1, 8))


def compute_node_dps(nodes: Dict[int, Node], max_budget: int) -> Dict[int, np.ndarray]:
    cache: Dict[int, np.ndarray] = {}
    def solve(node_id: int) -> np.ndarray:
        if node_id in cache:
            return cache[node_id]
        node = nodes[node_id]
        out = np.full(max_budget + 1, -np.inf, dtype=float)
        out[0] = 0.0
        if node.latency <= max_budget:
            acc = np.full(max_budget + 1, -np.inf, dtype=float)
            acc[node.latency] = node.reward
            for child_id in node.children:
                child = solve(child_id)
                nxt = np.full(max_budget + 1, -np.inf, dtype=float)
                for b in range(max_budget + 1):
                    if not np.isfinite(acc[b]):
                        continue
                    for cb in range(max_budget - b + 1):
                        if np.isfinite(child[cb]):
                            nxt[b + cb] = max(nxt[b + cb], acc[b] + child[cb])
                acc = np.maximum(acc, nxt)
            out = np.maximum(out, acc)
        cache[node_id] = out
        return out
    for node_id in nodes:
        solve(node_id)
    return cache


def forest_oracle(nodes: Dict[int, Node], roots: Sequence[int], budget: int, node_dps: Dict[int, np.ndarray]) -> float:
    dp = np.full(budget + 1, -np.inf, dtype=float)
    dp[0] = 0.0
    for root_id in roots:
        root = node_dps[root_id][: budget + 1]
        nxt = np.full(budget + 1, -np.inf, dtype=float)
        for b in range(budget + 1):
            if not np.isfinite(dp[b]):
                continue
            for rb in range(budget - b + 1):
                if np.isfinite(root[rb]):
                    nxt[b + rb] = max(nxt[b + rb], dp[b] + root[rb])
        dp = nxt
    return float(np.max(dp))


def target_cluster_ci(rows, policy_a: str, policy_b: str, budget: int, seed_offset: int = 0):
    by = {}
    for r in rows:
        if r["budget"] != budget or r["policy"] not in (policy_a, policy_b):
            continue
        by.setdefault(r["target"], []).append((r["policy"], r["oracle_fraction"]))
    deltas = []
    for vals in by.values():
        aa = [v for p, v in vals if p == policy_a]
        bb = [v for p, v in vals if p == policy_b]
        if aa and bb:
            deltas.append(float(np.mean(aa) - np.mean(bb)))
    x = np.asarray(deltas, float)
    if len(x) == 0:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")], "positive_targets": 0, "n_targets": 0}
    rng = np.random.default_rng(SEED + 500 + seed_offset)
    reps = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        reps[i] = float(np.mean(x[rng.integers(0, len(x), len(x))]))
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return {"mean": float(np.mean(x)), "ci95": [float(lo), float(hi)], "positive_targets": int(np.sum(x > 0)), "n_targets": int(len(x))}


def bucket_stats(rows, policy_a: str, policy_b: str, budget: int):
    vals = [[] for _ in range(8)]
    pair = {}
    for r in rows:
        if r["budget"] != budget or r["policy"] not in (policy_a, policy_b):
            continue
        key = (r["source"], r["target"])
        pair.setdefault(key, {})[r["policy"]] = r["oracle_fraction"]
    for (_source, target), d in pair.items():
        if policy_a not in d or policy_b not in d:
            continue
        b = int(hashlib.sha256(target.encode()).hexdigest()[:8], 16) % 8
        vals[b].append(d[policy_a] - d[policy_b])
    means = [float(np.mean(v)) if v else float("nan") for v in vals]
    return {"means": means, "positive": int(sum(np.isfinite(x) and x > 0 for x in means))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("pilot", "confirm"), default="pilot")
    args = ap.parse_args()

    out = Path(os.environ.get("AP_LM3A_OUT", "artifacts/ap_lm3a"))
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"; raw.mkdir(exist_ok=True)
    graph_tar = raw / "graph.tar.gz"; html_tar = raw / "html.tar.gz"; text_tar = raw / "plaintext.tar.gz"
    download(GRAPH_URL, graph_tar); download(HTML_URL, html_tar); download(SNAP_TEXT_URL, text_tar)

    articles, graph_links, missions = extract_graph(graph_tar, raw / "graph")
    idx = {a: i for i, a in enumerate(articles)}
    occ, anchors, contexts, edge_coverage = build_visible_link_corpus(html_tar, articles, graph_links)
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
    title_emb = enc.encode([a.replace("_", " ") for a in articles], batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    anchor_emb = enc.encode(anchors, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    context_emb = enc.encode(contexts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    body_emb = enc.encode(answer_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    latencies = np.asarray([latency_from_text(t) for t in answer_texts], int)

    @lru_cache(maxsize=120000)
    def visible_edges(src: int, goal: int):
        gv = title_emb[goal]
        best = {}
        for v, ai, ci in occ[src]:
            if v == src:
                continue
            sa = float(np.dot(anchor_emb[ai], gv)); sc = float(np.dot(context_emb[ci], gv)); st = float(np.dot(title_emb[v], gv))
            vis = 0.55 * sa + 0.25 * sc + 0.20 * st
            cur = best.get(v)
            if cur is None or vis > cur[0]:
                best[v] = (vis, sa, sc, st)
        return tuple(sorted(((v,) + x for v, x in best.items()), key=lambda z: (-z[1], z[0])))

    @lru_cache(maxsize=120000)
    def true_reward(article: int, goal: int):
        sim = float(np.dot(body_emb[article], body_emb[goal]))
        return max(0.0, sim) ** 2

    def build_tree(source: int, goal: int):
        nodes = {}; roots = []; used_articles = {source}; next_id = 0
        def add_node(article: int, depth: int, parent_id: Optional[int], edge_tuple) -> int:
            nonlocal next_id
            _, vis, sa, sc, st = edge_tuple
            nid = next_id; next_id += 1
            nodes[nid] = Node(nid, article, depth, parent_id, float(sa), float(sc), float(st), float(vis), float(true_reward(article, goal)), int(latencies[article]), [])
            return nid
        root_edges = [e for e in visible_edges(source, goal) if e[0] not in used_articles][:N_ROOTS]
        for e in root_edges:
            if len(nodes) >= MAX_NODES: break
            used_articles.add(e[0]); roots.append(add_node(e[0], 0, None, e))
        queue = list(roots)
        while queue and len(nodes) < MAX_NODES:
            nid = queue.pop(0); node = nodes[nid]
            if node.depth >= MAX_DEPTH: continue
            added = 0
            for e in visible_edges(node.article, goal):
                art = e[0]
                if art in used_articles: continue
                used_articles.add(art); cid = add_node(art, node.depth + 1, nid, e)
                node.children.append(cid); queue.append(cid); added += 1
                if added >= MAX_CHILDREN or len(nodes) >= MAX_NODES: break
        return nodes, roots

    def feature(nodes, node_id, remaining, budget, frontier):
        n = nodes[node_id]
        parent_reward = nodes[n.parent_id].reward if n.parent_id is not None else 0.0
        rank = 1 + sum(nodes[x].visible_score > n.visible_score for x in frontier)
        return np.asarray([n.anchor_score, n.context_score, n.title_score, n.visible_score, math.log1p(n.latency), n.depth / MAX_DEPTH,
                           math.log1p(parent_reward), remaining / budget, math.log1p(len(frontier)), n.visible_score / n.latency,
                           rank / max(1, len(frontier))], dtype=float)

    def greedy_choice(nodes, frontier, remaining):
        eligible = [nid for nid in frontier if nodes[nid].latency <= remaining]
        if not eligible: return None
        return max(eligible, key=lambda nid: (nodes[nid].visible_score / nodes[nid].latency, nodes[nid].visible_score, -nodes[nid].latency))

    def collect_teacher(task, budget):
        source, target = task; nodes, roots = build_tree(idx[source], idx[target])
        if len(roots) < 2: return [], [], []
        dps = compute_node_dps(nodes, max(BUDGETS)); remaining = budget; frontier = list(roots); xs = []; y_rec = []; y_imm = []
        while True:
            eligible = [nid for nid in frontier if nodes[nid].latency <= remaining]
            if not eligible: break
            for nid in eligible:
                xs.append(feature(nodes, nid, remaining, budget, frontier)); y_rec.append(float(np.max(dps[nid][:remaining + 1]))); y_imm.append(float(nodes[nid].reward))
            chosen = greedy_choice(nodes, frontier, remaining)
            if chosen is None: break
            frontier.remove(chosen); n = nodes[chosen]; remaining -= n.latency; frontier.extend(n.children)
        return xs, y_rec, y_imm

    rng = np.random.default_rng(SEED)
    targets = sorted({t for s, t in missions if s in idx and t in idx}); rng.shuffle(targets); nt = len(targets)
    fit_targets = set(targets[: int(0.40 * nt)]); pilot_targets = set(targets[int(0.40 * nt): int(0.60 * nt)]); test_targets = set(targets[int(0.60 * nt):])
    fit_tasks = choose_tasks(missions, fit_targets, FIT_N, np.random.default_rng(SEED + 1))
    eval_tasks = choose_tasks(missions, pilot_targets if args.phase == "pilot" else test_targets, PILOT_N if args.phase == "pilot" else TEST_N,
                              np.random.default_rng(SEED + (2 if args.phase == "pilot" else 3)))
    print("split", len(fit_tasks), len(eval_tasks), "targets", len(fit_targets), len(pilot_targets), len(test_targets), "phase", args.phase)

    X = []; yr = []; yi = []
    for i, task in enumerate(fit_tasks):
        for budget in BUDGETS:
            xx, rr, ii = collect_teacher(task, budget); X.extend(xx); yr.extend(rr); yi.extend(ii)
        if i % 150 == 0: print("fit_task", i, "rows", len(yr))
    if len(X) < 1000: raise RuntimeError(f"too few teacher rows: {len(X)}")
    Xn = np.vstack(X)
    model_rec = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)).fit(Xn, np.asarray(yr))
    model_imm = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)).fit(Xn, np.asarray(yi))

    def predictions(model, nodes, frontier, remaining, budget, candidates):
        return model.predict(np.vstack([feature(nodes, nid, remaining, budget, frontier) for nid in candidates]))

    def compress(model, nodes, frontier, remaining, budget, mode):
        if mode is None: return list(frontier)
        if isinstance(mode, int): k = mode
        elif mode == "adaptive":
            if len(frontier) <= 4: return list(frontier)
            pred = predictions(model, nodes, frontier, remaining, budget, frontier); order = np.argsort(-pred); top8_idx = order[: min(8, len(order))]
            pos = np.maximum(pred[top8_idx], 0.0); frac = float(pos[: min(4, len(pos))].sum() / pos.sum()) if pos.sum() > 0 else 1.0
            k = 4 if frac >= 0.90 else 8
        else: raise ValueError(mode)
        if len(frontier) <= k: return list(frontier)
        pred = predictions(model, nodes, frontier, remaining, budget, frontier); order = np.argsort(-pred)[:k]
        return [frontier[int(i)] for i in order]

    def rollout(task, budget, policy, k_mode=None):
        source, target = task; nodes, roots = build_tree(idx[source], idx[target])
        if len(roots) < 2: return None
        dps = compute_node_dps(nodes, max(BUDGETS)); oracle = forest_oracle(nodes, roots, budget, dps)
        if oracle <= 0: return None
        remaining = budget; total = 0.0; frontier = list(roots); retained = []
        if policy.startswith("rec"): frontier = compress(model_rec, nodes, frontier, remaining, budget, k_mode)
        while True:
            eligible = [nid for nid in frontier if nodes[nid].latency <= remaining]
            if not eligible: break
            retained.append(len(frontier))
            if policy == "greedy": chosen = greedy_choice(nodes, frontier, remaining)
            elif policy == "dfs": chosen = eligible[0]
            elif policy.startswith("rec"):
                pred = predictions(model_rec, nodes, frontier, remaining, budget, eligible); chosen = eligible[int(np.argmax(pred))]
            elif policy == "imm":
                pred = predictions(model_imm, nodes, frontier, remaining, budget, eligible); chosen = eligible[int(np.argmax(pred))]
            else: raise ValueError(policy)
            if chosen is None: break
            frontier.remove(chosen); n = nodes[chosen]; remaining -= n.latency; total += n.reward
            if policy == "dfs": frontier = list(n.children) + frontier
            else: frontier.extend(n.children)
            if policy.startswith("rec"): frontier = compress(model_rec, nodes, frontier, remaining, budget, k_mode)
        return total, oracle, float(np.mean(retained)) if retained else 0.0, len(nodes), len(roots)

    rows = []
    policies = [("dfs", None), ("greedy", None), ("imm", None), ("rec_full", None), ("rec_k2", 2), ("rec_k4", 4), ("rec_k8", 8), ("rec_adapt", "adaptive")]
    for ti, task in enumerate(eval_tasks):
        for budget in BUDGETS:
            for name, kmode in policies:
                base_policy = "rec" if name.startswith("rec_") else name; z = rollout(task, budget, base_policy, kmode)
                if z is None: continue
                reward, oracle, retained, n_nodes, n_roots = z
                rows.append({"source": task[0], "target": task[1], "budget": budget, "policy": name, "reward": reward, "oracle": oracle,
                             "oracle_fraction": reward / oracle, "mean_retained": retained, "n_nodes": n_nodes, "n_roots": n_roots})
        if ti % 100 == 0: print("eval_task", ti, "rows", len(rows))

    aggregate = {}
    for budget in BUDGETS:
        aggregate[str(budget)] = {}
        for name, _ in policies:
            rr = [r for r in rows if r["budget"] == budget and r["policy"] == name]
            aggregate[str(budget)][name] = {"n": len(rr),
                "mean_oracle_fraction": float(np.mean([r["oracle_fraction"] for r in rr])) if rr else None,
                "mean_reward": float(np.mean([r["reward"] for r in rr])) if rr else None,
                "mean_retained": float(np.mean([r["mean_retained"] for r in rr])) if rr else None}

    rec_vs_imm_12 = target_cluster_ci(rows, "rec_full", "imm", PRIMARY_BUDGET, 1); rec_vs_imm_20 = target_cluster_ci(rows, "rec_full", "imm", LONG_BUDGET, 2)
    rec_vs_greedy_12 = target_cluster_ci(rows, "rec_full", "greedy", PRIMARY_BUDGET, 3); rec_vs_greedy_20 = target_cluster_ci(rows, "rec_full", "greedy", LONG_BUDGET, 4)
    b_rec_imm = bucket_stats(rows, "rec_full", "imm", PRIMARY_BUDGET); b_rec_greedy = bucket_stats(rows, "rec_full", "greedy", PRIMARY_BUDGET)

    capacity = {}; minimal_k = None
    for k in K_VALUES:
        name = f"rec_k{k}"; gaps = {}; ok = True
        for budget in BUDGETS:
            gap = float(aggregate[str(budget)]["rec_full"]["mean_oracle_fraction"] - aggregate[str(budget)][name]["mean_oracle_fraction"])
            gaps[str(budget)] = gap
            if gap > 0.01: ok = False
        capacity[str(k)] = gaps
        if minimal_k is None and ok: minimal_k = k

    adaptive = {}
    for budget in BUDGETS:
        k8 = aggregate[str(budget)]["rec_k8"]; ad = aggregate[str(budget)]["rec_adapt"]
        adaptive[str(budget)] = {"k8_minus_adaptive_performance": float(k8["mean_oracle_fraction"] - ad["mean_oracle_fraction"]),
            "k8_mean_retained": k8["mean_retained"], "adaptive_mean_retained": ad["mean_retained"],
            "memory_reduction_fraction": float(1.0 - ad["mean_retained"] / k8["mean_retained"]) if k8["mean_retained"] else 0.0}

    result = {"name": "AP-LM3A grounded natural-language recursive answer trees", "phase": args.phase,
        "construct": "real Wikispeedia answer text; querying reveals grounded body plus answer-generated visible concepts", "seed": SEED,
        "fit_n": len(fit_tasks), "eval_n_requested": PILOT_N if args.phase == "pilot" else TEST_N, "teacher_rows": len(yr), "budgets": list(BUDGETS),
        "edge_coverage": edge_coverage, "body_coverage": body_coverage, "aggregate": aggregate,
        "recursive_vs_immediate_b12": rec_vs_imm_12, "recursive_vs_immediate_b20": rec_vs_imm_20,
        "recursive_vs_greedy_b12": rec_vs_greedy_12, "recursive_vs_greedy_b20": rec_vs_greedy_20,
        "bucket_recursive_vs_immediate_b12": b_rec_imm, "bucket_recursive_vs_greedy_b12": b_rec_greedy,
        "capacity_full_minus_k": capacity, "minimal_k_within_1pp_at_both_budgets": minimal_k, "adaptive": adaptive,
        "boundaries": ["grounded article text is a retrieval-backed answer surrogate, not a generative LM response",
                       "hallucination is absent by construction", "latency is derived from fixed answer length, not measured API wall time",
                       "semantic target-alignment reward is not human comprehension", "pilot targets and confirm targets are target-disjoint"]}

    if args.phase == "confirm":
        h1 = rec_vs_imm_12; h2 = rec_vs_greedy_12
        pass_h1 = h1["mean"] >= 0.01 and h1["ci95"][0] > 0 and b_rec_imm["positive"] >= 6 and rec_vs_imm_20["mean"] >= -0.005
        pass_h2 = h2["mean"] >= 0.03 and h2["ci95"][0] > 0 and b_rec_greedy["positive"] >= 6 and rec_vs_greedy_20["mean"] >= 0.0
        result["pass_h1"] = bool(pass_h1); result["pass_h2"] = bool(pass_h2); result["decision"] = "PASS" if pass_h1 and pass_h2 else "FAIL"

    (out / f"AP_LM3A_{args.phase.upper()}_RESULTS.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = [f"# AP-LM3A {args.phase} summary", "",
        f"Recursive vs immediate B=12: {100*rec_vs_imm_12['mean']:+.3f} pp, CI [{100*rec_vs_imm_12['ci95'][0]:+.3f},{100*rec_vs_imm_12['ci95'][1]:+.3f}] pp.",
        f"Recursive vs greedy B=12: {100*rec_vs_greedy_12['mean']:+.3f} pp, CI [{100*rec_vs_greedy_12['ci95'][0]:+.3f},{100*rec_vs_greedy_12['ci95'][1]:+.3f}] pp.",
        f"B=20 recursive-vs-immediate: {100*rec_vs_imm_20['mean']:+.3f} pp; recursive-vs-greedy: {100*rec_vs_greedy_20['mean']:+.3f} pp.",
        f"Minimum K within 1 pp at both budgets: {minimal_k}.", f"Adaptive: {json.dumps(adaptive)}"]
    if args.phase == "confirm": summary.insert(2, f"Decision: **{result['decision']}**")
    (out / f"AP_LM3A_{args.phase.upper()}_SUMMARY.md").write_text("\n\n".join(summary) + "\n", encoding="utf-8")

    print("AP_LM3A_PHASE", args.phase)
    if args.phase == "confirm": print("AP_LM3A_DECISION", result["decision"])
    print("AP_LM3A_REC_IMM", json.dumps(rec_vs_imm_12)); print("AP_LM3A_REC_GREEDY", json.dumps(rec_vs_greedy_12))
    print("AP_LM3A_LONG_IMM", json.dumps(rec_vs_imm_20)); print("AP_LM3A_LONG_GREEDY", json.dumps(rec_vs_greedy_20))
    print("AP_LM3A_MIN_K", minimal_k); print("AP_LM3A_ADAPT", json.dumps(adaptive))


if __name__ == "__main__":
    main()
