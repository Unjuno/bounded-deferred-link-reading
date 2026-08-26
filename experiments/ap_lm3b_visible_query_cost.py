#!/usr/bin/env python3
"""AP-LM3B development pilot: grounded recursive text with visible-only query cost.

AP-LM3A used destination answer length to define a cost that was visible before
querying. AP-LM3B removes that privileged-information path. A candidate's cost
is determined only by the visible anchor string. The final 40% target partition
is intentionally not evaluated by this development-only script.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tarfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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

SEED = 20260912
FIT_N = 700
PILOT_N = 400
BUDGETS = (12, 20)
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
    title_score: float
    visible_score: float
    query_cost: int
    reward: float
    children: List[int]


def visible_query_cost(anchor: str) -> int:
    """Cost known at query time; never reads destination answer text."""
    words = max(1, len(anchor.split()))
    return int(np.clip(1 + (words - 1) // 3, 1, 4))


def compute_node_dps(nodes: Dict[int, Node], max_budget: int) -> Dict[int, np.ndarray]:
    """Exact subtree value when a node itself must be queried before its children."""
    cache: Dict[int, np.ndarray] = {}

    def solve(node_id: int) -> np.ndarray:
        if node_id in cache:
            return cache[node_id]
        node = nodes[node_id]
        out = np.full(max_budget + 1, -np.inf, dtype=float)
        out[0] = 0.0
        if node.query_cost <= max_budget:
            acc = np.full(max_budget + 1, -np.inf, dtype=float)
            acc[node.query_cost] = node.reward
            for child_id in node.children:
                child = solve(child_id)
                nxt = np.full(max_budget + 1, -np.inf, dtype=float)
                for used in range(max_budget + 1):
                    if not np.isfinite(acc[used]):
                        continue
                    for child_used in range(max_budget - used + 1):
                        if np.isfinite(child[child_used]):
                            nxt[used + child_used] = max(
                                nxt[used + child_used], acc[used] + child[child_used]
                            )
                acc = np.maximum(acc, nxt)
            out = np.maximum(out, acc)
        cache[node_id] = out
        return out

    for node_id in nodes:
        solve(node_id)
    return cache


def forest_oracle(
    nodes: Dict[int, Node], roots: Sequence[int], budget: int, node_dps: Dict[int, np.ndarray]
) -> float:
    dp = np.full(budget + 1, -np.inf, dtype=float)
    dp[0] = 0.0
    for root_id in roots:
        root = node_dps[root_id][: budget + 1]
        nxt = np.full(budget + 1, -np.inf, dtype=float)
        for used in range(budget + 1):
            if not np.isfinite(dp[used]):
                continue
            for root_used in range(budget - used + 1):
                if np.isfinite(root[root_used]):
                    nxt[used + root_used] = max(
                        nxt[used + root_used], dp[used] + root[root_used]
                    )
        dp = nxt
    return float(np.max(dp))


def target_cluster_ci(rows, policy_a: str, policy_b: str, budget: int, offset: int):
    by_target = {}
    for r in rows:
        if r["budget"] != budget or r["policy"] not in (policy_a, policy_b):
            continue
        by_target.setdefault(r["target"], []).append((r["policy"], r["oracle_fraction"]))

    deltas = []
    for vals in by_target.values():
        aa = [v for p, v in vals if p == policy_a]
        bb = [v for p, v in vals if p == policy_b]
        if aa and bb:
            deltas.append(float(np.mean(aa) - np.mean(bb)))
    x = np.asarray(deltas, dtype=float)
    if not len(x):
        return {
            "mean": float("nan"),
            "ci95": [float("nan"), float("nan")],
            "positive_targets": 0,
            "n_targets": 0,
        }
    rng = np.random.default_rng(SEED + 1000 + offset)
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


def bucket_stats(rows, policy_a: str, policy_b: str, budget: int):
    pair = {}
    for r in rows:
        if r["budget"] != budget or r["policy"] not in (policy_a, policy_b):
            continue
        pair.setdefault((r["source"], r["target"]), {})[r["policy"]] = r["oracle_fraction"]
    vals = [[] for _ in range(8)]
    for (_source, target), d in pair.items():
        if policy_a not in d or policy_b not in d:
            continue
        b = int(hashlib.sha256(target.encode()).hexdigest()[:8], 16) % 8
        vals[b].append(d[policy_a] - d[policy_b])
    means = [float(np.mean(v)) if v else float("nan") for v in vals]
    return {"means": means, "positive": int(sum(np.isfinite(x) and x > 0 for x in means))}


def main():
    out = Path(os.environ.get("AP_LM3B_OUT", "artifacts/ap_lm3b"))
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
    occ, anchors, _contexts, edge_coverage = build_visible_link_corpus(
        html_tar, articles, graph_links
    )
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
        [a.replace("_", " ") for a in articles],
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    anchor_emb = enc.encode(
        anchors, batch_size=128, show_progress_bar=True, normalize_embeddings=True
    )
    body_emb = enc.encode(
        answer_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    anchor_costs = np.asarray([visible_query_cost(a) for a in anchors], dtype=int)

    @lru_cache(maxsize=120000)
    def visible_edges(src: int, goal: int):
        """All fields returned here are visible before querying the destination."""
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
        return tuple(
            sorted(
                ((v,) + x for v, x in best.items()),
                key=lambda z: (-z[1], z[4], z[0]),
            )
        )

    @lru_cache(maxsize=120000)
    def true_reward(article: int, goal: int):
        """Teacher/evaluation truth; never a pre-query feature for this candidate."""
        sim = float(np.dot(body_emb[article], body_emb[goal]))
        return max(0.0, sim) ** 2

    def build_tree(source: int, goal: int):
        nodes: Dict[int, Node] = {}
        roots: List[int] = []
        used_articles = {source}
        next_id = 0

        def add_node(article: int, depth: int, parent_id: Optional[int], e) -> int:
            nonlocal next_id
            _article, vis, sa, st, qcost = e
            nid = next_id
            next_id += 1
            nodes[nid] = Node(
                node_id=nid,
                article=article,
                depth=depth,
                parent_id=parent_id,
                anchor_score=float(sa),
                title_score=float(st),
                visible_score=float(vis),
                query_cost=int(qcost),
                reward=float(true_reward(article, goal)),
                children=[],
            )
            return nid

        root_edges = [e for e in visible_edges(source, goal) if e[0] not in used_articles][:N_ROOTS]
        for e in root_edges:
            if len(nodes) >= MAX_NODES:
                break
            used_articles.add(e[0])
            roots.append(add_node(e[0], 0, None, e))

        queue = list(roots)
        while queue and len(nodes) < MAX_NODES:
            nid = queue.pop(0)
            node = nodes[nid]
            if node.depth >= MAX_DEPTH:
                continue
            added = 0
            # Conceptually these links are revealed only after node is queried.
            for e in visible_edges(node.article, goal):
                article = e[0]
                if article in used_articles:
                    continue
                used_articles.add(article)
                cid = add_node(article, node.depth + 1, nid, e)
                node.children.append(cid)
                queue.append(cid)
                added += 1
                if added >= MAX_CHILDREN or len(nodes) >= MAX_NODES:
                    break
        return nodes, roots

    def feature(nodes, node_id, remaining, budget, frontier):
        n = nodes[node_id]
        # A child can only be in the frontier after its parent's answer was revealed.
        parent_observed_reward = nodes[n.parent_id].reward if n.parent_id is not None else 0.0
        rank = 1 + sum(nodes[x].visible_score > n.visible_score for x in frontier)
        return np.asarray(
            [
                n.anchor_score,
                n.title_score,
                n.visible_score,
                math.log1p(n.query_cost),
                n.depth / MAX_DEPTH,
                math.log1p(parent_observed_reward),
                remaining / budget,
                math.log1p(len(frontier)),
                n.visible_score / n.query_cost,
                rank / max(1, len(frontier)),
            ],
            dtype=float,
        )

    def greedy_choice(nodes, frontier, remaining):
        eligible = [nid for nid in frontier if nodes[nid].query_cost <= remaining]
        if not eligible:
            return None
        return max(
            eligible,
            key=lambda nid: (
                nodes[nid].visible_score / nodes[nid].query_cost,
                nodes[nid].visible_score,
                -nodes[nid].query_cost,
            ),
        )

    def collect_teacher(task, budget):
        source, target = task
        nodes, roots = build_tree(idx[source], idx[target])
        if len(roots) < 2:
            return [], [], []
        dps = compute_node_dps(nodes, max(BUDGETS))
        remaining = budget
        frontier = list(roots)
        xs, y_rec, y_imm = [], [], []
        while True:
            eligible = [nid for nid in frontier if nodes[nid].query_cost <= remaining]
            if not eligible:
                break
            for nid in eligible:
                xs.append(feature(nodes, nid, remaining, budget, frontier))
                y_rec.append(float(np.max(dps[nid][: remaining + 1])))
                y_imm.append(float(nodes[nid].reward))
            chosen = greedy_choice(nodes, frontier, remaining)
            if chosen is None:
                break
            frontier.remove(chosen)
            n = nodes[chosen]
            remaining -= n.query_cost
            frontier.extend(n.children)
        return xs, y_rec, y_imm

    rng = np.random.default_rng(SEED)
    targets = sorted({t for s, t in missions if s in idx and t in idx})
    rng.shuffle(targets)
    nt = len(targets)
    fit_targets = set(targets[: int(0.40 * nt)])
    pilot_targets = set(targets[int(0.40 * nt): int(0.60 * nt)])
    confirm_targets = set(targets[int(0.60 * nt):])
    fit_tasks = choose_tasks(
        missions, fit_targets, FIT_N, np.random.default_rng(SEED + 1)
    )
    pilot_tasks = choose_tasks(
        missions, pilot_targets, PILOT_N, np.random.default_rng(SEED + 2)
    )
    print(
        "split",
        len(fit_tasks),
        len(pilot_tasks),
        "targets",
        len(fit_targets),
        len(pilot_targets),
        len(confirm_targets),
        "confirm_untouched",
        True,
    )

    X, yr, yi = [], [], []
    for i, task in enumerate(fit_tasks):
        for budget in BUDGETS:
            xx, rr, ii = collect_teacher(task, budget)
            X.extend(xx)
            yr.extend(rr)
            yi.extend(ii)
        if i % 150 == 0:
            print("fit_task", i, "rows", len(yr))
    if len(X) < 1000:
        raise RuntimeError(f"too few teacher rows: {len(X)}")

    Xn = np.vstack(X)
    model_rec = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)).fit(
        Xn, np.asarray(yr, dtype=float)
    )
    model_imm = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)).fit(
        Xn, np.asarray(yi, dtype=float)
    )

    def predictions(model, nodes, frontier, remaining, budget, candidates):
        return model.predict(
            np.vstack(
                [feature(nodes, nid, remaining, budget, frontier) for nid in candidates]
            )
        )

    def compress(model, nodes, frontier, remaining, budget, mode):
        if mode is None:
            return list(frontier)
        if isinstance(mode, int):
            k = mode
        elif mode == "adaptive":
            if len(frontier) <= 4:
                return list(frontier)
            pred = predictions(model, nodes, frontier, remaining, budget, frontier)
            order = np.argsort(-pred)
            top = order[: min(8, len(order))]
            pos = np.maximum(pred[top], 0.0)
            frac = (
                float(pos[: min(4, len(pos))].sum() / pos.sum())
                if pos.sum() > 0
                else 1.0
            )
            k = 4 if frac >= 0.90 else 8
        else:
            raise ValueError(mode)
        if len(frontier) <= k:
            return list(frontier)
        pred = predictions(model, nodes, frontier, remaining, budget, frontier)
        order = np.argsort(-pred)[:k]
        return [frontier[int(i)] for i in order]

    def rollout(task, budget, policy, k_mode=None):
        source, target = task
        nodes, roots = build_tree(idx[source], idx[target])
        if len(roots) < 2:
            return None
        dps = compute_node_dps(nodes, max(BUDGETS))
        oracle = forest_oracle(nodes, roots, budget, dps)
        if oracle <= 0:
            return None
        remaining = budget
        total = 0.0
        frontier = list(roots)
        retained = []
        if policy == "rec":
            frontier = compress(model_rec, nodes, frontier, remaining, budget, k_mode)
        while True:
            eligible = [nid for nid in frontier if nodes[nid].query_cost <= remaining]
            if not eligible:
                break
            retained.append(len(frontier))
            if policy == "dfs":
                chosen = eligible[0]
            elif policy == "greedy":
                chosen = greedy_choice(nodes, frontier, remaining)
            elif policy == "imm":
                pred = predictions(model_imm, nodes, frontier, remaining, budget, eligible)
                chosen = eligible[int(np.argmax(pred))]
            elif policy == "rec":
                pred = predictions(model_rec, nodes, frontier, remaining, budget, eligible)
                chosen = eligible[int(np.argmax(pred))]
            else:
                raise ValueError(policy)
            if chosen is None:
                break
            frontier.remove(chosen)
            n = nodes[chosen]
            remaining -= n.query_cost
            total += n.reward
            if policy == "dfs":
                frontier = list(n.children) + frontier
            else:
                frontier.extend(n.children)
            if policy == "rec":
                frontier = compress(model_rec, nodes, frontier, remaining, budget, k_mode)
        return (
            total,
            oracle,
            float(np.mean(retained)) if retained else 0.0,
            len(nodes),
            len(roots),
        )

    policies = [
        ("dfs", "dfs", None),
        ("greedy", "greedy", None),
        ("imm", "imm", None),
        ("rec_full", "rec", None),
        ("rec_k2", "rec", 2),
        ("rec_k4", "rec", 4),
        ("rec_k8", "rec", 8),
        ("rec_adapt", "rec", "adaptive"),
    ]
    rows = []
    for ti, task in enumerate(pilot_tasks):
        for budget in BUDGETS:
            for name, policy, kmode in policies:
                z = rollout(task, budget, policy, kmode)
                if z is None:
                    continue
                reward, oracle, retained, n_nodes, n_roots = z
                rows.append(
                    {
                        "source": task[0],
                        "target": task[1],
                        "budget": budget,
                        "policy": name,
                        "reward": reward,
                        "oracle": oracle,
                        "oracle_fraction": reward / oracle,
                        "mean_retained": retained,
                        "n_nodes": n_nodes,
                        "n_roots": n_roots,
                    }
                )
        if ti % 100 == 0:
            print("pilot_task", ti, "rows", len(rows))

    aggregate = {}
    for budget in BUDGETS:
        aggregate[str(budget)] = {}
        for name, _policy, _kmode in policies:
            rr = [r for r in rows if r["budget"] == budget and r["policy"] == name]
            aggregate[str(budget)][name] = {
                "n": len(rr),
                "mean_oracle_fraction": float(np.mean([r["oracle_fraction"] for r in rr])) if rr else None,
                "mean_reward": float(np.mean([r["reward"] for r in rr])) if rr else None,
                "mean_retained": float(np.mean([r["mean_retained"] for r in rr])) if rr else None,
            }

    comparisons = {}
    for budget in BUDGETS:
        comparisons[str(budget)] = {
            "recursive_vs_immediate": target_cluster_ci(
                rows, "rec_full", "imm", budget, 10 + budget
            ),
            "recursive_vs_greedy": target_cluster_ci(
                rows, "rec_full", "greedy", budget, 20 + budget
            ),
            "bucket_recursive_vs_immediate": bucket_stats(
                rows, "rec_full", "imm", budget
            ),
            "bucket_recursive_vs_greedy": bucket_stats(
                rows, "rec_full", "greedy", budget
            ),
        }

    capacity = {}
    minimal_k = None
    for k in K_VALUES:
        gaps = {}
        ok = True
        for budget in BUDGETS:
            gap = float(
                aggregate[str(budget)]["rec_full"]["mean_oracle_fraction"]
                - aggregate[str(budget)][f"rec_k{k}"]["mean_oracle_fraction"]
            )
            gaps[str(budget)] = gap
            if gap > 0.01:
                ok = False
        capacity[str(k)] = gaps
        if minimal_k is None and ok:
            minimal_k = k

    adaptive = {}
    for budget in BUDGETS:
        k8 = aggregate[str(budget)]["rec_k8"]
        ad = aggregate[str(budget)]["rec_adapt"]
        adaptive[str(budget)] = {
            "k8_minus_adaptive_performance": float(
                k8["mean_oracle_fraction"] - ad["mean_oracle_fraction"]
            ),
            "k8_mean_retained": k8["mean_retained"],
            "adaptive_mean_retained": ad["mean_retained"],
            "memory_reduction_fraction": float(
                1.0 - ad["mean_retained"] / k8["mean_retained"]
            )
            if k8["mean_retained"]
            else 0.0,
        }

    promising_budgets = []
    for budget in BUDGETS:
        c = comparisons[str(budget)]
        a = c["recursive_vs_immediate"]
        g = c["recursive_vs_greedy"]
        k8_gap = capacity["8"][str(budget)]
        if a["mean"] > 0 and a["ci95"][0] > 0 and g["mean"] > 0 and k8_gap <= 0.01:
            promising_budgets.append(budget)

    result = {
        "name": "AP-LM3B visible-query-cost grounded recursive text pilot",
        "phase": "development_pilot",
        "seed": SEED,
        "fit_n": len(fit_tasks),
        "pilot_n_requested": PILOT_N,
        "teacher_rows": len(yr),
        "budgets": list(BUDGETS),
        "target_partition_counts": {
            "fit": len(fit_targets),
            "pilot": len(pilot_targets),
            "confirm_reserve_untouched": len(confirm_targets),
        },
        "edge_coverage": edge_coverage,
        "body_coverage": body_coverage,
        "query_cost_definition": "clip(1 + floor((visible_anchor_words-1)/3), 1, 4)",
        "aggregate": aggregate,
        "comparisons": comparisons,
        "capacity_full_minus_k": capacity,
        "minimal_k_within_1pp_at_both_budgets": minimal_k,
        "adaptive": adaptive,
        "promising_budgets_for_fresh_confirmation": promising_budgets,
        "development_gate": bool(promising_budgets),
        "boundaries": [
            "grounded article text is a retrieval-backed answer surrogate, not a generative LM response",
            "hallucination absent by construction",
            "query cost uses visible anchor string only and never hidden answer length",
            "semantic target-alignment reward is not human comprehension",
            "final 40 percent target partition was not evaluated by this pilot",
        ],
    }
    (out / "AP_LM3B_PILOT_RESULTS.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    summary = [
        "# AP-LM3B pilot summary",
        "",
        f"Development gate: **{result['development_gate']}**; promising budgets: {promising_budgets}.",
    ]
    for budget in BUDGETS:
        a = comparisons[str(budget)]["recursive_vs_immediate"]
        g = comparisons[str(budget)]["recursive_vs_greedy"]
        summary.extend(
            [
                f"B={budget} recursive vs immediate: {100*a['mean']:+.3f} pp, CI [{100*a['ci95'][0]:+.3f},{100*a['ci95'][1]:+.3f}] pp.",
                f"B={budget} recursive vs greedy: {100*g['mean']:+.3f} pp, CI [{100*g['ci95'][0]:+.3f},{100*g['ci95'][1]:+.3f}] pp.",
            ]
        )
    summary.extend(
        [
            f"Minimum K within 1 pp at both budgets: {minimal_k}.",
            f"Adaptive: {json.dumps(adaptive)}",
            "Final 40% target partition remains untouched.",
        ]
    )
    (out / "AP_LM3B_PILOT_SUMMARY.md").write_text(
        "\n\n".join(summary) + "\n", encoding="utf-8"
    )

    print("AP_LM3B_DEVELOPMENT_GATE", result["development_gate"])
    print("AP_LM3B_PROMISING_BUDGETS", promising_budgets)
    for budget in BUDGETS:
        print(
            "AP_LM3B_REC_IMM",
            budget,
            json.dumps(comparisons[str(budget)]["recursive_vs_immediate"]),
        )
        print(
            "AP_LM3B_REC_GREEDY",
            budget,
            json.dumps(comparisons[str(budget)]["recursive_vs_greedy"]),
        )
    print("AP_LM3B_MIN_K", minimal_k)
    print("AP_LM3B_ADAPT", json.dumps(adaptive))
    print("AP_LM3B_CONFIRM_RESERVE_TOUCHED", False)


if __name__ == "__main__":
    main()
