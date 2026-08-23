#!/usr/bin/env python3
"""AP-RS10: preregistered audit of full visited-set memory.

RS5/RS6/RS8/RS9 use a complete visited-page set to suppress cycles. That shared
state does not invalidate their paired policy deltas, but it prevents describing
the real-semantic simulations as strictly bounded-total-memory policies.

AP-RS10 isolates this issue with the strong anchor-only greedy local baseline.
It compares full visited memory against bounded tabu windows of the most recent
4 pages, the most recent 1 page, and no visited-page suppression. No model is
fit and no threshold is tuned.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from experiments.ap_rs5_real_anchor_context_policy import (
    GRAPH_URL, HTML_URL, build_visible_link_corpus, download, extract_graph,
)

SEED = 20260829
TEST_N = 6000
N_BOOT = 3000
BUDGETS = (16, 32)
MODES = ("full", "recent4", "recent1", "none")


def target_ci(rows, key, offset=0):
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


def buckets(rows, key):
    vv = [[] for _ in range(8)]
    for r in rows:
        b = int(hashlib.md5(r["target"].encode()).hexdigest()[:8], 16) % 8
        vv[b].append(r[key])
    means = [float(np.mean(v)) if v else float("nan") for v in vv]
    z = np.asarray([x for x in means if np.isfinite(x)], float)
    uc = float(z.std(ddof=1) / math.sqrt(len(z))) if len(z) > 1 else float("nan")
    return {"means": means, "positive": int(sum(x > 0 for x in means if np.isfinite(x))),
            "u_c": uc, "U_k1.96": float(1.96 * uc) if np.isfinite(uc) else None}


def main():
    out = Path(os.environ.get("AP_RS10_OUT", "artifacts/ap_rs10"))
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    gt = raw / "graph.tar.gz"
    ht = raw / "html.tar.gz"
    download(GRAPH_URL, gt)
    download(HTML_URL, ht)
    articles, links, missions = extract_graph(gt, raw / "graph")
    idx = {a: i for i, a in enumerate(articles)}
    occ, anchors, contexts, coverage = build_visible_link_corpus(ht, articles, links)
    if coverage < .85:
        raise RuntimeError(f"visible edge coverage too low: {coverage:.3f}")

    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    title_emb = enc.encode([a.replace("_", " ") for a in articles], batch_size=128,
                           show_progress_bar=True, normalize_embeddings=True)
    anchor_emb = enc.encode(anchors, batch_size=128, show_progress_bar=True, normalize_embeddings=True)

    @lru_cache(maxsize=50000)
    def scored(src, goal):
        gv = title_emb[goal]
        best = {}
        for v, ai, ci in occ[src]:
            s = float(np.dot(anchor_emb[ai], gv))
            if v not in best or s > best[v]:
                best[v] = s
        return tuple(sorted(best.items(), key=lambda x: (-x[1], x[0])))

    def run(task, budget, mode):
        source, target = task
        cur = idx[source]
        goal = idx[target]
        if cur == goal:
            return True, 0, 0
        full_seen = {cur}
        if mode == "recent4":
            recent = deque([cur], maxlen=4)
        elif mode == "recent1":
            recent = deque([cur], maxlen=1)
        else:
            recent = deque([], maxlen=1)
        revisits = 0
        steps = 0
        history = [cur]
        while steps < budget:
            if mode == "full":
                blocked = full_seen
            elif mode in ("recent4", "recent1"):
                blocked = set(recent)
            else:
                blocked = set()
            opts = [(v, s) for v, s in scored(cur, goal) if v not in blocked]
            if not opts:
                return False, steps, revisits
            nxt = opts[0][0]
            if nxt in full_seen:
                revisits += 1
            cur = nxt
            steps += 1
            if cur == goal:
                return True, steps, revisits
            full_seen.add(cur)
            if mode in ("recent4", "recent1"):
                recent.append(cur)
            history.append(cur)
        return False, steps, revisits

    rng = np.random.default_rng(SEED)
    usable = [(s, t) for s, t in missions if s in idx and t in idx and s != t]
    rng.shuffle(usable)
    tasks = usable[:min(TEST_N, len(usable))]
    rows = []
    for i, task in enumerate(tasks):
        r = {"source": task[0], "target": task[1]}
        for budget in BUDGETS:
            for mode in MODES:
                ok, steps, rev = run(task, budget, mode)
                r[f"{mode}_S{budget}"] = int(ok)
                r[f"{mode}_revisit_S{budget}"] = int(rev)
            r[f"recent4_minus_full_S{budget}"] = r[f"recent4_S{budget}"] - r[f"full_S{budget}"]
            r[f"recent1_minus_full_S{budget}"] = r[f"recent1_S{budget}"] - r[f"full_S{budget}"]
            r[f"none_minus_full_S{budget}"] = r[f"none_S{budget}"] - r[f"full_S{budget}"]
        rows.append(r)
        if i % 1000 == 0:
            print("tasks", i)

    summary = {}
    for budget in BUDGETS:
        base = float(np.mean([r[f"full_S{budget}"] for r in rows]))
        sb = {"full": base}
        for mode, off in (("recent4", 1), ("recent1", 17), ("none", 33)):
            s = float(np.mean([r[f"{mode}_S{budget}"] for r in rows]))
            key = f"{mode}_minus_full_S{budget}"
            delta = float(np.mean([r[key] for r in rows]))
            ci = target_ci(rows, key, offset=budget + off)
            sb[mode] = {
                "success": s,
                "delta_vs_full": delta,
                "target_cluster_CI95": ci,
                "target_buckets": buckets(rows, key),
                "mean_revisits": float(np.mean([r[f"{mode}_revisit_S{budget}"] for r in rows])),
            }
        summary[f"S{budget}"] = sb

    d16 = summary["S16"]["recent4"]["delta_vs_full"]
    d32 = summary["S32"]["recent4"]["delta_vs_full"]
    ci16 = summary["S16"]["recent4"]["target_cluster_CI95"]
    ci32 = summary["S32"]["recent4"]["target_cluster_CI95"]
    conditions = {
        "test_n_ge_2000": len(rows) >= 2000,
        "visible_edge_coverage_ge_0.85": coverage >= .85,
        "recent4_S16_mean_loss_le_0_5pp": d16 >= -.005,
        "recent4_S16_CI_lower_ge_minus_1pp": ci16[0] >= -.01,
        "recent4_S32_mean_loss_le_0_5pp": d32 >= -.005,
        "recent4_S32_CI_lower_ge_minus_1pp": ci32[0] >= -.01,
    }
    decision = "PASS" if all(conditions.values()) else "FAIL"
    result = {
        "phase": "AP-RS10",
        "name": "bounded visited-memory audit",
        "decision": decision,
        "preregistered_conditions": conditions,
        "construct": "real Wikispeedia anchor-only greedy navigation; audit of cycle-avoidance memory only",
        "data": {"test_tasks": len(rows), "articles": len(articles), "graph_links": len(links), "visible_edge_coverage": coverage},
        "policy": {"link_scorer": "anchor MiniLM cosine to target title", "visited_memory_modes": list(MODES), "no_model_fit": True},
        "results": summary,
        "boundary": [
            "This audits visited-page memory, not the deferred-alternative buffer itself.",
            "Recent-4 is O(1) in path depth; full visited-set is O(path length).",
            "Goal-directed navigation remains distinct from human comprehension.",
        ],
    }
    (out / "AP_RS10_RESULTS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "AP_RS10_SUMMARY.md").write_text(
        f"# AP-RS10 — Bounded visited-memory audit\n\n**Decision: {decision}**\n\n"
        f"- tasks: {len(rows)}\n"
        f"- recent4 - full S@16: {100*d16:+.3f} pp, CI {ci16}\n"
        f"- recent4 - full S@32: {100*d32:+.3f} pp, CI {ci32}\n"
        f"- recent1 - full S@16: {100*summary['S16']['recent1']['delta_vs_full']:+.3f} pp\n"
        f"- no-memory - full S@16: {100*summary['S16']['none']['delta_vs_full']:+.3f} pp\n",
        encoding="utf-8",
    )
    print("AP_RS10_DECISION", decision)
    print("RECENT4_S16", d16, ci16)
    print("RECENT4_S32", d32, ci32)
    print("RECENT1_S16", summary["S16"]["recent1"])
    print("NONE_S16", summary["S16"]["none"])


if __name__ == "__main__":
    main()
