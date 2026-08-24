#!/usr/bin/env python3
"""AP-RS12: oracle capacity of a short-lived anchor-only deferred-link buffer.

Purpose
-------
AP-RS5/6 showed near-null bounded-memory gains with the frozen equal
anchor+paragraph scorer. AP-RS8 then showed that a strictly visible learned K=4
gate is still near-null even with the materially stronger anchor-only local
scorer. RS12 asks the remaining mechanism question without fitting another gate:

    Is there material one-shot rescue value inside the previous page's top-K
    abandoned alternatives at all?

If oracle K=4 has substantial value while the already-run RS8 learned gate is
near zero, the bottleneck is value identification / triggering. If oracle K=4
itself is small, strong anchor-only local navigation leaves little opportunity
for deferred alternatives in this goal-directed construct.

This is a post-RS8 internal diagnostic on the same Wikispeedia universe, not an
independent external replication and not a human-comprehension experiment.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
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

SEED = 20260827  # mirrors AP-RS8 split generation
FIT_N = 700
TUNE_N = 500
TEST_N = 1200
KS = (1, 2, 4, 8)
BUDGETS = (16, 32)
N_BOOT = 2000


def choose_tasks(missions, target_set, n, rng):
    xs = [x for x in missions if x[1] in target_set]
    rng.shuffle(xs)
    return xs[:min(n, len(xs))]


def cluster_ci(rows, key, offset=0):
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
    finite = np.asarray([v for v in means if np.isfinite(v)], float)
    uc = float(finite.std(ddof=1) / math.sqrt(len(finite))) if len(finite) > 1 else float("nan")
    return {
        "means": means,
        "positive": int(sum(v > 0 for v in means if np.isfinite(v))),
        "u_c": uc,
        "U_k1.96": float(1.96 * uc) if np.isfinite(uc) else None,
    }


def main():
    out = Path(os.environ.get("AP_RS12_OUT", "artifacts/ap_rs12"))
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    raw.mkdir(exist_ok=True)

    graph_tar = raw / "graph.tar.gz"
    html_tar = raw / "html.tar.gz"
    download(GRAPH_URL, graph_tar)
    download(HTML_URL, html_tar)

    articles, links, missions = extract_graph(graph_tar, raw / "graph")
    idx = {a: i for i, a in enumerate(articles)}
    occ, anchors, _contexts, coverage = build_visible_link_corpus(html_tar, articles, links)
    if coverage < .85:
        raise RuntimeError(f"visible edge coverage too low: {coverage:.3f}")

    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    title_emb = enc.encode(
        [a.replace("_", " ") for a in articles],
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    anchor_emb = enc.encode(
        anchors,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    @lru_cache(maxsize=40000)
    def scored(src, goal):
        gv = title_emb[goal]
        best = {}
        for v, ai, _ci in occ[src]:
            s = float(np.dot(anchor_emb[ai], gv))
            if v not in best or s > best[v]:
                best[v] = s
        return tuple(sorted(best.items(), key=lambda z: (-z[1], z[0])))

    def avail(src, goal, seen):
        return [(v, s) for v, s in scored(src, goal) if v not in seen]

    def local_rollout(start, goal, steps, budget, seen):
        cur = start
        ss = set(seen)
        if cur == goal:
            return True
        while steps < budget:
            xs = avail(cur, goal, ss)
            if not xs:
                return False
            cur = xs[0][0]
            steps += 1
            if cur == goal:
                return True
            ss.add(cur)
        return False

    # Reproduce AP-RS8's target-disjoint split exactly, including RNG advances
    # from fit and tune task shuffling before drawing the test missions.
    rng = np.random.default_rng(SEED)
    targets = sorted({t for s, t in missions if s in idx and t in idx})
    rng.shuffle(targets)
    nt = len(targets)
    fit_targets = set(targets[:int(.40 * nt)])
    tune_targets = set(targets[int(.40 * nt):int(.60 * nt)])
    test_targets = set(targets[int(.60 * nt):])
    _fit = choose_tasks(missions, fit_targets, FIT_N, rng)
    _tune = choose_tasks(missions, tune_targets, TUNE_N, rng)
    test = choose_tasks(missions, test_targets, TEST_N, rng)
    print("test", len(test), "test targets", len(test_targets), "coverage", coverage)

    def run_local(task, budget):
        s, t = task
        return local_rollout(idx[s], idx[t], 0, budget, {idx[s]})

    def run_oracle(task, budget, k):
        """Never-harm one-shot oracle over the previous page's top-k alternatives.

        BACK+alternative traversal costs two actions. The oracle intervenes only
        when local continuation from the current state would fail within budget
        and at least one retained alternative would succeed under subsequent
        anchor-only local rollout.
        """
        source, target = task
        cur = idx[source]
        goal = idx[target]
        seen = {cur}
        steps = 0
        prev = []
        used = False
        if cur == goal:
            return True, used
        while steps < budget:
            xs = avail(cur, goal, seen)
            if prev and steps + 2 <= budget:
                cont = local_rollout(cur, goal, steps, budget, seen)
                if cont:
                    return True, used
                for alt, _score in prev:
                    if alt in seen:
                        continue
                    if local_rollout(alt, goal, steps + 2, budget, seen | {alt}):
                        return True, True
            if not xs:
                return False, used
            prev = xs[1:1 + k]
            cur = xs[0][0]
            steps += 1
            if cur == goal:
                return True, used
            seen.add(cur)
        return False, used

    base = {b: [run_local(task, b) for task in test] for b in BUDGETS}
    local_rates = {str(b): float(np.mean(base[b])) for b in BUDGETS}
    rows_by_k = {}
    capacity = {}

    for k in KS:
        rows = []
        for i, task in enumerate(test):
            r = {"source": task[0], "target": task[1]}
            for b in BUDGETS:
                ok, used = run_oracle(task, b, k)
                r[f"oracle{b}"] = int(ok)
                r[f"local{b}"] = int(base[b][i])
                r[f"d{b}"] = int(ok) - int(base[b][i])
                r[f"used{b}"] = int(used)
            rows.append(r)
        rows_by_k[k] = rows
        capacity[str(k)] = {}
        for b in BUDGETS:
            d = float(np.mean([r[f"d{b}"] for r in rows]))
            capacity[str(k)][f"delta_S{b}"] = d
            capacity[str(k)][f"target_cluster_CI95_S{b}"] = cluster_ci(rows, f"d{b}", 100*k+b)
            capacity[str(k)][f"target_bucket_S{b}"] = bucket_stats(rows, f"d{b}")
            capacity[str(k)][f"oracle_use_rate_S{b}"] = float(np.mean([r[f"used{b}"] for r in rows]))
        print("K", k, capacity[str(k)])

    # Paired incremental value of larger buffers.
    increments = {}
    for ka, kb in ((1, 2), (2, 4), (4, 8)):
        rr = []
        for a, b in zip(rows_by_k[ka], rows_by_k[kb]):
            rr.append({
                "source": a["source"], "target": a["target"],
                "d16": b["d16"] - a["d16"],
                "d32": b["d32"] - a["d32"],
            })
        increments[f"K{kb}_minus_K{ka}"] = {
            "delta_S16": float(np.mean([r["d16"] for r in rr])),
            "CI95_S16": cluster_ci(rr, "d16", 300 + ka + kb),
            "delta_S32": float(np.mean([r["d32"] for r in rr])),
            "CI95_S32": cluster_ci(rr, "d32", 400 + ka + kb),
        }

    k4 = capacity["4"]
    k8 = capacity["8"]
    conditions = {
        "test_n_ge_400": len(test) >= 400,
        "visible_edge_coverage_ge_0_85": coverage >= .85,
        "oracle_K4_S16_gain_ge_2pp": k4["delta_S16"] >= .02,
        "oracle_K4_S16_CI_lower_gt_0": k4["target_cluster_CI95_S16"][0] > 0,
        "K8_minus_K4_S16_le_0_5pp": (k8["delta_S16"] - k4["delta_S16"]) <= .005,
    }
    material = conditions["oracle_K4_S16_gain_ge_2pp"] and conditions["oracle_K4_S16_CI_lower_gt_0"]
    plateau = conditions["K8_minus_K4_S16_le_0_5pp"]
    if material and plateau:
        diagnosis = "MATERIAL_K4_BUFFER_OPPORTUNITY_WITH_K4_PLATEAU"
    elif material:
        diagnosis = "MATERIAL_BUFFER_OPPORTUNITY_BUT_K4_NOT_PLATEAU"
    else:
        diagnosis = "LOW_BUFFER_OPPORTUNITY_UNDER_STRONG_ANCHOR_LOCAL"

    result = {
        "phase": "AP-RS12",
        "name": "anchor-only oracle short-lived buffer capacity diagnostic",
        "status": "POST_RS8_INTERNAL_DIAGNOSTIC",
        "diagnosis": diagnosis,
        "predeclared_diagnostic_conditions": conditions,
        "construct": "real Wikispeedia visible anchor text, real graph, human mission distribution; simulated oracle diagnostic",
        "data": {
            "articles": len(articles),
            "graph_links": len(links),
            "unique_missions": len(missions),
            "visible_edge_coverage": coverage,
            "test_tasks": len(test),
            "test_targets": len(test_targets),
        },
        "split": {
            "matches_AP_RS8_generation": True,
            "target_disjoint": True,
            "seed": SEED,
        },
        "scorer": "visible anchor MiniLM cosine to target title",
        "memory": {
            "deferred_buffer": "immediately previous page only",
            "K_values": list(KS),
            "max_discretionary_returns": 1,
            "back_plus_alternative_action_cost": 2,
            "visited_set": "full path visited-set, matching AP-RS8; AP-RS10 shows this is distinct from deferred-option memory",
        },
        "local": {f"S{b}": local_rates[str(b)] for b in BUDGETS},
        "oracle_capacity": capacity,
        "paired_increments": increments,
        "comparison_boundary": {
            "AP_RS8_learned_gate_result": "near-null: S@16 +0.083pp, S@32 0 on its monitored run",
            "interpretation_if_RS12_material": "available buffer value exists but AP-RS8 visible learned gate failed to recover it",
        },
        "boundary": [
            "RS12 was designed after observing RS5/6/8 and is a mechanism diagnostic, not independent confirmation.",
            "Oracle access to counterfactual rollout outcomes is not deployable information.",
            "The full visited-set is retained only for cycle avoidance and is a separate memory construct from the top-K deferred alternatives.",
            "Goal-directed navigation does not establish human comprehension or retention benefit.",
        ],
    }
    (out / "AP_RS12_RESULTS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md = f"""# AP-RS12 — Anchor-only oracle buffer capacity\n\n**Diagnosis: {diagnosis}**\n\n- test missions: {len(test)}\n- anchor-local S@16 / S@32: {local_rates['16']:.4f} / {local_rates['32']:.4f}\n- oracle S@16 gains K=1/2/4/8: {[round(100*capacity[str(k)]['delta_S16'],3) for k in KS]} pp\n- oracle K4 S@16 CI: {k4['target_cluster_CI95_S16']}\n- K8-K4 S@16: {100*(k8['delta_S16']-k4['delta_S16']):+.3f} pp\n- paired increments: {increments}\n\n## Boundary\nPost-RS8 internal oracle diagnostic. Oracle gains quantify opportunity, not deployable performance.\n"""
    (out / "AP_RS12_SUMMARY.md").write_text(md, encoding="utf-8")
    print("AP_RS12_DIAGNOSIS", diagnosis)
    print("AP_RS12_LOCAL", local_rates)
    print("AP_RS12_CAPACITY", json.dumps(capacity))
    print("AP_RS12_INCREMENTS", json.dumps(increments))


if __name__ == "__main__":
    main()
