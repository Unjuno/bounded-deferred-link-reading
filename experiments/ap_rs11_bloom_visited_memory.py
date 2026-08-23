#!/usr/bin/env python3
"""AP-RS11: preregistered fixed-memory Bloom visited-set audit.

AP-RS10 showed that replacing the full visited-page set with a recency window
materially harms real-semantic Wikispeedia navigation because old pages are
revisited. AP-RS11 tests a different O(1)-in-depth representation: a Bloom
filter for approximate visited membership.

Primary condition: 256 bits (32 bytes), 3 deterministic hashes. Bloom filters
have no false negatives after insertion; false positives can suppress some
unvisited links. Full exact visited-set navigation is the paired reference.
64/128/512-bit filters are prespecified secondary sensitivities.
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
    GRAPH_URL, HTML_URL, build_visible_link_corpus, download, extract_graph,
)

SEED = 20260830
TEST_N = 6000
N_BOOT = 3000
BUDGETS = (16, 32)
BIT_SIZES = (64, 128, 256, 512)
N_HASH = 3
PRIMARY_BITS = 256


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


class Bloom:
    __slots__ = ("m", "bits")
    def __init__(self, m):
        self.m = int(m)
        self.bits = 0
    def _positions(self, x):
        # Deterministic integer mixing; article indices are non-negative ints.
        z = int(x) + 0x9E3779B97F4A7C15
        for i in range(N_HASH):
            y = (z + i * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
            y ^= y >> 30
            y = (y * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
            y ^= y >> 27
            y = (y * 0x94D049BB133111EB) & ((1 << 64) - 1)
            y ^= y >> 31
            yield y % self.m
    def add(self, x):
        for p in self._positions(x):
            self.bits |= 1 << p
    def contains(self, x):
        return all((self.bits >> p) & 1 for p in self._positions(x))


def main():
    out = Path(os.environ.get("AP_RS11_OUT", "artifacts/ap_rs11"))
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"; raw.mkdir(exist_ok=True)
    gt = raw / "graph.tar.gz"; ht = raw / "html.tar.gz"
    download(GRAPH_URL, gt); download(HTML_URL, ht)
    articles, links, missions = extract_graph(gt, raw / "graph")
    idx = {a: i for i, a in enumerate(articles)}
    occ, anchors, contexts, coverage = build_visible_link_corpus(ht, articles, links)
    if coverage < .85:
        raise RuntimeError(f"visible edge coverage too low: {coverage:.3f}")

    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    te = enc.encode([a.replace("_", " ") for a in articles], batch_size=128,
                    show_progress_bar=True, normalize_embeddings=True)
    ae = enc.encode(anchors, batch_size=128, show_progress_bar=True, normalize_embeddings=True)

    @lru_cache(maxsize=50000)
    def scored(src, goal):
        gv = te[goal]; best = {}
        for v, ai, ci in occ[src]:
            s = float(np.dot(ae[ai], gv))
            if v not in best or s > best[v]:
                best[v] = s
        return tuple(sorted(best.items(), key=lambda x: (-x[1], x[0])))

    def run_full(task, budget):
        cur = idx[task[0]]; goal = idx[task[1]]; seen = {cur}; steps = 0
        while steps < budget and cur != goal:
            xs = [(v, s) for v, s in scored(cur, goal) if v not in seen]
            if not xs: return False, steps
            cur = xs[0][0]; steps += 1
            if cur == goal: return True, steps
            seen.add(cur)
        return cur == goal, steps

    def run_bloom(task, budget, m):
        cur = idx[task[0]]; goal = idx[task[1]]; bf = Bloom(m); bf.add(cur)
        exact_seen = {cur}  # diagnostics only; never used for choice
        steps = 0; false_pos = 0; checked = 0; true_revisit = 0
        while steps < budget and cur != goal:
            xs = []
            for v, s in scored(cur, goal):
                checked += 1
                blocked = bf.contains(v)
                if blocked:
                    if v not in exact_seen: false_pos += 1
                    continue
                xs.append((v, s))
            if not xs: return False, steps, false_pos, checked, true_revisit
            cur = xs[0][0]; steps += 1
            if cur in exact_seen: true_revisit += 1  # should be 0 absent Bloom false negatives
            if cur == goal: return True, steps, false_pos, checked, true_revisit
            exact_seen.add(cur); bf.add(cur)
        return cur == goal, steps, false_pos, checked, true_revisit

    rng = np.random.default_rng(SEED)
    tasks = [(s, t) for s, t in missions if s in idx and t in idx and s != t]
    rng.shuffle(tasks); tasks = tasks[:min(TEST_N, len(tasks))]
    rows = []
    for qi, task in enumerate(tasks):
        r = {"source": task[0], "target": task[1]}
        for b in BUDGETS:
            full, _ = run_full(task, b); r[f"full_S{b}"] = int(full)
            for m in BIT_SIZES:
                ok, _, fp, chk, rev = run_bloom(task, b, m)
                r[f"bloom{m}_S{b}"] = int(ok)
                r[f"bloom{m}_minus_full_S{b}"] = int(ok) - int(full)
                r[f"bloom{m}_fp_rate_S{b}"] = fp / max(1, chk)
                r[f"bloom{m}_revisit_S{b}"] = rev
        rows.append(r)
        if qi % 1000 == 0: print("tasks", qi)

    result_by_budget = {}
    for b in BUDGETS:
        rb = {"full": float(np.mean([r[f"full_S{b}"] for r in rows]))}
        for m in BIT_SIZES:
            key = f"bloom{m}_minus_full_S{b}"
            rb[str(m)] = {
                "success": float(np.mean([r[f"bloom{m}_S{b}"] for r in rows])),
                "delta_vs_full": float(np.mean([r[key] for r in rows])),
                "target_cluster_CI95": target_ci(rows, key, b + m),
                "target_buckets": buckets(rows, key),
                "mean_candidate_false_positive_rate": float(np.mean([r[f"bloom{m}_fp_rate_S{b}"] for r in rows])),
                "mean_true_revisits": float(np.mean([r[f"bloom{m}_revisit_S{b}"] for r in rows])),
            }
        result_by_budget[f"S{b}"] = rb

    p16 = result_by_budget["S16"][str(PRIMARY_BITS)]
    p32 = result_by_budget["S32"][str(PRIMARY_BITS)]
    cond = {
        "test_n_ge_2000": len(rows) >= 2000,
        "coverage_ge_0_85": coverage >= .85,
        "bloom256_S16_mean_loss_le_0_5pp": p16["delta_vs_full"] >= -.005,
        "bloom256_S16_CI_lower_ge_minus_1pp": p16["target_cluster_CI95"][0] >= -.01,
        "bloom256_S32_mean_loss_le_0_5pp": p32["delta_vs_full"] >= -.005,
        "bloom256_S32_CI_lower_ge_minus_1pp": p32["target_cluster_CI95"][0] >= -.01,
        "bloom256_no_true_revisits": p16["mean_true_revisits"] == 0 and p32["mean_true_revisits"] == 0,
    }
    decision = "PASS" if all(cond.values()) else "FAIL"
    result = {
        "phase": "AP-RS11", "name": "fixed-memory Bloom visited-set audit", "decision": decision,
        "preregistered_conditions": cond,
        "construct": "real Wikispeedia anchor-only greedy navigation; fixed-memory approximate visited membership",
        "data": {"test_tasks": len(rows), "articles": len(articles), "graph_links": len(links), "visible_edge_coverage": coverage},
        "memory": {"primary_bits": PRIMARY_BITS, "primary_bytes": PRIMARY_BITS // 8, "hashes": N_HASH,
                   "secondary_bit_sizes": list(BIT_SIZES), "depth_independent": True},
        "results": result_by_budget,
        "boundary": [
            "Bloom false positives may block genuinely unvisited candidates; false negatives do not occur after insertion.",
            "The exact set retained inside run_bloom is diagnostics-only and is not consulted by the policy.",
            "This audits cycle-avoidance memory, not deferred-option memory or human comprehension.",
        ],
    }
    (out / "AP_RS11_RESULTS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "AP_RS11_SUMMARY.md").write_text(
        f"# AP-RS11 — Fixed-memory Bloom visited set\n\n**Decision: {decision}**\n\n"
        f"- tasks: {len(rows)}\n- primary memory: 256 bits / 32 bytes, 3 hashes\n"
        f"- S@16 bloom256-full: {100*p16['delta_vs_full']:+.3f} pp, CI {p16['target_cluster_CI95']}\n"
        f"- S@32 bloom256-full: {100*p32['delta_vs_full']:+.3f} pp, CI {p32['target_cluster_CI95']}\n"
        f"- candidate FP rate S@16/S@32: {p16['mean_candidate_false_positive_rate']:.5f}/{p32['mean_candidate_false_positive_rate']:.5f}\n",
        encoding="utf-8")
    print("AP_RS11_DECISION", decision)
    print("BLOOM256_S16", p16)
    print("BLOOM256_S32", p32)
    for m in BIT_SIZES: print("SENS", m, result_by_budget['S16'][str(m)]['delta_vs_full'], result_by_budget['S32'][str(m)]['delta_vs_full'])

if __name__ == "__main__": main()
