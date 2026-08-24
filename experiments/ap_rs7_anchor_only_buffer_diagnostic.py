#!/usr/bin/env python3
"""AP-RS7: anchor-only scorer vs bounded-memory mechanism diagnostic.

Post-AP-RS5 internal diagnostic. RS5 showed that, for goal-directed Wikispeedia
navigation, anchor-only local navigation was materially stronger than the frozen
0.5 anchor + 0.5 paragraph scorer. RS6 then showed that removing unvisited-page
metadata did not rescue the equal-score bounded policy.

RS7 separates three questions under a stronger *visible anchor-only* local scorer:
  1. Does a short-lived previous-page buffer contain material counterfactual value?
  2. Is K=4 close to K=8 in oracle value?
  3. Can a strictly visible 3-feature learned gate recover deployable value?

This is an internal target-disjoint diagnostic on the same Wikispeedia universe,
not an independent external replication and not a human-comprehension experiment.
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
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.ap_rs5_real_anchor_context_policy import (
    GRAPH_URL, HTML_URL, download, extract_graph, norm_title, target_from_href,
)

SEED = 20260826
FIT_N = 700
TUNE_N = 500
TEST_N = 1200
K_LEARNED = 4
KS = (1, 2, 4, 8)
PRIMARY_BUDGET = 16
SAFETY_BUDGET = 32
N_BOOT = 2000
MARGINS = [-0.20, -0.10, -0.05, 0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 999.0]


def build_anchor_corpus(html_tar: Path, articles: list[str], graph_links: list[tuple[str, str]]):
    """Extract only visible anchor text; do not materialize paragraph embeddings."""
    article_idx = {a: i for i, a in enumerate(articles)}
    alias = {norm_title(a).casefold(): a for a in articles}
    graph_edges = {(s, t) for s, t in graph_links if s in article_idx and t in article_idx}
    anchors: list[str] = []
    anchor_idx: dict[str, int] = {}
    edge_occ: list[list[tuple[int, int]]] = [[] for _ in articles]
    seen_edges = set()

    def aid(text: str) -> int:
        text = " ".join(text.split()).strip()
        if text not in anchor_idx:
            anchor_idx[text] = len(anchors)
            anchors.append(text)
        return anchor_idx[text]

    with tarfile.open(html_tar, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        print("html files", len(members))
        for mi, m in enumerate(members):
            src = alias.get(norm_title(Path(m.name).name).casefold())
            if src is None:
                continue
            fh = tf.extractfile(m)
            if fh is None:
                continue
            soup = BeautifulSoup(fh.read(), "lxml")
            si = article_idx[src]
            for a in soup.find_all("a"):
                raw_t = target_from_href(a.get("href", ""))
                if raw_t is None:
                    continue
                tgt = alias.get(raw_t.casefold())
                if tgt is None or (src, tgt) not in graph_edges:
                    continue
                anchor = a.get_text(" ", strip=True) or tgt.replace("_", " ")
                edge_occ[si].append((article_idx[tgt], aid(anchor)))
                seen_edges.add((src, tgt))
            if mi % 1000 == 0:
                print("parsed html", mi, "visible edges", len(seen_edges))
    coverage = len(seen_edges) / max(1, len(graph_edges))
    return edge_occ, anchors, coverage


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
    out = Path(os.environ.get("AP_RS7_OUT", "artifacts/ap_rs7"))
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    graph_tar = raw / "wikispeedia_paths-and-graph.tar.gz"
    html_tar = raw / "wikispeedia_articles_html.tar.gz"
    download(GRAPH_URL, graph_tar)
    download(HTML_URL, html_tar)

    articles, graph_links, missions = extract_graph(graph_tar, raw / "graph")
    article_idx = {a: i for i, a in enumerate(articles)}
    edge_occ, anchors, coverage = build_anchor_corpus(html_tar, articles, graph_links)
    print("articles", len(articles), "graph links", len(graph_links), "missions", len(missions),
          "anchors", len(anchors), "coverage", coverage)
    if coverage < .85:
        raise RuntimeError(f"visible anchor coverage too low: {coverage:.3f}")

    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    title_emb = enc.encode([a.replace("_", " ") for a in articles], batch_size=128,
                           show_progress_bar=True, normalize_embeddings=True)
    anchor_emb = enc.encode(anchors, batch_size=128, show_progress_bar=True,
                            normalize_embeddings=True)

    @lru_cache(maxsize=40000)
    def scored(src_i: int, goal_i: int):
        gv = title_emb[goal_i]
        best = {}
        for tgt_i, ai in edge_occ[src_i]:
            s = float(np.dot(anchor_emb[ai], gv))
            if tgt_i not in best or s > best[tgt_i]:
                best[tgt_i] = s
        return tuple(sorted(best.items(), key=lambda x: (-x[1], x[0])))

    def available(src_i, goal_i, seen):
        return [(v, s) for v, s in scored(src_i, goal_i) if v not in seen]

    def local_rollout(start_i, goal_i, steps, budget, seen):
        cur = start_i
        ss = set(seen)
        if cur == goal_i:
            return True
        while steps < budget:
            xs = available(cur, goal_i, ss)
            if not xs:
                return False
            cur = xs[0][0]
            steps += 1
            if cur == goal_i:
                return True
            ss.add(cur)
        return False

    rng = np.random.default_rng(SEED)
    targets = sorted({t for s, t in missions if s in article_idx and t in article_idx})
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

    # Build one-intervention teacher on fit targets only, K=4.
    teacher = []
    for ti, (source, target) in enumerate(fit_tasks):
        cur = article_idx[source]
        goal = article_idx[target]
        seen = {cur}
        steps = 0
        prev = []
        while steps < PRIMARY_BUDGET and cur != goal:
            opts = available(cur, goal, seen)
            current_best = opts[0][1] if opts else 0.0
            if prev:
                cont = local_rollout(cur, goal, steps, PRIMARY_BUDGET, seen)
                for alt_i, alt_s, origin_count, rank in prev:
                    if steps + 2 > PRIMARY_BUDGET or alt_i in seen:
                        continue
                    back = local_rollout(alt_i, goal, steps + 2, PRIMARY_BUDGET, seen | {alt_i})
                    teacher.append({
                        "origin_count": origin_count,
                        "rel": alt_s - current_best,
                        "rank": rank,
                        "y": float(back) - float(cont),
                    })
            if not opts:
                break
            prev = [(v, s, len(opts), rank)
                    for rank, (v, s) in enumerate(opts[1:1 + K_LEARNED], start=2)]
            cur = opts[0][0]
            steps += 1
            if cur == goal:
                break
            seen.add(cur)
        if ti % 100 == 0:
            print("teacher tasks", ti, "rows", len(teacher))

    def fvec(r):
        return [math.log1p(r["origin_count"]), r["rel"], float(r["rank"])]

    X = np.asarray([fvec(r) for r in teacher], float)
    y = np.asarray([r["y"] for r in teacher], float)
    reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    reg.fit(X, y)
    pp = reg.predict(X)
    non = y != 0
    fit_auc = None
    if non.sum() and len(np.unique(y[non] > 0)) == 2:
        fit_auc = float(roc_auc_score((y[non] > 0).astype(int), pp[non]))
    print("teacher", len(y), "non-tie", int(non.sum()), "auc", fit_auc)

    def run_local(task, budget):
        source, target = task
        return local_rollout(article_idx[source], article_idx[target], 0, budget,
                             {article_idx[source]})

    def run_oracle(task, budget, k):
        """One-shot oracle over previous-page top-k alternatives; never harms by design."""
        source, target = task
        cur = article_idx[source]
        goal = article_idx[target]
        seen = {cur}
        steps = 0
        prev = []
        if cur == goal:
            return True, False
        while steps < budget:
            opts = available(cur, goal, seen)
            if prev and steps + 2 <= budget:
                cont = local_rollout(cur, goal, steps, budget, seen)
                if cont:
                    return True, False
                for alt_i, _alt_s in prev:
                    if alt_i in seen:
                        continue
                    if local_rollout(alt_i, goal, steps + 2, budget, seen | {alt_i}):
                        return True, True
            if not opts:
                return False, False
            prev = opts[1:1 + k]
            cur = opts[0][0]
            steps += 1
            if cur == goal:
                return True, False
            seen.add(cur)
        return False, False

    def run_learned(task, budget, margin):
        source, target = task
        cur = article_idx[source]
        goal = article_idx[target]
        seen = {cur}
        steps = 0
        prev = []
        used = False
        if cur == goal:
            return True, used
        while steps < budget:
            opts = available(cur, goal, seen)
            current_best = opts[0][1] if opts else 0.0
            if (not used) and steps < PRIMARY_BUDGET and prev and steps + 2 <= budget:
                rows = []
                valid = []
                for alt_i, alt_s, origin_count, rank in prev:
                    if alt_i in seen:
                        continue
                    rows.append({"origin_count": origin_count, "rel": alt_s - current_best,
                                 "rank": rank})
                    valid.append(alt_i)
                if rows:
                    pred = reg.predict(np.asarray([fvec(r) for r in rows], float))
                    j = int(np.argmax(pred))
                    if float(pred[j]) > margin:
                        cur = valid[j]
                        steps += 2
                        used = True
                        prev = []
                        if cur == goal:
                            return True, used
                        seen.add(cur)
                        continue
            if not opts:
                return False, used
            prev = [(v, s, len(opts), rank)
                    for rank, (v, s) in enumerate(opts[1:1 + K_LEARNED], start=2)]
            cur = opts[0][0]
            steps += 1
            if cur == goal:
                return True, used
            seen.add(cur)
        return False, used

    def eval_learned(tasks, margin):
        rows = []
        for task in tasks:
            l16 = run_local(task, 16)
            l32 = run_local(task, 32)
            p16, u16 = run_learned(task, 16, margin)
            p32, u32 = run_learned(task, 32, margin)
            rows.append({
                "source": task[0], "target": task[1],
                "local16": int(l16), "local32": int(l32),
                "policy16": int(p16), "policy32": int(p32),
                "d16": int(p16) - int(l16), "d32": int(p32) - int(l32),
                "intervened": int(u16 or u32),
            })
        return rows

    # Tune learned threshold on tune targets only.
    tune_grid = []
    selected = None
    for margin in MARGINS:
        rr = eval_learned(tune_tasks, margin)
        d16 = float(np.mean([r["d16"] for r in rr]))
        d32 = float(np.mean([r["d32"] for r in rr]))
        intr = float(np.mean([r["intervened"] for r in rr]))
        row = {"margin": margin, "d16": d16, "d32": d32, "intervention_rate": intr}
        tune_grid.append(row)
        if d32 >= -.005 and (selected is None or d16 > selected["d16"] or
                             (d16 == selected["d16"] and d32 > selected["d32"])):
            selected = row
    assert selected is not None
    margin = selected["margin"]
    print("selected", selected)

    learned_rows = eval_learned(test_tasks, margin)
    local16 = float(np.mean([r["local16"] for r in learned_rows]))
    local32 = float(np.mean([r["local32"] for r in learned_rows]))
    learned16 = float(np.mean([r["policy16"] for r in learned_rows]))
    learned32 = float(np.mean([r["policy32"] for r in learned_rows]))
    ld16 = float(np.mean([r["d16"] for r in learned_rows]))
    ld32 = float(np.mean([r["d32"] for r in learned_rows]))
    lci16 = target_cluster_ci(learned_rows, "d16", 1)
    lci32 = target_cluster_ci(learned_rows, "d32", 37)
    lb16 = bucket_stats(learned_rows, "d16")
    intr = float(np.mean([r["intervened"] for r in learned_rows]))

    # Oracle capacity curve on the exact same test missions.
    oracle = {}
    oracle_rows_by_k = {}
    for k in KS:
        rr = []
        for task in test_tasks:
            l16 = run_local(task, 16)
            l32 = run_local(task, 32)
            o16, u16 = run_oracle(task, 16, k)
            o32, u32 = run_oracle(task, 32, k)
            rr.append({
                "source": task[0], "target": task[1],
                "d16": int(o16) - int(l16), "d32": int(o32) - int(l32),
                "used16": int(u16), "used32": int(u32),
            })
        oracle_rows_by_k[k] = rr
        d16 = float(np.mean([r["d16"] for r in rr]))
        d32 = float(np.mean([r["d32"] for r in rr]))
        oracle[str(k)] = {
            "delta_S16": d16,
            "delta_S32": d32,
            "target_cluster_CI95_S16": target_cluster_ci(rr, "d16", 100 + k),
            "target_cluster_CI95_S32": target_cluster_ci(rr, "d32", 200 + k),
            "target_bucket_S16": bucket_stats(rr, "d16"),
            "use_rate_S16": float(np.mean([r["used16"] for r in rr])),
            "use_rate_S32": float(np.mean([r["used32"] for r in rr])),
        }
        print("oracle K", k, oracle[str(k)])

    k4 = oracle["4"]
    k8 = oracle["8"]
    conditions = {
        "test_n_ge_400": len(test_tasks) >= 400,
        "visible_anchor_coverage_ge_0_85": coverage >= .85,
        "oracle_K4_S16_gain_ge_2pp": k4["delta_S16"] >= .02,
        "oracle_K4_S16_CI_lower_gt_0": k4["target_cluster_CI95_S16"][0] > 0,
        "K8_minus_K4_S16_le_0_5pp": (k8["delta_S16"] - k4["delta_S16"]) <= .005,
        "learned_S16_gain_ge_1pp": ld16 >= .01,
        "learned_S16_CI_lower_gt_0": lci16[0] > 0,
        "learned_S32_mean_noninferior_minus_0_5pp": ld32 >= -.005,
        "learned_S32_CI_lower_ge_minus_1pp": lci32[0] >= -.01,
    }
    if conditions["oracle_K4_S16_gain_ge_2pp"] and conditions["oracle_K4_S16_CI_lower_gt_0"]:
        if conditions["learned_S16_gain_ge_1pp"] and conditions["learned_S16_CI_lower_gt_0"] and conditions["learned_S32_mean_noninferior_minus_0_5pp"] and conditions["learned_S32_CI_lower_ge_minus_1pp"]:
            diagnosis = "BUFFER_AND_GATE_SIGNAL"
        else:
            diagnosis = "GATE_BOTTLENECK_WITH_BUFFER_OPPORTUNITY"
    else:
        diagnosis = "LOW_BUFFER_OPPORTUNITY_UNDER_STRONG_ANCHOR_LOCAL"

    result = {
        "phase": "AP-RS7",
        "name": "anchor-only scorer / bounded-memory mechanism separation",
        "status": "POST_RS5_INTERNAL_DIAGNOSTIC",
        "diagnosis": diagnosis,
        "predeclared_diagnostic_conditions": conditions,
        "construct": "real Wikispeedia visible anchor text, real graph, human mission distribution; simulated navigation",
        "data": {
            "articles": len(articles), "graph_links": len(graph_links),
            "unique_missions": len(missions), "visible_anchor_edge_coverage": coverage,
            "unique_anchor_strings": len(anchors),
        },
        "split": {
            "target_disjoint": True,
            "fit_tasks": len(fit_tasks), "tune_tasks": len(tune_tasks), "test_tasks": len(test_tasks),
            "fit_targets": len(fit_targets), "tune_targets": len(tune_targets), "test_targets": len(test_targets),
            "note": "Internal split within the same Wikispeedia universe; not independent external replication.",
        },
        "scorer": {
            "mode": "anchor_only",
            "encoder": "sentence-transformers/all-MiniLM-L6-v2",
            "selected_after_RS5_finding": True,
        },
        "teacher": {
            "rows": int(len(y)), "non_tie_rows": int(non.sum()), "non_tie_sign_auc_fit": fit_auc,
            "features": ["log1p(origin_candidate_count)", "relative_anchor_score", "origin_rank"],
        },
        "tuning": {"grid": tune_grid, "selected_margin": margin, "safety_constraint": "mean S@32 harm <= 0.5pp"},
        "test": {
            "local_anchor": {"S16": local16, "S32": local32},
            "learned_top4": {
                "S16": learned16, "S32": learned32,
                "delta_S16": ld16, "delta_S32": ld32,
                "target_cluster_CI95_S16": lci16, "target_cluster_CI95_S32": lci32,
                "target_bucket_S16": lb16, "intervention_rate": intr,
            },
            "oracle_capacity": oracle,
        },
        "boundary": [
            "Anchor-only was motivated by the already-observed RS5 scorer comparison, so RS7 is diagnostic rather than independent confirmation.",
            "Oracle results measure available counterfactual value, not deployable performance.",
            "All learned-gate features are available from the current or immediately previous page.",
            "Navigation outcomes are simulated and do not establish human comprehension benefit.",
        ],
    }
    (out / "AP_RS7_RESULTS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "test_rows_learned.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=learned_rows[0].keys())
        w.writeheader(); w.writerows(learned_rows)
    md = f"""# AP-RS7 — Anchor-only scorer / memory mechanism separation

**Diagnosis: {diagnosis}**

- test missions: {len(test_tasks)} (target-disjoint internal split)
- anchor-only local S@16 / S@32: {local16:.4f} / {local32:.4f}
- learned top-4 delta S@16: {100*ld16:+.3f} pp, CI {lci16}
- learned top-4 delta S@32: {100*ld32:+.3f} pp, CI {lci32}
- learned intervention rate: {intr:.4f}
- oracle K1/K2/K4/K8 S@16 gains: {[100*oracle[str(k)]['delta_S16'] for k in KS]} pp
- oracle K4 S@16 CI: {k4['target_cluster_CI95_S16']}
- oracle K8-K4 S@16: {100*(k8['delta_S16']-k4['delta_S16']):+.3f} pp

## Interpretation boundary
RS7 was designed after observing RS5's anchor-only advantage. It is a mechanism diagnostic, not an independent external replication.
"""
    (out / "AP_RS7_SUMMARY.md").write_text(md, encoding="utf-8")
    print("AP_RS7_DIAGNOSIS", diagnosis)
    print("AP_RS7_LOCAL", local16, local32)
    print("AP_RS7_LEARNED", ld16, lci16, ld32, lci32, "intr", intr)
    print("AP_RS7_ORACLE", json.dumps(oracle))


if __name__ == "__main__":
    main()
