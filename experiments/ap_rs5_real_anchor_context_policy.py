#!/usr/bin/env python3
"""AP-RS5: causal bounded deferred-link policy on real Wikispeedia anchor/context semantics.

Primary construct
-----------------
Visible link semantics only: anchor text + containing paragraph, scored against the
known target title with a frozen 0.5/0.5 MiniLM cosine mixture. The hyperlink graph,
HTML, and source-target missions are real Wikispeedia data.

Fit/tune/test are target-disjoint. The compact policy keeps only the previous page's
top 4 abandoned alternatives and permits at most one discretionary return. A
3-feature counterfactual trajectory-utility Ridge model is fit only on fit targets;
its threshold is chosen only on tune targets under an S@32 safety constraint.

This is the principal computational real-semantic gate. It is still not a human
comprehension experiment.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tarfile
import urllib.parse
import urllib.request
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

GRAPH_URL = "https://snap.stanford.edu/data/wikispeedia/wikispeedia_paths-and-graph.tar.gz"
HTML_URL = "https://snap.stanford.edu/data/wikispeedia/wikispeedia_articles_html.tar.gz"
SEED = 20260824
K = 4
FIT_N = 700
TUNE_N = 500
TEST_N = 1200
N_BOOT = 2000
MARGINS = [-0.20, -0.10, -0.05, 0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 999.0]
PRIMARY_BUDGET = 16
SAFETY_BUDGET = 32


def dec(x: object) -> str:
    return urllib.parse.unquote(str(x))


def norm_title(s: str) -> str:
    s = dec(s).strip().replace(" ", "_")
    for suffix in (".htm", ".html", ".txt"):
        if s.lower().endswith(suffix):
            s = s[:-len(suffix)]
    return s


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    print("download", url)
    urllib.request.urlretrieve(url, path)


def noncomment_rows(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            yield next(csv.reader([line], delimiter="\t"))


def find_file(root: Path, name: str) -> Path:
    xs = list(root.rglob(name))
    if not xs:
        raise FileNotFoundError(name)
    return xs[0]


def extract_graph(graph_tar: Path, root: Path):
    if not root.exists() or not any(root.iterdir()):
        root.mkdir(exist_ok=True)
        with tarfile.open(graph_tar, "r:gz") as tf:
            tf.extractall(root)
    articles_file = find_file(root, "articles.tsv")
    links_file = find_file(root, "links.tsv")
    paths_file = find_file(root, "paths_finished.tsv")
    articles = [dec(r[0]) for r in noncomment_rows(articles_file) if r]
    links = [(dec(r[0]), dec(r[1])) for r in noncomment_rows(links_file) if len(r) >= 2]
    missions = set()
    for r in noncomment_rows(paths_file):
        if len(r) < 4:
            continue
        toks = [dec(x) for x in r[3].split(";")]
        arts = [x for x in toks if x != "<"]
        if len(arts) >= 2 and arts[0] != arts[-1]:
            missions.add((arts[0], arts[-1]))
    return articles, links, sorted(missions)


def target_from_href(href: str) -> str | None:
    if not href:
        return None
    href = href.split("#", 1)[0].split("?", 1)[0]
    if not href or href.startswith(("http:", "https:", "mailto:", "javascript:")):
        return None
    base = href.rstrip("/").split("/")[-1]
    if not base:
        return None
    return norm_title(base)


def build_visible_link_corpus(html_tar: Path, articles: list[str], graph_links: list[tuple[str, str]]):
    article_idx = {a: i for i, a in enumerate(articles)}
    alias = {norm_title(a).casefold(): a for a in articles}
    graph_edges = {(s, t) for s, t in graph_links if s in article_idx and t in article_idx}
    anchors: list[str] = []
    contexts: list[str] = []
    aidx: dict[str, int] = {}
    cidx: dict[str, int] = {}
    edge_occ: list[list[tuple[int, int, int]]] = [[] for _ in articles]
    seen_edges = set()

    def sid(text: str, table: dict[str, int], arr: list[str]) -> int:
        text = " ".join(text.split()).strip()
        if text not in table:
            table[text] = len(arr)
            arr.append(text)
        return table[text]

    with tarfile.open(html_tar, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        print("html files", len(members))
        for mi, m in enumerate(members):
            src_key = norm_title(Path(m.name).name).casefold()
            src = alias.get(src_key)
            if src is None:
                continue
            fh = tf.extractfile(m)
            if fh is None:
                continue
            data = fh.read()
            soup = BeautifulSoup(data, "lxml")
            si = article_idx[src]
            for a in soup.find_all("a"):
                raw_t = target_from_href(a.get("href", ""))
                if raw_t is None:
                    continue
                tgt = alias.get(raw_t.casefold())
                if tgt is None or (src, tgt) not in graph_edges:
                    continue
                anchor = a.get_text(" ", strip=True)
                if not anchor:
                    anchor = tgt.replace("_", " ")
                p = a.find_parent("p")
                if p is None:
                    p = a.parent
                context = p.get_text(" ", strip=True) if p is not None else anchor
                context = context[:1200] if context else anchor
                ai = sid(anchor, aidx, anchors)
                ci = sid(context, cidx, contexts)
                edge_occ[si].append((article_idx[tgt], ai, ci))
                seen_edges.add((src, tgt))
            if mi % 500 == 0:
                print("parsed html", mi, "unique visible edges", len(seen_edges))
    coverage = len(seen_edges) / max(1, len(graph_edges))
    return edge_occ, anchors, contexts, coverage


def choose_tasks(missions, target_set, n, rng):
    xs = [x for x in missions if x[1] in target_set]
    rng.shuffle(xs)
    return xs[:min(n, len(xs))]


def target_cluster_ci(rows, key, n_boot=N_BOOT, offset=0):
    by = defaultdict(list)
    for r in rows:
        by[r["target"]].append(r[key])
    groups = [(sum(v), len(v)) for v in by.values()]
    rng = np.random.default_rng(SEED + offset)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        ii = rng.integers(0, len(groups), len(groups))
        ss = sum(groups[i][0] for i in ii)
        nn = sum(groups[i][1] for i in ii)
        reps[b] = ss / nn
    return [float(np.quantile(reps, .025)), float(np.quantile(reps, .975))]


def bucket_stats(rows, key):
    vals = [[] for _ in range(8)]
    for r in rows:
        b = int(hashlib.md5(r["target"].encode()).hexdigest()[:8], 16) % 8
        vals[b].append(r[key])
    means = [float(np.mean(v)) if v else float("nan") for v in vals]
    finite = np.asarray([x for x in means if np.isfinite(x)], float)
    uc = float(finite.std(ddof=1) / math.sqrt(len(finite))) if len(finite) > 1 else float("nan")
    return {"means": means, "positive": int(sum(x > 0 for x in means if np.isfinite(x))),
            "u_c": uc, "U_k1.96": float(1.96 * uc) if np.isfinite(uc) else None}


def main():
    out = Path(os.environ.get("AP_RS5_OUT", "artifacts/ap_rs5"))
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    graph_tar = raw / "wikispeedia_paths-and-graph.tar.gz"
    html_tar = raw / "wikispeedia_articles_html.tar.gz"
    download(GRAPH_URL, graph_tar)
    download(HTML_URL, html_tar)

    articles, graph_links, missions = extract_graph(graph_tar, raw / "graph")
    article_idx = {a: i for i, a in enumerate(articles)}
    edge_occ, anchors, contexts, edge_coverage = build_visible_link_corpus(html_tar, articles, graph_links)
    print("articles", len(articles), "graph links", len(graph_links), "missions", len(missions))
    print("anchors", len(anchors), "contexts", len(contexts), "edge coverage", edge_coverage)
    if edge_coverage < 0.85:
        raise RuntimeError(f"visible anchor/context edge coverage too low: {edge_coverage:.3f}")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    title_emb = model.encode([a.replace("_", " ") for a in articles], batch_size=128,
                             show_progress_bar=True, normalize_embeddings=True)
    anchor_emb = model.encode(anchors, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    context_emb = model.encode(contexts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    @lru_cache(maxsize=30000)
    def scored_candidates(src_i: int, goal_i: int, mode: str = "equal"):
        gv = title_emb[goal_i]
        best = {}
        for tgt_i, ai, ci in edge_occ[src_i]:
            sa = float(np.dot(anchor_emb[ai], gv))
            sc = float(np.dot(context_emb[ci], gv))
            if mode == "anchor":
                s = sa
            elif mode == "context":
                s = sc
            else:
                s = 0.5 * sa + 0.5 * sc
            if tgt_i not in best or s > best[tgt_i]:
                best[tgt_i] = s
        return tuple(sorted(best.items(), key=lambda x: (-x[1], x[0])))

    def available(src_i, goal_i, visited, mode="equal"):
        xs = [(v, s) for v, s in scored_candidates(src_i, goal_i, mode) if v not in visited]
        return xs

    def local_rollout(start_i, goal_i, steps_used, budget, visited, mode="equal"):
        cur = start_i
        seen = set(visited)
        if cur == goal_i:
            return True
        while steps_used < budget:
            xs = available(cur, goal_i, seen, mode)
            if not xs:
                return False
            cur = xs[0][0]
            steps_used += 1
            if cur == goal_i:
                return True
            seen.add(cur)
        return False

    rng = np.random.default_rng(SEED)
    targets = sorted({t for s, t in missions if s in article_idx and t in article_idx})
    rng.shuffle(targets)
    n_t = len(targets)
    fit_targets = set(targets[:int(.40*n_t)])
    tune_targets = set(targets[int(.40*n_t):int(.60*n_t)])
    test_targets = set(targets[int(.60*n_t):])
    fit_tasks = choose_tasks(missions, fit_targets, FIT_N, rng)
    tune_tasks = choose_tasks(missions, tune_targets, TUNE_N, rng)
    test_tasks = choose_tasks(missions, test_targets, TEST_N, rng)
    print("task split", len(fit_tasks), len(tune_tasks), len(test_tasks), "targets", len(fit_targets), len(tune_targets), len(test_targets))

    # Counterfactual one-intervention teacher at budget 16.
    X = []
    y = []
    for ti, (source, target) in enumerate(fit_tasks):
        cur = article_idx[source]
        goal = article_idx[target]
        seen = {cur}
        steps = 0
        prev_alts = []
        while steps < PRIMARY_BUDGET and cur != goal:
            cur_opts = available(cur, goal, seen, "equal")
            current_best_score = cur_opts[0][1] if cur_opts else 0.0
            if prev_alts:
                cont = local_rollout(cur, goal, steps, PRIMARY_BUDGET, seen, "equal")
                for alt_i, alt_score, origin_count in prev_alts:
                    if steps + 2 > PRIMARY_BUDGET or alt_i in seen:
                        continue
                    back = local_rollout(alt_i, goal, steps + 2, PRIMARY_BUDGET, seen | {alt_i}, "equal")
                    X.append([math.log1p(len(edge_occ[alt_i])), math.log1p(origin_count), alt_score-current_best_score])
                    y.append(float(back) - float(cont))
            if not cur_opts:
                break
            chosen_i, chosen_s = cur_opts[0]
            alts = cur_opts[1:1+K]
            prev_alts = [(v, s, len(cur_opts)) for v, s in alts]
            cur = chosen_i
            steps += 1
            if cur == goal:
                break
            seen.add(cur)
        if ti % 100 == 0:
            print("teacher tasks", ti, "rows", len(y))

    X = np.asarray(X, float)
    y = np.asarray(y, float)
    reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    reg.fit(X, y)
    pred = reg.predict(X)
    mask = y != 0
    non_tie_auc = None
    if mask.sum() and len(np.unique(y[mask] > 0)) == 2:
        non_tie_auc = float(roc_auc_score((y[mask] > 0).astype(int), pred[mask]))
    print("teacher rows", len(y), "non-tie", int(mask.sum()), "auc", non_tie_auc)

    def run_policy(task, budget, margin=None, mode="equal"):
        source, target = task
        cur = article_idx[source]
        goal = article_idx[target]
        seen = {cur}
        steps = 0
        prev_alts = []
        intervened = False
        if cur == goal:
            return True, steps, False
        while steps < budget:
            cur_opts = available(cur, goal, seen, mode)
            current_best_score = cur_opts[0][1] if cur_opts else 0.0
            if margin is not None and (not intervened) and steps < PRIMARY_BUDGET and prev_alts and steps + 2 <= budget:
                feats = []
                valid = []
                for alt_i, alt_score, origin_count in prev_alts:
                    if alt_i in seen:
                        continue
                    feats.append([math.log1p(len(edge_occ[alt_i])), math.log1p(origin_count), alt_score-current_best_score])
                    valid.append(alt_i)
                if feats:
                    pp = reg.predict(np.asarray(feats, float))
                    j = int(np.argmax(pp))
                    if float(pp[j]) > margin:
                        cur = valid[j]
                        steps += 2  # browser Back to parent + click deferred alternative
                        intervened = True
                        prev_alts = []
                        if cur == goal:
                            return True, steps, True
                        seen.add(cur)
                        continue
            if not cur_opts:
                return False, steps, intervened
            chosen_i, chosen_s = cur_opts[0]
            if mode == "equal":
                prev_alts = [(v, s, len(cur_opts)) for v, s in cur_opts[1:1+K]]
            else:
                prev_alts = []
            cur = chosen_i
            steps += 1
            if cur == goal:
                return True, steps, intervened
            seen.add(cur)
        return False, steps, intervened

    def eval_tasks(tasks, margin=None, mode="equal"):
        rows = []
        for task in tasks:
            a16 = run_policy(task, 16, None, mode)[0]
            a32 = run_policy(task, 32, None, mode)[0]
            if margin is None:
                b16, b32, intr = a16, a32, False
            else:
                b16, _, i16 = run_policy(task, 16, margin, mode)
                b32, _, i32 = run_policy(task, 32, margin, mode)
                intr = i16 or i32
            rows.append({"source": task[0], "target": task[1], "local16": int(a16), "local32": int(a32),
                         "policy16": int(b16), "policy32": int(b32), "d16": int(b16)-int(a16),
                         "d32": int(b32)-int(a32), "intervened": int(intr)})
        return rows

    tune_grid = []
    selected = None
    for margin in MARGINS:
        rr = eval_tasks(tune_tasks, margin, "equal")
        d16 = float(np.mean([r["d16"] for r in rr]))
        d32 = float(np.mean([r["d32"] for r in rr]))
        intr = float(np.mean([r["intervened"] for r in rr]))
        tune_grid.append({"margin": margin, "d16": d16, "d32": d32, "intervention_rate": intr})
        if d32 >= -0.005 and (selected is None or d16 > selected["d16"] or (d16 == selected["d16"] and d32 > selected["d32"])):
            selected = tune_grid[-1]
    assert selected is not None
    margin = selected["margin"]
    print("selected margin", selected)

    test_rows = eval_tasks(test_tasks, margin, "equal")
    for r in test_rows:
        r["local_anchor16"] = int(run_policy((r["source"], r["target"]), 16, None, "anchor")[0])
        r["local_anchor32"] = int(run_policy((r["source"], r["target"]), 32, None, "anchor")[0])
        r["local_context16"] = int(run_policy((r["source"], r["target"]), 16, None, "context")[0])
        r["local_context32"] = int(run_policy((r["source"], r["target"]), 32, None, "context")[0])

    d16 = float(np.mean([r["d16"] for r in test_rows]))
    d32 = float(np.mean([r["d32"] for r in test_rows]))
    ci16 = target_cluster_ci(test_rows, "d16", offset=1)
    ci32 = target_cluster_ci(test_rows, "d32", offset=37)
    buckets16 = bucket_stats(test_rows, "d16")
    buckets32 = bucket_stats(test_rows, "d32")
    local16 = float(np.mean([r["local16"] for r in test_rows]))
    local32 = float(np.mean([r["local32"] for r in test_rows]))
    pol16 = float(np.mean([r["policy16"] for r in test_rows]))
    pol32 = float(np.mean([r["policy32"] for r in test_rows]))
    intr = float(np.mean([r["intervened"] for r in test_rows]))

    local_modes = {}
    for mode in ("anchor", "context", "equal"):
        if mode == "equal":
            s16, s32 = local16, local32
        else:
            s16 = float(np.mean([r[f"local_{mode}16"] for r in test_rows]))
            s32 = float(np.mean([r[f"local_{mode}32"] for r in test_rows]))
        local_modes[mode] = {"S16": s16, "S32": s32}

    conditions = {
        "test_n_ge_400": len(test_rows) >= 400,
        "visible_edge_coverage_ge_0.85": edge_coverage >= 0.85,
        "S16_gain_ge_2pp": d16 >= 0.02,
        "S16_target_CI_lower_gt_0": ci16[0] > 0,
        "S16_positive_target_buckets_ge_6_of_8": buckets16["positive"] >= 6,
        "S32_mean_noninferior_minus_0_5pp": d32 >= -0.005,
        "S32_target_CI_lower_ge_minus_1pp": ci32[0] >= -0.01,
    }
    decision = "PASS" if all(conditions.values()) else "FAIL"

    result = {
        "phase": "AP-RS5",
        "name": "real anchor/context causal bounded-deferred-link policy",
        "decision": decision,
        "preregistered_conditions": conditions,
        "construct": "real Wikispeedia HTML anchor + containing paragraph, real hyperlink graph, real human mission distribution; simulated policy outcome",
        "data": {"graph_url": GRAPH_URL, "html_url": HTML_URL, "articles": len(articles),
                 "graph_links": len(graph_links), "unique_missions": len(missions),
                 "visible_edge_coverage": edge_coverage, "unique_anchor_strings": len(anchors),
                 "unique_context_strings": len(contexts)},
        "split": {"target_disjoint": True, "fit_tasks": len(fit_tasks), "tune_tasks": len(tune_tasks),
                  "test_tasks": len(test_tasks), "fit_targets": len(fit_targets), "tune_targets": len(tune_targets),
                  "test_targets": len(test_targets)},
        "scorer": {"encoder": "sentence-transformers/all-MiniLM-L6-v2",
                   "equal_weight": {"anchor": 0.5, "containing_paragraph": 0.5},
                   "weight_retuned": False},
        "teacher": {"budget": 16, "rows": int(len(y)), "non_tie_rows": int(mask.sum()),
                    "non_tie_sign_auc_fit": non_tie_auc,
                    "features": ["log1p(candidate_outdegree)", "log1p(origin_candidate_count)", "relative_semantic_score"]},
        "tuning": {"safety_constraint": "mean S@32 harm <= 0.5pp on tune targets", "grid": tune_grid,
                   "selected_margin": margin},
        "test": {"local_modes": local_modes, "bounded_policy": {"S16": pol16, "S32": pol32,
                 "delta_S16": d16, "delta_S32": d32, "target_cluster_CI95_S16": ci16,
                 "target_cluster_CI95_S32": ci32, "target_bucket_S16": buckets16,
                 "target_bucket_S32": buckets32, "intervention_rate": intr}},
        "boundary": [
            "This is the first causal multi-hop policy test here using visible real anchor + containing-context semantics.",
            "Mission pairs come from human Wikispeedia, but policy trajectories are simulated.",
            "The result is not a human comprehension/retention effect.",
            "A sidebar agent that prefetches candidate-page content would define a different information regime and should be tested separately.",
        ],
    }
    (out / "AP_RS5_RESULTS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "test_rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=test_rows[0].keys())
        w.writeheader(); w.writerows(test_rows)
    md = f"""# AP-RS5 — Real anchor/context causal policy

**Decision: {decision}**

- test tasks: {len(test_rows)} (target-disjoint)
- visible edge coverage: {edge_coverage:.4f}
- selected margin: {margin}
- local equal S@16 / S@32: {local16:.4f} / {local32:.4f}
- bounded policy S@16 / S@32: {pol16:.4f} / {pol32:.4f}
- delta S@16: {100*d16:+.3f} pp, target-cluster CI {ci16}
- delta S@32: {100*d32:+.3f} pp, target-cluster CI {ci32}
- positive S@16 target buckets: {buckets16['positive']}/8
- intervention rate: {intr:.4f}

## Local scorer comparison
{json.dumps(local_modes, indent=2)}

## Claim boundary
Real HTML anchor/context and real graph are used, but the outcome is simulated navigation, not human comprehension.
"""
    (out / "AP_RS5_SUMMARY.md").write_text(md, encoding="utf-8")
    print("AP_RS5_DECISION", decision)
    print("AP_RS5_S16", d16, ci16, buckets16)
    print("AP_RS5_S32", d32, ci32, buckets32)


if __name__ == "__main__":
    main()
