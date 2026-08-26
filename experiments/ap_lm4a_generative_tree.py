#!/usr/bin/env python3
"""AP-LM4A: frozen generative answer-tree development pilot.

An actual causal LM generates answers and follow-up questions. All fit/pilot trees
are generated and persisted before any scheduling policy is fitted or evaluated.
The final 40% target partition is neither generated nor evaluated here.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import tarfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.ap_rs5_real_anchor_context_policy import (
    GRAPH_URL,
    HTML_URL,
    build_visible_link_corpus,
    choose_tasks,
    download,
    extract_graph,
)
from experiments.ap_rs4_article_body_semantics import SNAP_TEXT_URL, load_article_texts

SEED = 20260920
FIT_N = 32
PILOT_N = 24
BUDGETS = (8, 12)
N_ROOTS = 3
MAX_DEPTH = 2
MAX_CHILDREN = 2
MAX_NODES = 15
MAX_BODY_CHARS = 1800
MAX_NEW_TOKENS = 112
GEN_BATCH = 8
N_BOOT = 4000
RIDGE_ALPHA = 10.0
GEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
K_VALUES = (4, 8)


@dataclass
class Node:
    node_id: int
    query: str
    depth: int
    parent_id: Optional[int]
    query_score: float
    query_cost: int
    answer: str = ""
    reward: float = 0.0
    children: List[int] = None
    parse_ok: bool = False

    def __post_init__(self):
        if self.children is None:
            self.children = []


@dataclass
class Tree:
    source: str
    target: str
    nodes: Dict[int, Node]
    roots: List[int]


def norm_space(s: str) -> str:
    return " ".join(str(s).replace("_", " ").split())


def phrase_key(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", norm_space(s).lower()).strip()


def contains_target(query: str, target: str) -> bool:
    q = f" {phrase_key(query)} "
    t = phrase_key(target)
    if not t:
        return False
    return f" {t} " in q


def visible_query_cost(query: str) -> int:
    words = max(1, len(query.split()))
    return int(np.clip(1 + (words - 1) // 4, 1, 4))


def parse_generation(text: str) -> Tuple[str, List[str], bool]:
    text = text.strip()
    ma = re.search(r"ANSWER\s*:\s*(.*?)(?=\n\s*Q1\s*:|\n\s*Q2\s*:|$)", text, flags=re.I | re.S)
    m1 = re.search(r"(?:^|\n)\s*Q1\s*:\s*([^\n]+)", text, flags=re.I)
    m2 = re.search(r"(?:^|\n)\s*Q2\s*:\s*([^\n]+)", text, flags=re.I)
    answer = ma.group(1).strip() if ma else ""
    if not answer:
        # Recover answer text, but do not fabricate children.
        answer = re.split(r"(?:^|\n)\s*Q[12]\s*:", text, maxsplit=1, flags=re.I)[0].strip()
    qs = []
    for m in (m1, m2):
        if m:
            q = m.group(1).strip().strip("-• ")
            if q and not q.endswith("?"):
                q += "?"
            if q:
                qs.append(q)
    return answer, qs[:MAX_CHILDREN], bool(ma and m1 and m2)


def compute_node_dps(nodes: Dict[int, Node], max_budget: int) -> Dict[int, np.ndarray]:
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


def forest_oracle(nodes: Dict[int, Node], roots: Sequence[int], budget: int, dps) -> float:
    dp = np.full(budget + 1, -np.inf, dtype=float)
    dp[0] = 0.0
    for root_id in roots:
        root = dps[root_id][: budget + 1]
        nxt = np.full(budget + 1, -np.inf, dtype=float)
        for used in range(budget + 1):
            if not np.isfinite(dp[used]):
                continue
            for ru in range(budget - used + 1):
                if np.isfinite(root[ru]):
                    nxt[used + ru] = max(nxt[used + ru], dp[used] + root[ru])
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
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")], "positive_targets": 0, "n_targets": 0}
    rng = np.random.default_rng(SEED + 1000 + offset)
    reps = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        reps[i] = float(np.mean(x[rng.integers(0, len(x), len(x))]))
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return {"mean": float(np.mean(x)), "ci95": [float(lo), float(hi)], "positive_targets": int(np.sum(x > 0)), "n_targets": int(len(x))}


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
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    out = Path(os.environ.get("AP_LM4A_OUT", "artifacts/ap_lm4a"))
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

    rng = np.random.default_rng(SEED)
    targets = sorted({t for s, t in missions if s in idx and t in idx})
    rng.shuffle(targets)
    nt = len(targets)
    fit_targets = set(targets[: int(0.40 * nt)])
    pilot_targets = set(targets[int(0.40 * nt): int(0.60 * nt)])
    confirm_targets = set(targets[int(0.60 * nt):])
    fit_tasks = choose_tasks(missions, fit_targets, FIT_N, np.random.default_rng(SEED + 1))
    pilot_tasks = choose_tasks(missions, pilot_targets, PILOT_N, np.random.default_rng(SEED + 2))
    active_tasks = [("fit", x) for x in fit_tasks] + [("pilot", x) for x in pilot_tasks]
    print("partition", len(fit_tasks), len(pilot_tasks), len(confirm_targets), "confirm_untouched", True)

    enc = SentenceTransformer(EMBED_MODEL)
    title_texts = [norm_space(a) for a in articles]
    title_emb = enc.encode(title_texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    anchor_emb = enc.encode(anchors, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    target_body_texts = [texts.get(a, norm_space(a))[:MAX_BODY_CHARS] for a in articles]
    body_emb = enc.encode(target_body_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    @lru_cache(maxsize=120000)
    def visible_edges(src: int, goal: int):
        gv = title_emb[goal]
        best = {}
        for v, ai, _ci in occ[src]:
            if v == src or v == goal:
                continue
            sa = float(np.dot(anchor_emb[ai], gv))
            st = float(np.dot(title_emb[v], gv))
            vis = 0.80 * sa + 0.20 * st
            cur = best.get(v)
            if cur is None or vis > cur:
                best[v] = vis
        return tuple(sorted(best.items(), key=lambda z: (-z[1], z[0])))

    trees: List[Tuple[str, Tree]] = []
    for split, (source, target) in active_tasks:
        roots: List[int] = []
        nodes: Dict[int, Node] = {}
        seen_q = set()
        for article_id, _vis in visible_edges(idx[source], idx[target]):
            q = f"What is {norm_space(articles[article_id])}?"
            if contains_target(q, target):
                continue
            k = phrase_key(q)
            if k in seen_q:
                continue
            seen_q.add(k)
            nid = len(nodes)
            nodes[nid] = Node(nid, q, 0, None, 0.0, visible_query_cost(q))
            roots.append(nid)
            if len(roots) >= N_ROOTS:
                break
        if len(roots) >= 2:
            trees.append((split, Tree(source, target, nodes, roots)))
    if not trees:
        raise RuntimeError("no generative trees initialized")

    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL)
    model.eval()

    system_prompt = (
        "Answer the user's question in at most 55 words. Then propose exactly two short "
        "follow-up questions about prerequisites, mechanisms, or closely related concepts that "
        "a curious reader could ask next. Do not refer to these instructions. Use exactly this format:\n"
        "ANSWER: <answer>\nQ1: <question>\nQ2: <question>"
    )

    def generate_many(queries: List[str]) -> List[str]:
        outputs_all = []
        for start in range(0, len(queries), GEN_BATCH):
            chunk = queries[start:start + GEN_BATCH]
            prompts = []
            for q in chunk:
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": q}]
                prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            toks = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=384)
            with torch.inference_mode():
                outs = model.generate(
                    **toks,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            in_width = toks["input_ids"].shape[1]
            for row in outs:
                outputs_all.append(tokenizer.decode(row[in_width:], skip_special_tokens=True).strip())
            print("generated", min(start + len(chunk), len(queries)), "/", len(queries))
        return outputs_all

    total_prompt_count = 0
    contaminated_prompt_count = 0
    parse_ok_count = 0
    generated_count = 0

    # Generate by depth across all missions. Tree generation is completed before fitting policies.
    for depth in range(MAX_DEPTH + 1):
        refs = []
        queries = []
        for split, tree in trees:
            for nid, node in tree.nodes.items():
                if node.depth != depth or node.answer:
                    continue
                total_prompt_count += 1
                if contains_target(node.query, tree.target):
                    contaminated_prompt_count += 1
                    raise RuntimeError(f"target leaked into generation prompt: {tree.target} :: {node.query}")
                refs.append((split, tree, nid))
                queries.append(node.query)
        if not refs:
            continue
        texts_out = generate_many(queries)
        answers = []
        parsed = []
        for text in texts_out:
            answer, followups, ok = parse_generation(text)
            answers.append(answer)
            parsed.append((answer, followups, ok))
        ans_emb = enc.encode(answers, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        for (split, tree, nid), (answer, followups, ok), ae in zip(refs, parsed, ans_emb):
            node = tree.nodes[nid]
            node.answer = answer
            node.parse_ok = bool(ok)
            parse_ok_count += int(ok)
            generated_count += 1
            goal = idx[tree.target]
            sim = float(np.dot(ae, body_emb[goal])) if answer else 0.0
            node.reward = max(0.0, sim) ** 2
            if depth >= MAX_DEPTH:
                continue
            seen = {phrase_key(n.query) for n in tree.nodes.values()}
            for fq in followups:
                if len(tree.nodes) >= MAX_NODES:
                    break
                if contains_target(fq, tree.target):
                    continue
                k = phrase_key(fq)
                if not k or k in seen:
                    continue
                seen.add(k)
                cid = max(tree.nodes) + 1
                tree.nodes[cid] = Node(cid, fq, depth + 1, nid, 0.0, visible_query_cost(fq))
                node.children.append(cid)

        # Query scores are target-title semantic features, computed only after candidate text exists.
        new_refs = []
        new_queries = []
        for _split, tree in trees:
            for nid, node in tree.nodes.items():
                if node.depth <= depth + 1 and node.query_score == 0.0:
                    new_refs.append((tree, nid))
                    new_queries.append(node.query)
        if new_refs:
            qe = enc.encode(new_queries, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
            for (tree, nid), e in zip(new_refs, qe):
                tree.nodes[nid].query_score = float(np.dot(e, title_emb[idx[tree.target]]))

    if contaminated_prompt_count != 0:
        raise RuntimeError("target prompt contamination detected")

    # Persist frozen trees before fitting any scheduling model.
    frozen_path = out / "AP_LM4A_FROZEN_TREES.jsonl.gz"
    with gzip.open(frozen_path, "wt", encoding="utf-8") as f:
        for split, tree in trees:
            row = {
                "split": split,
                "source": tree.source,
                "target": tree.target,
                "roots": tree.roots,
                "nodes": {str(k): asdict(v) for k, v in tree.nodes.items()},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("FROZEN_TREES_WRITTEN", frozen_path, "trees", len(trees))

    # Free generator memory before policy fitting.
    del model
    del tokenizer

    fit_trees = [t for split, t in trees if split == "fit"]
    pilot_trees = [t for split, t in trees if split == "pilot"]

    def feature(tree: Tree, node_id: int, remaining: int, budget: int, frontier: Sequence[int]):
        n = tree.nodes[node_id]
        parent_reward = tree.nodes[n.parent_id].reward if n.parent_id is not None else 0.0
        rank = 1 + sum(tree.nodes[x].query_score > n.query_score for x in frontier)
        return np.asarray([
            n.query_score,
            math.log1p(n.query_cost),
            n.depth / MAX_DEPTH,
            math.log1p(parent_reward),
            remaining / budget,
            math.log1p(len(frontier)),
            n.query_score / n.query_cost,
            rank / max(1, len(frontier)),
        ], dtype=float)

    def greedy_choice(tree: Tree, frontier: Sequence[int], remaining: int):
        eligible = [nid for nid in frontier if tree.nodes[nid].query_cost <= remaining]
        if not eligible:
            return None
        return max(eligible, key=lambda nid: (tree.nodes[nid].query_score / tree.nodes[nid].query_cost, tree.nodes[nid].query_score, -tree.nodes[nid].query_cost))

    def collect_teacher(tree: Tree, budget: int):
        if len(tree.roots) < 2:
            return [], [], []
        dps = compute_node_dps(tree.nodes, max(BUDGETS))
        remaining = budget
        frontier = list(tree.roots)
        xs, yr, yi = [], [], []
        while True:
            frontier = [nid for nid in frontier if tree.nodes[nid].query_cost <= remaining]
            if not frontier:
                break
            for nid in frontier:
                xs.append(feature(tree, nid, remaining, budget, frontier))
                yr.append(float(np.max(dps[nid][: remaining + 1])))
                yi.append(float(tree.nodes[nid].reward))
            chosen = greedy_choice(tree, frontier, remaining)
            if chosen is None:
                break
            frontier.remove(chosen)
            n = tree.nodes[chosen]
            remaining -= n.query_cost
            frontier.extend(n.children)
        return xs, yr, yi

    X, yr, yi = [], [], []
    for i, tree in enumerate(fit_trees):
        for budget in BUDGETS:
            xx, rr, ii = collect_teacher(tree, budget)
            X.extend(xx); yr.extend(rr); yi.extend(ii)
        if i % 8 == 0:
            print("teacher_tree", i, "rows", len(yr))
    if len(X) < 200:
        raise RuntimeError(f"too few teacher rows: {len(X)}")
    Xn = np.vstack(X)
    model_rec = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)).fit(Xn, np.asarray(yr, dtype=float))
    model_imm = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)).fit(Xn, np.asarray(yi, dtype=float))

    def predict(model_, tree: Tree, frontier, remaining, budget, candidates):
        return model_.predict(np.vstack([feature(tree, nid, remaining, budget, frontier) for nid in candidates]))

    def compress(tree: Tree, frontier, remaining, budget, k):
        feasible = [nid for nid in frontier if tree.nodes[nid].query_cost <= remaining]
        if len(feasible) <= k:
            return feasible
        pred = predict(model_rec, tree, feasible, remaining, budget, feasible)
        order = np.argsort(-pred)[:k]
        return [feasible[int(i)] for i in order]

    def rollout(tree: Tree, budget: int, policy: str, k_mode: Optional[int] = None):
        dps = compute_node_dps(tree.nodes, max(BUDGETS))
        oracle = forest_oracle(tree.nodes, tree.roots, budget, dps)
        if oracle <= 0:
            return None
        remaining = budget
        total = 0.0
        frontier = list(tree.roots)
        retained = []
        while True:
            frontier = [nid for nid in frontier if tree.nodes[nid].query_cost <= remaining]
            if policy == "rec" and k_mode is not None:
                frontier = compress(tree, frontier, remaining, budget, k_mode)
            if not frontier:
                break
            retained.append(len(frontier))
            if policy == "greedy":
                chosen = greedy_choice(tree, frontier, remaining)
            elif policy == "imm":
                pred = predict(model_imm, tree, frontier, remaining, budget, frontier)
                chosen = frontier[int(np.argmax(pred))]
            elif policy == "rec":
                pred = predict(model_rec, tree, frontier, remaining, budget, frontier)
                chosen = frontier[int(np.argmax(pred))]
            else:
                raise ValueError(policy)
            frontier.remove(chosen)
            n = tree.nodes[chosen]
            remaining -= n.query_cost
            total += n.reward
            frontier.extend(n.children)
        return total, oracle, float(np.mean(retained)) if retained else 0.0

    policies = [("greedy", "greedy", None), ("imm", "imm", None), ("rec_full", "rec", None), ("rec_k4", "rec", 4), ("rec_k8", "rec", 8)]
    rows = []
    for ti, tree in enumerate(pilot_trees):
        for budget in BUDGETS:
            for name, policy, kmode in policies:
                z = rollout(tree, budget, policy, kmode)
                if z is None:
                    continue
                reward, oracle, retained = z
                rows.append({"source": tree.source, "target": tree.target, "budget": budget, "policy": name, "reward": reward, "oracle": oracle, "oracle_fraction": reward / oracle, "mean_retained": retained, "n_nodes": len(tree.nodes)})
        if ti % 6 == 0:
            print("pilot_tree", ti, "rows", len(rows))

    aggregate = {}
    for budget in BUDGETS:
        aggregate[str(budget)] = {}
        for name, _p, _k in policies:
            rr = [r for r in rows if r["budget"] == budget and r["policy"] == name]
            aggregate[str(budget)][name] = {
                "n": len(rr),
                "mean_oracle_fraction": float(np.mean([r["oracle_fraction"] for r in rr])) if rr else None,
                "mean_reward": float(np.mean([r["reward"] for r in rr])) if rr else None,
                "mean_retained": float(np.mean([r["mean_retained"] for r in rr])) if rr else None,
            }

    comparisons = {}
    capacity = {}
    promising = []
    parse_success = parse_ok_count / max(1, generated_count)
    mean_nodes = float(np.mean([len(t.nodes) for _s, t in trees]))
    for budget in BUDGETS:
        ri = target_cluster_ci(rows, "rec_full", "imm", budget, 10 + budget)
        rg = target_cluster_ci(rows, "rec_full", "greedy", budget, 20 + budget)
        bi = bucket_stats(rows, "rec_full", "imm", budget)
        bg = bucket_stats(rows, "rec_full", "greedy", budget)
        comparisons[str(budget)] = {"recursive_vs_immediate": ri, "recursive_vs_greedy": rg, "bucket_recursive_vs_immediate": bi, "bucket_recursive_vs_greedy": bg}
        capacity[str(budget)] = {
            "full_minus_k4": float(aggregate[str(budget)]["rec_full"]["mean_oracle_fraction"] - aggregate[str(budget)]["rec_k4"]["mean_oracle_fraction"]),
            "full_minus_k8": float(aggregate[str(budget)]["rec_full"]["mean_oracle_fraction"] - aggregate[str(budget)]["rec_k8"]["mean_oracle_fraction"]),
        }
        eval_n = aggregate[str(budget)]["rec_full"]["n"] or 0
        gate = (
            eval_n >= 15
            and parse_success >= 0.85
            and mean_nodes >= 4.0
            and ri["mean"] > 0
            and ri["ci95"][0] > 0
            and rg["mean"] > 0
            and bi["positive"] >= 5
        )
        if gate:
            promising.append(budget)

    all_nodes = [n for _s, t in trees for n in t.nodes.values() if n.answer]
    depth_counts = {}
    for n in all_nodes:
        depth_counts[str(n.depth)] = depth_counts.get(str(n.depth), 0) + 1
    result = {
        "name": "AP-LM4A frozen generative answer-tree development pilot",
        "phase": "development_pilot",
        "seed": SEED,
        "generator": {"model": GEN_MODEL, "do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
        "fit_n": len(fit_trees),
        "pilot_n": len(pilot_trees),
        "confirm_target_count_untouched": len(confirm_targets),
        "confirm_reserve_generated": False,
        "teacher_rows": len(yr),
        "edge_coverage": edge_coverage,
        "body_coverage": body_coverage,
        "generation": {
            "generated_nodes": generated_count,
            "parse_success": parse_success,
            "mean_nodes_per_tree": mean_nodes,
            "depth_counts": depth_counts,
            "mean_query_cost": float(np.mean([n.query_cost for n in all_nodes])) if all_nodes else None,
            "mean_reward": float(np.mean([n.reward for n in all_nodes])) if all_nodes else None,
            "target_prompt_contamination_fraction": contaminated_prompt_count / max(1, total_prompt_count),
        },
        "budgets": list(BUDGETS),
        "aggregate": aggregate,
        "comparisons": comparisons,
        "capacity": capacity,
        "promising_budgets_for_fresh_confirmation": promising,
        "development_gate": bool(promising),
        "boundaries": [
            "actual generative LM answers and follow-up questions",
            "generation tree frozen before policy fitting/evaluation",
            "semantic target alignment is not factual correctness",
            "query word-count cost is not measured wall-clock LM latency",
            "human comprehension and metacognitive unknown detection are not measured",
        ],
    }
    result_path = out / "AP_LM4A_PILOT_RESULTS.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = [
        "# AP-LM4A pilot summary",
        "",
        f"- development gate: **{result['development_gate']}**",
        f"- promising budgets: {promising}",
        f"- fit/pilot trees: {len(fit_trees)}/{len(pilot_trees)}",
        f"- generated nodes: {generated_count}",
        f"- parse success: {parse_success:.4f}",
        f"- mean nodes/tree: {mean_nodes:.3f}",
        f"- target-prompt contamination: {result['generation']['target_prompt_contamination_fraction']:.6f}",
    ]
    for budget in BUDGETS:
        c = comparisons[str(budget)]
        summary += [
            "",
            f"## B={budget}",
            f"- recursive vs immediate: {c['recursive_vs_immediate']}",
            f"- recursive vs greedy: {c['recursive_vs_greedy']}",
            f"- buckets rec-vs-imm: {c['bucket_recursive_vs_immediate']}",
            f"- capacity: {capacity[str(budget)]}",
        ]
    (out / "AP_LM4A_PILOT_SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("AP_LM4A_DEVELOPMENT_GATE", result["development_gate"])
    print("AP_LM4A_PROMISING_BUDGETS", promising)
    print("AP_LM4A_GENERATION", json.dumps(result["generation"]))
    for budget in BUDGETS:
        print("AP_LM4A_REC_IMM", budget, json.dumps(comparisons[str(budget)]["recursive_vs_immediate"]))
        print("AP_LM4A_REC_GREEDY", budget, json.dumps(comparisons[str(budget)]["recursive_vs_greedy"]))
        print("AP_LM4A_BUCKETS", budget, json.dumps(comparisons[str(budget)]["bucket_recursive_vs_immediate"]))
        print("AP_LM4A_CAPACITY", budget, json.dumps(capacity[str(budget)]))
    print("AP_LM4A_CONFIRM_RESERVE_TOUCHED", False)


if __name__ == "__main__":
    main()
