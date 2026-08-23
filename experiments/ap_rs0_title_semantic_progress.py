#!/usr/bin/env python3
"""AP-RS0: real human Wikispeedia title-semantic progress gate.

This is a mechanism gate, not a human-comprehension test and not the final
anchor+containing-context AO-P2 validation.

Primary question:
Do successful human Wikispeedia trajectories, excluding the mechanically
trivial terminal click into the target, tend to move toward titles that are
semantically more similar to the target title?

Primary inference unit: one unique source-target human path.
Primary semantic channel: SBERT title cosine from the public ADASpeedia
processed dataset. BERT title cosine and backtrack-straightened paths are
secondary checks.
"""
from __future__ import annotations

import ast
import json
import math
import os
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

MIRROR_REF = "69052d52bbbfe57ed25e9bccbd36a5acbc0f988d"
BASE = f"https://raw.githubusercontent.com/epfl-ada/ada-2024-project-adaspeedia/{MIRROR_REF}/data"
PATHS_URL = f"{BASE}/paths_finished_unique.tsv"
SIM_URL = f"{BASE}/article_similarities.csv"
N_BOOT = 2000
BOOT_SEED = 20260823
MIN_PATHS = 400


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, path)


def decode_title(x: str) -> str:
    return urllib.parse.unquote(str(x))


def reconstruct_states(tokens: list[str]) -> list[str]:
    """Reconstruct actual page states, interpreting '<' as browser BACK."""
    stack: list[str] = []
    states: list[str] = []
    for raw in tokens:
        tok = decode_title(raw)
        if tok == "<":
            if len(stack) > 1:
                stack.pop()
                states.append(stack[-1])
            continue
        stack.append(tok)
        states.append(tok)
    return states


def straighten(tokens: list[str]) -> list[str]:
    """Resolve BACK tokens to the final forward-choice stack."""
    stack: list[str] = []
    for raw in tokens:
        tok = decode_title(raw)
        if tok == "<":
            if stack:
                stack.pop()
        else:
            stack.append(tok)
    return stack


def parse_pair(x: str):
    try:
        p = ast.literal_eval(x)
        if isinstance(p, (tuple, list)) and len(p) == 2:
            return decode_title(p[0]), decode_title(p[1])
    except Exception:
        pass
    return None


def cluster_boot_ci(df: pd.DataFrame, value_col: str, cluster_col: str, n_boot=N_BOOT):
    g = df.groupby(cluster_col, dropna=False)[value_col].apply(np.asarray)
    keys = list(g.index)
    vals = [g[k] for k in keys]
    rng = np.random.default_rng(BOOT_SEED + (17 if cluster_col == "target" else 0))
    reps = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, len(vals), size=len(vals))
        sample = np.concatenate([vals[i] for i in idx])
        reps[b] = float(np.mean(sample))
    return [float(np.quantile(reps, .025)), float(np.quantile(reps, .975))]


def basic_stats(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "sd": float(np.std(x, ddof=1)),
        "u_c": float(np.std(x, ddof=1) / math.sqrt(len(x))),
        "positive_fraction": float(np.mean(x > 0)),
        "zero_fraction": float(np.mean(x == 0)),
    }


def path_metric(states: list[str], target: str, sim: dict[tuple[str, str], float]):
    # Exclude the terminal transition into target to avoid a mechanically
    # guaranteed jump to self-similarity ~= 1.
    if len(states) < 3:
        return None
    deltas = []
    missing = 0
    for a, b in zip(states[:-2], states[1:-1]):
        sa = sim.get((target, a))
        sb = sim.get((target, b))
        if sa is None or sb is None or not np.isfinite(sa) or not np.isfinite(sb):
            missing += 1
            continue
        deltas.append(sb - sa)
    if not deltas:
        return None
    return {
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "positive_step_fraction": float(np.mean(np.asarray(deltas) > 0)),
        "n_transitions": len(deltas),
        "missing_transitions": missing,
    }


def main():
    out = Path(os.environ.get("AP_RS0_OUT", "artifacts/ap_rs0"))
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    paths_file = raw / "paths_finished_unique.tsv"
    sim_file = raw / "article_similarities.csv"
    download(PATHS_URL, paths_file)
    download(SIM_URL, sim_file)

    paths = pd.read_csv(paths_file, sep="\t")
    sims = pd.read_csv(sim_file)
    print("paths columns", list(paths.columns), "n", len(paths))
    print("sim columns", list(sims.columns), "n", len(sims))

    sims["pair_parsed"] = sims["pair"].map(parse_pair)
    sims = sims[sims["pair_parsed"].notna()].copy()
    sbert = dict(zip(sims["pair_parsed"], pd.to_numeric(sims["sbert_cosine_similarity"], errors="coerce")))
    bert = dict(zip(sims["pair_parsed"], pd.to_numeric(sims["cosine_similarity"], errors="coerce")))

    rows = []
    for _, r in paths.iterrows():
        toks = str(r["path"]).split(";")
        st_raw = reconstruct_states(toks)
        st_straight = straighten(toks)
        if not st_raw:
            continue
        target = decode_title(st_raw[-1])
        source = decode_title(st_raw[0])
        # The unique-path preprocessing makes source-target pairs unique.
        for channel, sim in [("sbert", sbert), ("bert", bert)]:
            for representation, states in [("raw", st_raw), ("straightened", st_straight)]:
                m = path_metric(states, target, sim)
                if m is None:
                    continue
                rows.append({
                    "path_id": r.get("path_id", len(rows)),
                    "user": str(r.get("hashedIpAddress", "NA")),
                    "source": source,
                    "target": target,
                    "channel": channel,
                    "representation": representation,
                    "path_len_states": len(states),
                    "had_backtrack": "<" in [decode_title(t) for t in toks],
                    **m,
                })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "path_metrics.csv", index=False)
    primary = metrics[(metrics.channel == "sbert") & (metrics.representation == "raw")].copy()
    sec_bert = metrics[(metrics.channel == "bert") & (metrics.representation == "raw")].copy()
    sec_straight = metrics[(metrics.channel == "sbert") & (metrics.representation == "straightened")].copy()

    def summarize(d: pd.DataFrame):
        stat = basic_stats(d.mean_delta.to_numpy())
        stat["user_clusters"] = int(d.user.nunique())
        stat["target_clusters"] = int(d.target.nunique())
        stat["source_clusters"] = int(d.source.nunique())
        stat["user_cluster_ci95"] = cluster_boot_ci(d, "mean_delta", "user")
        stat["target_cluster_ci95"] = cluster_boot_ci(d, "mean_delta", "target")
        stat["mean_positive_step_fraction"] = float(d.positive_step_fraction.mean())
        stat["backtrack_path_fraction"] = float(d.had_backtrack.mean())
        return stat

    primary_stats = summarize(primary)
    bert_stats = summarize(sec_bert)
    straight_stats = summarize(sec_straight)

    conditions = {
        "n_ge_400": primary_stats["n"] >= MIN_PATHS,
        "mean_gt_0": primary_stats["mean"] > 0,
        "user_ci_lower_gt_0": primary_stats["user_cluster_ci95"][0] > 0,
        "target_ci_lower_gt_0": primary_stats["target_cluster_ci95"][0] > 0,
        "positive_paths_gt_half": primary_stats["positive_fraction"] > 0.5,
    }
    decision = "PASS" if all(conditions.values()) else "FAIL"

    # Sensitivity by path length quartile and backtracking status.
    sens = {}
    if len(primary):
        try:
            primary["length_q"] = pd.qcut(primary.path_len_states, 4, duplicates="drop")
            sens["by_length_quartile"] = {
                str(k): basic_stats(g.mean_delta.to_numpy()) for k, g in primary.groupby("length_q", observed=True)
            }
        except Exception as e:
            sens["length_error"] = str(e)
        sens["by_backtrack"] = {
            str(k): basic_stats(g.mean_delta.to_numpy()) for k, g in primary.groupby("had_backtrack")
        }

    result = {
        "phase": "AP-RS0",
        "name": "real human Wikispeedia title-semantic progress gate",
        "decision": decision,
        "preregistered_conditions": conditions,
        "construct": "real human navigation + article-title semantics; NOT article-body/anchor/context semantics",
        "data": {
            "source_repo": "epfl-ada/ada-2024-project-adaspeedia",
            "source_ref": MIRROR_REF,
            "paths_url": PATHS_URL,
            "similarities_url": SIM_URL,
            "unique_source_target_preprocessing": True,
            "terminal_transition_excluded": True,
        },
        "primary": {"channel": "SBERT title cosine", "path_representation": "raw BACK-reconstructed", **primary_stats},
        "secondary_bert_raw": bert_stats,
        "secondary_sbert_straightened": straight_stats,
        "sensitivity": sens,
        "interpretation_boundary": [
            "A PASS would establish only a title-semantic progress signal in successful human Wikispeedia navigation.",
            "It would not validate real article-body semantics, anchor/context scoring, bounded deferred-link causality, or human comprehension gains."
        ],
    }
    with open(out / "AP_RS0_RESULTS.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    md = [
        "# AP-RS0 — Real human Wikispeedia title-semantic progress gate",
        "",
        f"**Decision: {decision}**",
        "",
        "Primary unit: unique source-target human path. Terminal transition into target excluded.",
        "",
        "## Primary (SBERT title cosine; BACK reconstructed)",
        f"- n paths: {primary_stats['n']}",
        f"- mean per-path nonterminal delta: {primary_stats['mean']:.6f}",
        f"- median: {primary_stats['median']:.6f}",
        f"- positive-path fraction: {primary_stats['positive_fraction']:.4f}",
        f"- user-cluster 95% CI: {primary_stats['user_cluster_ci95']}",
        f"- target-cluster 95% CI: {primary_stats['target_cluster_ci95']}",
        f"- mean positive-step fraction: {primary_stats['mean_positive_step_fraction']:.4f}",
        "",
        "## Boundary",
        "This is real human navigation with real article titles, but title embeddings are not article-body, anchor, or containing-context semantics. It is a mechanism bridge, not the final real-semantic reading validation.",
    ]
    (out / "AP_RS0_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
