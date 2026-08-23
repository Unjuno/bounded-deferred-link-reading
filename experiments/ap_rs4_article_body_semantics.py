#!/usr/bin/env python3
"""AP-RS4: real Wikispeedia article-body semantic progress and BACK diagnostics.

This phase upgrades AP-RS3 from title semantics to actual Wikispeedia article text.
It is still observational on human paths: it tests whether successful human navigation
moves toward the target in article-body semantic space, and whether the first eligible
BACK replacement improves body-semantic closeness.

Claim boundary: this is article-body semantics, not anchor/context decision scoring and
not a causal human-comprehension test.
"""
from __future__ import annotations

import json
import math
import os
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

SNAP_TEXT_URL = "https://snap.stanford.edu/data/wikispeedia/wikispeedia_articles_plaintext.tar.gz"
ADAS_REF = "69052d52bbbfe57ed25e9bccbd36a5acbc0f988d"
PATHS_URL = f"https://raw.githubusercontent.com/epfl-ada/ada-2024-project-adaspeedia/{ADAS_REF}/data/paths_finished_unique.tsv"
SEED = 20260824
N_BOOT = 2000
MIN_PROGRESS = 400
MIN_BACK = 300
MAX_CHARS = 6000


def dec(x: object) -> str:
    return urllib.parse.unquote(str(x))


def norm_title(s: str) -> str:
    s = dec(s).strip().replace(" ", "_")
    if s.endswith(".txt"):
        s = s[:-4]
    return s


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"download {url}")
    urllib.request.urlretrieve(url, path)


def reconstruct(tokens: list[str]) -> list[str]:
    stack: list[str] = []
    states: list[str] = []
    for raw in tokens:
        t = dec(raw)
        if t == "<":
            if len(stack) > 1:
                stack.pop()
                states.append(stack[-1])
        else:
            stack.append(t)
            states.append(t)
    return states


def back_episodes(tokens: list[str]):
    stack: list[str] = []
    active = None
    out = []
    for raw in tokens:
        t = dec(raw)
        if t == "<":
            if not stack:
                continue
            if active is None:
                active = {"abandoned": stack[-1], "n_back": 0}
            if len(stack) > 1:
                stack.pop()
                active["n_back"] += 1
            continue
        if active is not None:
            out.append((active["abandoned"], stack[-1] if stack else None, t, active["n_back"]))
            active = None
        stack.append(t)
    return out


def cluster_boot(df: pd.DataFrame, col: str, cluster: str, offset: int = 0):
    g = df.groupby(cluster, dropna=False)[col].agg(["sum", "count"])
    sums = g["sum"].to_numpy(float)
    counts = g["count"].to_numpy(float)
    rng = np.random.default_rng(SEED + offset)
    reps = np.empty(N_BOOT)
    for b in range(N_BOOT):
        ii = rng.integers(0, len(g), len(g))
        reps[b] = sums[ii].sum() / counts[ii].sum()
    return [float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))]


def stat(df: pd.DataFrame, col: str):
    x = df[col].to_numpy(float)
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "sd": float(x.std(ddof=1)),
        "u_c": float(x.std(ddof=1) / math.sqrt(len(x))),
        "positive_fraction": float(np.mean(x > 0)),
        "zero_fraction": float(np.mean(x == 0)),
        "user_clusters": int(df.user.nunique()),
        "target_clusters": int(df.target.nunique()),
        "user_cluster_ci95": cluster_boot(df, col, "user", 0),
        "target_cluster_ci95": cluster_boot(df, col, "target", 31),
    }


def load_article_texts(extracted: Path, wanted: set[str]):
    wanted_norm = {norm_title(x): x for x in wanted}
    texts: dict[str, str] = {}
    files = [p for p in extracted.rglob("*") if p.is_file()]
    print("extracted files", len(files))
    for p in files:
        # Official archive uses one file per article; basename matching covers the corpus.
        candidates = [norm_title(p.name), norm_title(p.stem)]
        hit = None
        for c in candidates:
            if c in wanted_norm:
                hit = wanted_norm[c]
                break
        if hit is None:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            continue
        if txt:
            texts[hit] = txt[:MAX_CHARS]
    return texts


def main():
    out = Path(os.environ.get("AP_RS4_OUT", "artifacts/ap_rs4"))
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    raw.mkdir(exist_ok=True)

    paths_file = raw / "paths_finished_unique.tsv"
    text_tar = raw / "wikispeedia_articles_plaintext.tar.gz"
    download(PATHS_URL, paths_file)
    download(SNAP_TEXT_URL, text_tar)

    extract_dir = raw / "plaintext"
    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        extract_dir.mkdir(exist_ok=True)
        with tarfile.open(text_tar, "r:gz") as tf:
            tf.extractall(extract_dir)

    paths = pd.read_csv(paths_file, sep="\t")
    parsed = []
    titles: set[str] = set()
    for _, r in paths.iterrows():
        toks = str(r.path).split(";")
        arts = [dec(t) for t in toks if dec(t) != "<"]
        if len(arts) < 2:
            continue
        parsed.append((r, toks, arts))
        titles.update(arts)

    texts = load_article_texts(extract_dir, titles)
    coverage = len(texts) / max(1, len(titles))
    print("unique path titles", len(titles), "body texts", len(texts), "coverage", coverage)
    if coverage < 0.90:
        sample_missing = sorted(titles - set(texts))[:30]
        raise RuntimeError(f"article-body coverage too low: {coverage:.3f}; missing sample={sample_missing}")

    usable_titles = sorted(texts)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # The encoder truncates long documents to its configured sequence length. We keep a
    # reproducible body prefix and explicitly report this limitation.
    bodies = [texts[t] for t in usable_titles]
    body_emb = model.encode(bodies, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    title_emb = model.encode([t.replace("_", " ") for t in usable_titles], batch_size=128,
                             show_progress_bar=True, normalize_embeddings=True)
    idx = {t: i for i, t in enumerate(usable_titles)}

    def sim_body(a: str, b: str) -> float:
        return float(np.dot(body_emb[idx[a]], body_emb[idx[b]]))

    def sim_title(a: str, b: str) -> float:
        return float(np.dot(title_emb[idx[a]], title_emb[idx[b]]))

    def sim_equal(a: str, b: str) -> float:
        return 0.5 * sim_body(a, b) + 0.5 * sim_title(a, b)

    channels = {"body": sim_body, "title_internal": sim_title, "equal_title_body": sim_equal}
    progress_rows = {k: [] for k in channels}
    back_rows = {k: [] for k in channels}

    usable_paths = 0
    for r, toks, arts in parsed:
        target = arts[-1]
        source = arts[0]
        if target not in idx:
            continue
        states = reconstruct(toks)
        if any(s not in idx for s in states):
            continue
        usable_paths += 1
        user = str(r.get("hashedIpAddress", "NA"))
        pid = r.get("path_id", 0)
        had_back = "<" in [dec(t) for t in toks]

        for name, sim in channels.items():
            if len(states) >= 3:
                # Exclude terminal transition to avoid the trivial sim(target,target)=1 jump.
                ds = [sim(b, target) - sim(a, target) for a, b in zip(states[:-2], states[1:-1])]
                if ds:
                    progress_rows[name].append({
                        "path_id": pid, "user": user, "source": source, "target": target,
                        "mean_delta": float(np.mean(ds)),
                        "positive_step_fraction": float(np.mean(np.asarray(ds) > 0)),
                        "n_transitions": len(ds), "had_backtrack": had_back,
                    })

            for ei, (abandoned, returned, replacement, nback) in enumerate(back_episodes(toks)):
                if returned is None or replacement == target:
                    continue
                if abandoned not in idx or returned not in idx or replacement not in idx:
                    continue
                back_rows[name].append({
                    "path_id": pid, "user": user, "source": source, "target": target,
                    "episode_index": ei,
                    "return_delta": sim(returned, target) - sim(abandoned, target),
                    "replacement_delta": sim(replacement, target) - sim(abandoned, target),
                    "replacement_vs_return_delta": sim(replacement, target) - sim(returned, target),
                    "n_back": nback,
                })
                break

    results = {}
    for name in channels:
        prog = pd.DataFrame(progress_rows[name])
        backs = pd.DataFrame(back_rows[name])
        prog.to_csv(out / f"progress_{name}.csv", index=False)
        backs.to_csv(out / f"back_{name}.csv", index=False)
        ps = stat(prog, "mean_delta")
        pc = {
            "n_ge_400": ps["n"] >= MIN_PROGRESS,
            "mean_gt_0": ps["mean"] > 0,
            "user_ci_lower_gt_0": ps["user_cluster_ci95"][0] > 0,
            "target_ci_lower_gt_0": ps["target_cluster_ci95"][0] > 0,
            "positive_paths_gt_half": ps["positive_fraction"] > 0.5,
        }
        if len(backs):
            bs = stat(backs, "replacement_delta")
            rs = stat(backs, "return_delta")
            br = stat(backs, "replacement_vs_return_delta")
            bc = {
                "n_ge_300": bs["n"] >= MIN_BACK,
                "mean_gt_0": bs["mean"] > 0,
                "user_ci_lower_gt_0": bs["user_cluster_ci95"][0] > 0,
                "target_ci_lower_gt_0": bs["target_cluster_ci95"][0] > 0,
                "positive_paths_gt_half": bs["positive_fraction"] > 0.5,
            }
        else:
            bs = rs = br = {}
            bc = {"n_ge_300": False, "mean_gt_0": False, "user_ci_lower_gt_0": False,
                  "target_ci_lower_gt_0": False, "positive_paths_gt_half": False}
        results[name] = {
            "progress": {"decision": "PASS" if all(pc.values()) else "FAIL", "conditions": pc, "stats": ps},
            "back_replacement": {
                "decision": "PASS" if all(bc.values()) else "FAIL", "conditions": bc,
                "replacement_vs_abandoned": bs, "return_vs_abandoned": rs,
                "replacement_vs_returned_ancestor": br,
            },
        }

    primary = results["body"]
    result = {
        "phase": "AP-RS4",
        "name": "real Wikispeedia article-body semantic progress and BACK diagnostic",
        "primary_channel": "MiniLM article-body-prefix cosine",
        "primary_progress_decision": primary["progress"]["decision"],
        "primary_back_decision": primary["back_replacement"]["decision"],
        "data": {
            "wikispeedia_plaintext_url": SNAP_TEXT_URL,
            "human_paths_url": PATHS_URL,
            "unique_path_titles": len(titles),
            "article_body_coverage": coverage,
            "usable_paths": usable_paths,
            "max_chars_per_article": MAX_CHARS,
        },
        "encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "construct": "real human Wikispeedia navigation + article-body semantic outcome; observational, not anchor/context action scoring",
        "channels": results,
        "boundary": [
            "Article-body semantics are a stronger external-semantic outcome than title-only RS3.",
            "The body of an unvisited candidate page is not assumed visible to an unaided human reader.",
            "This phase does not establish causal benefit of bounded deferred links or human comprehension gains.",
        ],
    }
    (out / "AP_RS4_RESULTS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    bp = primary["progress"]["stats"]
    bb = primary["back_replacement"]["replacement_vs_abandoned"]
    md = f"""# AP-RS4 — Real article-body semantic diagnostic

## Body-semantic path progress: {primary['progress']['decision']}
- usable paths: {bp.get('n')}
- mean nonterminal semantic delta: {bp.get('mean')}
- positive path fraction: {bp.get('positive_fraction')}
- user-cluster CI: {bp.get('user_cluster_ci95')}
- target-cluster CI: {bp.get('target_cluster_ci95')}

## First nonterminal BACK replacement: {primary['back_replacement']['decision']}
- n: {bb.get('n')}
- replacement vs abandoned mean: {bb.get('mean')}
- positive fraction: {bb.get('positive_fraction')}
- user-cluster CI: {bb.get('user_cluster_ci95')}
- target-cluster CI: {bb.get('target_cluster_ci95')}

## Boundary
This uses actual Wikispeedia article text as a semantic outcome space. It is still observational and does not yet use visible anchor + containing-context features to choose actions.
"""
    (out / "AP_RS4_SUMMARY.md").write_text(md, encoding="utf-8")
    print("AP_RS4_BODY_PROGRESS", primary["progress"]["decision"], bp)
    print("AP_RS4_BODY_BACK", primary["back_replacement"]["decision"], bb)


if __name__ == "__main__":
    main()
