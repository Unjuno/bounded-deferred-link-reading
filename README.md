# Bounded Deferred-Link Reading

Computational experiments on bounded-memory hypertext reading and navigation, testing **context-first reading**, **deferred link options**, and **one-shot trajectory-aware reconsideration**.

> **Claim boundary:** this repository reports computational and hypothesis-generating results. It does **not** yet establish an optimal human reading strategy. Real anchor/context multi-hop validation and human comprehension/retention experiments remain open.

## Core idea

When link value is uncertain, a reader or agent need not immediately traverse every promising hyperlink. A compact alternative is:

1. continue acquiring local context;
2. keep a small, short-lived buffer of promising abandoned alternatives;
3. reconsider those alternatives at the next decision point;
4. allow at most one discretionary return when predicted trajectory-level utility is positive;
5. avoid optimizing short-horizon gain at the expense of long-horizon success.

The current compact bridge policy uses the **immediately previous decision point**, keeps the **top 4 abandoned alternatives**, and scores them with a **3-feature Ridge counterfactual trajectory-utility model** using:

- candidate degree,
- origin candidate count,
- relative score versus the current best candidate.

## Main findings

### 1. Context matters

Across the earlier information-acquisition experiments, link/anchor semantics were more useful when combined with the containing sentence or paragraph context than when considered alone. This motivates treating a hyperlink as an option whose value depends on the surrounding context, rather than as an instruction to click immediately.

### 2. One remembered runner-up is not universally sufficient

Under a distance-like controlled cue, remembering the immediate-parent runner-up was useful. Under a spectral/non-distance bridge, however, the same one-scalar idea failed and could materially hurt navigation. This rejected the universal form of the "one remembered alternative is enough" hypothesis.

### 3. Rich history contains value, but all-history memory is unnecessary

Oracle experiments showed that richer frontiers can contain substantial unrealized value. Subsequent compression experiments found that almost all useful deployable signal in the tested bridge could be recovered from a bounded immediate-parent buffer.

### 4. Four candidates form a performance plateau in the current bridge

The confirmatory compact-policy replication (AP-S43) froze the model after selection and evaluated **12 new seeds × 500 tasks = 6,000 tasks**.

- **S@16:** +3.9167 percentage points versus local-only, 95% cluster+task CI **[+3.133, +4.683] pp**, positive in **12/12** seeds.
- **S@32:** +1.1500 pp, 95% CI **[+0.250, +2.067] pp**, positive in **9/12** seeds.

A later capacity sweep (AP-S44) found that moving from K=4 to K=7 or K=9 produced no material additional gain. This is a computational bridge result, **not** evidence that human working-memory capacity is exactly four items.

### 5. Reconsider quickly, not many pages later

Delayed and persistent multi-page retention experiments did not establish an advantage over reconsidering at the next decision point. The strongest tested version is therefore a **short-lived option buffer**, not a long-lived collection of deferred tabs.

### 6. Memory can be traded for a small amount of performance

AP-S49 tested a resource-saving adaptive buffer on **12 new seeds × 500 tasks = 6,000 tasks**. Reducing the cap on high-branching states cut mean stored alternatives from **2.987 to 2.364 per forward page** (about **20.9% less candidate memory**) while losing only **0.433 pp at S@16** and **0.400 pp at S@32** relative to fixed K=4.

This supports a **resource/performance trade-off**. It does not show that reducing links under human cognitive load improves comprehension.

## Current computational policy

```text
Read/score the current context
        |
        +--> follow the current best candidate
        |
        +--> retain up to 4 strong abandoned alternatives
                         |
                    next decision point
                         |
              score retained alternatives
              with trajectory utility
                         |
             utility > threshold (0.05)?
                    /             \
                  no               yes
                  |                 |
            continue local      one return
                                   only
```

The current policy is **O(1) in history depth**: it keeps only the previous decision point's bounded candidate set. The tested constant is four candidate identities, not one scalar.

## Human-readable hypothesis

A cautious translation for human hypertext reading is:

> **Do not treat every hyperlink as an immediate command to leave the page. Read enough surrounding context to understand why the link matters, keep only a few promising alternatives in mind or in an external note, and reconsider them soon—preferably once—after additional context has changed your estimate of their value.**

This is a hypothesis for human reading, not yet a validated prescription.

## What failed or remained uncertain

The repository intentionally preserves negative results because they materially changed the theory:

- large recurrent or nonlinear state did not provide a stable universal advantage;
- global/all-history frontier memory was not needed for the best compact deployable policy;
- immediate-parent runner-up memory was not cue-universal;
- simple confidence/reliability gates did not reliably fix high-noise behavior;
- repeatedly applying a one-shot trajectory gate degraded performance;
- optimizing S@16 alone could strongly damage S@32;
- persistent multi-page deferred-link retention was not supported;
- task-demand adaptive K=3/4/5 did not materially outperform fixed K=4.

## Research status

**Ready for:** public computational / hypothesis-generating release.

**Not yet ready for:** a claim of an "optimal human hyperlink reading strategy."

The two major external-validity gates are:

1. **400+ independent real anchor/context semantic tasks**, preferably with real multi-hop outcomes;
2. **human experiments** measuring comprehension, retention, reading time, navigation behavior, and cognitive load.

## Repository map

- `docs/RESEARCH_SUMMARY.md` — research program, major hypothesis revisions, and claim boundaries.
- `docs/HUMAN_READING_HYPOTHESIS.md` — human-readable theory and proposed human experiment.
- `docs/EXPERIMENT_TIMELINE.md` — compact timeline of decisive experiments.
- `results/` — selected machine-readable confirmatory result files.
- `NOTICE.md` — data/reproducibility and external-dataset notes.

## Reproducibility note

The experiments use Wikipedia-derived graph data and locally generated intermediate artifacts. External datasets are **not redistributed here unless their licenses permit it**. Public releases should prefer download/preprocessing instructions over copying third-party datasets into the repository.

## License

Code and repository text are released under the repository's MIT License unless a file states otherwise. Third-party datasets and derived resources remain subject to their original licenses.

---

## 日本語要約

現時点の計算実験では、未知のハイパーテキストに対して「リンクを見たら即座に飛ぶ」より、**まず周辺文脈を読み、直前の判断点で捨てた有望候補を4個程度だけ短時間保持し、次の判断点で一度だけ再評価する**方策が有望です。7〜9候補へ増やしても大きな追加利得はなく、何ページも候補を保持し続ける方策も支持されませんでした。

ただし、これは人間の読解実験ではありません。4という値を人間のworking-memory定数と解釈することもできません。人間向けには「少数の保留リンクを持ち、文脈が増えたところですぐ再評価する」という検証可能な仮説として扱います。
