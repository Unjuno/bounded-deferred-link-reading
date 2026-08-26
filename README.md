# Bounded Deferred Reading and Querying

Computational and analytic work on **when to keep reading, when to defer an uncertainty, and when to resolve it** under finite time, attention, and memory.

> **Current phase:** the project has moved from broad experimental policy search to analytic modeling. The strongest working principle is: **resolve an uncertainty only when the expected value of resolving it now exceeds the expected value of deferring it.**

> **Claim boundary:** the repository contains computational, grounded-language, and analytic results. It does **not** yet establish an optimal human reading strategy or a validated human-comprehension interface.

## Core model

A reader or agent maintains

- the current context,
- a bounded frontier of unresolved items,
- a remaining time/action budget.

The key actions are:

`READ` — continue acquiring context,

`ASK(q)` — resolve one unresolved item.

The central decision is an optimal-stopping comparison:

`ASK(q) iff value(resolve q now) > value(defer q and keep reading)`.

Deferral has positive option value because later context can resolve an unknown for free, reveal that it is unimportant, or make a later query better targeted.

## What the experiments established

### Controlled hypertext

A short-lived deferred frontier plus **trajectory/downstream utility** was consistently stronger than purely local value in the controlled spectral bridge. AP-S43 reproduced a compact top-4 one-shot policy:

- S@16: **+3.9167 pp**, 95% CI **[+3.133,+4.683]**;
- S@32: **+1.1500 pp**, 95% CI **[+0.250,+2.067]**.

The number four is environment-specific, not a human working-memory claim.

### Real Wikispeedia

Real navigation showed strong short-range branch correction, but deployable visible trigger policies often failed despite large oracle opportunity. This exposed an important distinction:

> **The optimal action may exist while the visible state is insufficient to identify it.**

The project therefore separates the full-information control problem from the statistical observability problem.

### Recursive LM-style querying

AP-LM1 showed that learned **recursive subtree value** beats visible greedy selection:

- B=16: **+4.765 pp**, CI **[+4.449,+5.078]**;
- B=32: **+7.034 pp**, CI **[+6.531,+7.546]**.

AP-LM2 added asynchronous slack and found additional benefit from latency/slack awareness and from recursive rather than immediate-value teachers.

AP-LM3B transferred the mechanism to grounded natural-language recursive answers on **1,197 held-out missions / 753 target clusters**:

- recursive vs matched immediate, B=12: **+3.112 pp**, CI **[+2.292,+3.951]**, 8/8 buckets positive;
- recursive vs visible greedy: **+3.109 pp**, CI **[+2.238,+4.010]**, 8/8 positive;
- K=8 was within **0.551 pp** of the full frontier;
- at B=20 the full policies were essentially saturated near oracle.

This supports the general prediction that scheduling matters most when the information budget is binding.

## Analytic formulation

For a known recursive query tree, each query `q` has cost `c_q`, reward `r_q`, and children revealed after querying. The exact full-information problem is a precedence-constrained tree knapsack and can be solved by dynamic programming.

The correct object is a **budget-conditioned recursive value curve** `V(q,b)`, not one static importance scalar.

See [`docs/ANALYTIC_THEORY.md`](docs/ANALYTIC_THEORY.md).

## Interface implication

The theory suggests that users should not be required to estimate importance manually.

A better reading/chat interface can:

1. predict high-downstream-value terms or passages;
2. emphasize them visually, e.g. restrained bold/underline/markers;
3. allow one-click context-conditioned questions;
4. keep unresolved items in a deferred frontier rather than opening every explanation immediately;
5. reprioritize or retire items as later context arrives;
6. choose not only **what** and **when** to explain, but **how deeply**.

This points beyond ordinary linear chat toward:

`main reading stream + AI-ranked unresolved frontier + chat/explanation channel`.

See [`docs/INTERFACE_IMPLICATIONS.md`](docs/INTERFACE_IMPLICATIONS.md).

## Current research interpretation

The project is no longer mainly asking:

> Which heuristic should we try next?

It is now asking:

> **How closely can a bounded, partially observed reader/agent approximate the analytic full-information oracle?**

The remaining variables are principally:

- observability of downstream importance,
- contextual self-resolution,
- bounded frontier capacity,
- asynchronous latency,
- stochastic / hallucinating answers,
- interface costs and human cognitive load.

## Minimal remaining work

The project is close to a reasonable stopping point for the current computational phase. The highest-value remaining work is deliberately small:

1. analytic sanity checks / propositions for defer-vs-resolve thresholds and budget saturation;
2. one minimal generative-LM transfer check if execution infrastructure permits;
3. one small interface study comparing ordinary chat with AI highlighting + deferred frontier;
4. public synthesis rather than another broad parameter sweep.

## Repository map

- `docs/ANALYTIC_THEORY.md` — Bellman/tree-knapsack formulation and current theoretical interpretation.
- `docs/INTERFACE_IMPLICATIONS.md` — implications for highlighting, deferred questions, frontier chat, and future reading interfaces.
- `docs/RESEARCH_SUMMARY.md` — historical evolution of the experimental program.
- `docs/HUMAN_READING_HYPOTHESIS.md` — cautious human-reading translation and proposed study.
- `docs/EXPERIMENT_TIMELINE.md` — timeline of decisive experiments.
- `results/` — selected machine-readable confirmatory results.

## Reproducibility and boundary

External Wikipedia/Wikispeedia resources are not redistributed unless licensing permits it. Navigation performance, semantic similarity, and synthetic query reward should not be presented as direct measurements of human comprehension.

## License

Repository code and original text are MIT-licensed unless otherwise noted. Third-party data and derived resources remain subject to their original licenses.

---

## 日本語要約

この研究で見えてきた中心原理は、**「分からないものを見つけた瞬間に全部解決する」のではなく、まず読み続け、文脈で自然解決する可能性を残し、それでも下流の理解に重要になったものだけを必要な時点で問い合わせる**というものです。

さらに重要度判定まで読者だけに任せる必要はありません。AIが下流依存の大きそうな語句を太字・下線などで控えめに強調し、質問候補を保留frontierに保存し、後続文脈に応じて重要度を更新するインターフェースが自然に導かれます。

したがって将来像は、単なる一本道のチャットではなく、**本文 + AI重要度表示 + 保留質問frontier + チャット**です。現在の研究フェーズは、この方策をさらに実験で探索する段階から、解析的oracleを定義し、そのoracleを限られた観測・記憶・計算でどこまで近似できるかを調べる段階へ移っています。
