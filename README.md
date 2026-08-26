# Bounded Deferred Reading and Querying

Computational and analytic work on **when to keep reading, when to defer an uncertainty, and when to resolve it** under finite time, attention, and memory.

> **Current phase:** broad experimental policy search is essentially complete. The project now treats deferred reading/querying as an optimal-control problem and uses experiments mainly as validation of theory-derived predictions.

> **Claim boundary:** the repository contains computational, grounded-language, and analytic results. It does **not** establish an optimal human reading strategy or a validated human-comprehension interface.

## Core principle

A reader or agent maintains:

- the current context;
- a bounded frontier of unresolved items;
- a remaining time/action budget.

The main actions are `READ` and `ASK(q)`.

The working principle is:

> **Resolve an uncertainty only when the expected value of resolving it now exceeds the expected value of deferring it.**

Deferral has positive option value because later context can resolve an unknown for free, reveal that it is unimportant, or make a later query better targeted.

## What the experiments established

### Controlled hypertext

A short-lived deferred frontier plus **trajectory/downstream utility** was stronger than purely local value in the controlled spectral bridge. AP-S43 reproduced a compact one-shot policy:

- S@16: **+3.9167 pp**, 95% CI **[+3.133,+4.683]**;
- S@32: **+1.1500 pp**, 95% CI **[+0.250,+2.067]**.

The tested capacity values are environment-specific; they are not claims about human working-memory constants.

### Real Wikispeedia

Real navigation showed strong short-range branch correction, while deployable visible trigger policies often failed despite large oracle opportunity. This exposed an important distinction:

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

The key theoretical object is a **budget-conditioned recursive value curve** `V(q,b)`, not one static importance scalar.

The main research question is therefore now:

> **How closely can a bounded, partially observed reader or agent approximate the analytic full-information oracle?**

See [`docs/ANALYTIC_THEORY.md`](docs/ANALYTIC_THEORY.md).

## Interface proposal

The theory also yields a concrete, but not yet human-validated, interface proposal.

Users should not have to estimate the importance of every unclear term themselves. A reading assistant can:

1. estimate which terms or passages have high downstream dependency;
2. use restrained visual emphasis such as bold, underline, or margin markers;
3. make context-conditioned explanation available with one action;
4. **not** automatically interrupt reading with every explanation;
5. retain unresolved items in a deferred frontier;
6. reprioritize or retire them as later context arrives;
7. use chat as the explanation channel while returning new follow-up unknowns to the frontier;
8. adapt explanation depth to the importance of the concept.

A plausible layout is:

`main document + AI-guided salience + unresolved frontier + chat/explanation channel`

The detailed proposal, including a minimal prototype and testable UI hypotheses, is in [`docs/INTERFACE_PROPOSAL.md`](docs/INTERFACE_PROPOSAL.md).

## Theory-following sanity check

AP-T1 uses exact dynamic programming on small random recursive trees as a sanity check of consequences already implied by the analytic model.

Across 200 trees it verified:

- value is monotone in budget;
- value is monotone in frontier capacity;
- the full oracle is never worse than immediate greedy;
- policy differences disappear once budget is sufficient to perform all useful work;
- the single-unknown defer/resolve threshold produced zero contradictions in 144 enumerated checks.

This is a mathematical/computational consistency check, not evidence about human comprehension.

## Current stopping point

The current computational phase has a reasonable stopping point here. Additional broad parameter sweeps are unlikely to add much.

The main optional external-validity test, if pursued later, is a small human interface study comparing ordinary document+chat with AI-guided salience and a deferred unresolved frontier.

That study is **future validation**, not required to make the current computational and analytic results available.

## Repository map

- `docs/ANALYTIC_THEORY.md` — Bellman/tree-knapsack formulation and current theoretical interpretation.
- `docs/INTERFACE_PROPOSAL.md` — concrete reading/chat UI proposal derived from the theory.
- `docs/RESEARCH_SUMMARY.md` — historical evolution of the experimental program and claim boundaries.
- `docs/HUMAN_READING_HYPOTHESIS.md` — cautious translation to human reading and possible validation study.
- `docs/EXPERIMENT_TIMELINE.md` — timeline of decisive experiments.
- `experiments/` — experiment and sanity-check code.
- `results/` — selected machine-readable results.

## Reproducibility and boundary

External Wikipedia/Wikispeedia resources are not redistributed unless licensing permits it. Navigation performance, semantic similarity, and synthetic query reward should not be presented as direct measurements of human comprehension.

## License

Repository code and original text are MIT-licensed unless otherwise noted. Third-party data and derived resources remain subject to their original licenses.

---

## 日本語要約

この研究で見えてきた中心原理は、**分からないものを見つけた瞬間に全部解決するのではなく、まず読み続け、文脈で自然解決する可能性を残し、それでも下流の理解に重要になったものだけを必要な時点で問い合わせる**というものです。

重要度判定まで読者だけに任せる必要もありません。AIが後続理解への依存度が高そうな語句を控えめに強調し、質問候補を保留frontierに保存し、後続文脈に応じて重要度を更新するUIを提案できます。ただし、このUIの人間に対する効果はまだ直接検証していません。

研究の現在地は、heuristicをさらに探索する段階ではなく、解析的oracleを定義し、そのoracleを限られた観測・記憶・計算でどこまで近似できるかを考える段階です。
