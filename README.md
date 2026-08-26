# Bounded Deferred Reading and Querying

Computational and analytic work on **when to keep reading, when to defer an uncertainty, and when to resolve it** under finite time, attention, memory, and observability.

> **Current phase:** broad heuristic/policy search is complete enough to stop. The project now treats deferred reading/querying as an optimal-control problem and uses experiments mainly to validate theory-derived predictions.

> **Claim boundary:** these results do **not** establish an optimal human reading strategy, a universal working-memory capacity, or a validated human-comprehension interface.

## Core principle

An unresolved item is an option, not an immediate command to stop and explain it.

The working decision rule is:

`resolve q now` only when its expected value exceeds the expected value of deferring it and continuing to read.

Deferral can be valuable because more context may:

- resolve the item for free;
- reveal that it is irrelevant;
- show that it is a prerequisite;
- improve the scope of a later query.

For a fully known recursive query tree with integer costs, the full-information problem is a precedence-constrained tree knapsack and can be solved by dynamic programming. The important value object is therefore a **budget-conditioned downstream value** `V(q,b)`, not one static importance score.

See [`docs/ANALYTIC_THEORY.md`](docs/ANALYTIC_THEORY.md).

## What the computational program found

### Controlled hypertext

A short-lived deferred frontier plus trajectory/downstream utility outperformed purely local value in the controlled spectral bridge. The frozen AP-S43 compact policy replicated on 6,000 fresh tasks:

- S@16: **+3.9167 pp**, 95% CI **[+3.133,+4.683]**;
- S@32: **+1.1500 pp**, 95% CI **[+0.250,+2.067]**.

The tested K values are environment-specific; K≈4 is **not** a human-memory claim.

### Real Wikispeedia

Real human paths showed strong target-directed semantic progress and predominantly short-range BACK behavior. However, deployable visible trigger policies often failed despite substantial oracle opportunity.

This exposed a central distinction:

> **A valuable full-information action can exist while the visible state is insufficient to identify it reliably.**

The real-data phase therefore shifted attention from “is there latent value?” to the **observability bottleneck**.

### Recursive LM-style querying

When answers can reveal new unresolved questions, immediate answer value is not enough: a query opens a future information subtree.

AP-LM1:

- recursive vs visible greedy, B=16: **+4.765 pp**, CI **[+4.449,+5.078]**;
- B=32: **+7.034 pp**, CI **[+6.531,+7.546]**.

AP-LM2 added asynchronous/slack-aware scheduling and found additional benefit from valuing effective blocking/opportunity cost rather than raw latency alone.

AP-LM3B then ran a fresh held-out confirmation on grounded natural-language recursive answer trees after removing a privileged-information latency path found in AP-LM3A.

Held-out sample: **1,197 missions / 753 target clusters**.

At B=12:

- recursive vs matched immediate: **+3.112 pp**, CI **[+2.292,+3.951]**, 8/8 buckets positive;
- recursive vs visible greedy: **+3.109 pp**, CI **[+2.238,+4.010]**, 8/8 positive.

At B=20, the main policies were essentially saturated near oracle, supporting the prediction that scheduling matters most when the information budget is binding.

## Analytic sanity check

AP-T1 follows the theory rather than searching for a new heuristic. Across 200 small random recursive trees it verified:

- value is monotone in budget;
- value is monotone in frontier capacity;
- the full oracle is never worse than immediate greedy;
- policy differences disappear once the budget permits all useful work;
- 144 enumerated single-unknown threshold checks produced zero contradictions.

This is a mathematical/computational consistency check, not evidence about human cognition.

## Interface proposal

The theory suggests a concrete but unvalidated interface direction.

The **smallest useful version** is not a large frontier dashboard. It is:

1. keep the original document primary;
2. sparsely emphasize concepts predicted to have high **downstream dependency**, not merely high difficulty;
3. let the reader choose a small explanation action such as `No explanation / 1 line / Example / Detailed / Later`;
4. insert routine short explanations directly into the reading flow rather than forcing a switch to a separate chat;
5. learn explanation needs from those choices;
6. keep deferred unresolved items mostly as hidden system state.

Over time this can progress from manual depth choice to personalized defaults and, eventually, conservative automatic inline supplements or a continuous adaptive reading stream. “Continuous” means the system need not stop for explicit query turns; the reader must still be able to pause and take control.

See [`docs/INTERFACE_PROPOSAL.md`](docs/INTERFACE_PROPOSAL.md).

## Current scientific position

### Supported computationally

- immediate resolution of every uncertainty is not generally required for good bounded information acquisition;
- deferral can have positive option value;
- local relevance and downstream/trajectory value are distinct;
- recursive querying increases the importance of downstream subtree value;
- scheduling matters most when budget is binding;
- useful full-information actions can be hard to identify from visible state;
- small bounded frontiers can often approximate richer history, with environment-dependent capacity;
- asynchronous scheduling should account for effective blocking/slack.

### Not established

- an optimal human reading strategy;
- a universal human memory capacity such as K=4 or K=8;
- a causal cognitive explanation of observed human BACK behavior;
- improved human comprehension or retention from the proposed UI;
- robust control under hallucinating/stochastic real generative-LM answers.

## Stopping point

The broad computational exploration phase is complete enough to stop without another large parameter sweep.

Remaining high-value directions are optional follow-ups:

- formal approximation/threshold results for the analytic control problem;
- stochastic or hallucinating-answer extensions;
- a small human interface study if external validation is desired.

## Repository map

- [`docs/ANALYTIC_THEORY.md`](docs/ANALYTIC_THEORY.md) — Bellman/tree-knapsack formulation and current theoretical interpretation.
- [`docs/RESEARCH_SUMMARY.md`](docs/RESEARCH_SUMMARY.md) — full research evolution and claim boundaries.
- [`docs/EXPERIMENT_TIMELINE.md`](docs/EXPERIMENT_TIMELINE.md) — selective decisive-experiment timeline.
- [`docs/INTERFACE_PROPOSAL.md`](docs/INTERFACE_PROPOSAL.md) — minimal UI, personalization path, and continuous-stream direction.
- [`docs/HUMAN_READING_HYPOTHESIS.md`](docs/HUMAN_READING_HYPOTHESIS.md) — cautious testable translation to human reading.
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — environment, commands, and external-resource notes.
- [`experiments/`](experiments/) — experiment code and index.
- [`results/`](results/) — selected machine-readable results and index.
- [`NOTICE.md`](NOTICE.md) — data/model licensing, audit corrections, and claim boundaries.

## Reproducibility

Python 3.12 is the reference environment for the later experiments. A practical installation is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The quickest self-contained check is:

```bash
python experiments/ap_t1_analytic_sanity.py
```

Real Wikispeedia/grounded-language experiments download third-party resources at runtime. Those resources and model weights are not redistributed by this repository. See `REPRODUCIBILITY.md` and `NOTICE.md` before rerunning or redistributing data.

## License

Project-authored code and text are MIT-licensed unless a file states otherwise. Third-party datasets, model weights, and derived resources remain subject to their original terms.

---

## 日本語要約

この研究で見えてきた中心原理は、**分からないものを見つけた瞬間に全部解決するのではなく、まず読み続け、文脈で自然解決する可能性を残し、それでも下流の理解に重要になったものだけを必要な時点・必要な深さで解決する**というものです。

完全な問い合わせ木が分かっている場合、この問題は動的計画法で解析的に解けます。現在の中心課題は、限られた観測・記憶・時間のもとで、そのfull-information oracleをどこまで近似できるかです。

UIとしては、文章を主役のまま保ち、AIが後続理解に重要そうな箇所だけを疎に示し、`不要 / 1行 / 例 / 詳しく / あとで`のような小さい選択で説明量を決める方式が最小案です。その選択履歴から個人化し、将来的には必要な補足だけが本文の流れに自然に混ざるcontinuous adaptive streamへ拡張できます。ただし、このUIの人間に対する効果はまだ直接検証していません。
