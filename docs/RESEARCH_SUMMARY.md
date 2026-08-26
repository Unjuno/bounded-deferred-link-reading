# Research Summary

## Research question

This project began with a practical question:

> When a reader encounters something potentially important but not yet understood, should it be resolved immediately or deferred while more context is acquired?

The work uses hypertext navigation, synthetic recursive-query environments, and grounded natural-language tasks as computational test beds. These environments are not direct measurements of human comprehension.

Two constructs remain distinct throughout:

- **reading / information acquisition** — acquiring useful information under limited time and attention;
- **goal-directed navigation / query scheduling** — reaching a target or collecting useful information under an action budget.

## Evolution of the theory

### Stage 1 — Context-sensitive local value

Early information-acquisition experiments supported compact semantic scoring and showed that containing context can improve the estimated value of a link or concept. Large recurrent/global state was not consistently necessary.

### Stage 2 — Short-lived deferred alternatives

Controlled navigation experiments initially suggested that retaining a recent runner-up could help. A later spectral/non-distance bridge showed that one scalar alternative was not universally sufficient.

The theory therefore changed from “remember one runner-up” to “retain a small frontier of deferred options whose value is reevaluated after more context.”

### Stage 3 — Downstream trajectory value

Richer-frontier oracle experiments showed substantial unrealized opportunity, while local/myopic scores often failed to identify which deferred branch should be revisited.

The important distinction became:

> Local attractiveness is not the same as the value of taking an action and then continuing under the future policy.

This motivated trajectory-level counterfactual teachers.

AP-S43 reproduced a compact spectral-bridge policy with a short-lived top-4 frontier and one discretionary reconsideration:

- S@16: **+3.9167 pp**, 95% CI **[+3.133,+4.683]**;
- S@32: **+1.1500 pp**, 95% CI **[+0.250,+2.067]**.

Later capacity/resource tests showed that the useful frontier could often be compressed substantially. The literal value `K=4` is environment-specific, not a human-memory claim.

### Stage 4 — Real Wikispeedia and the observability bottleneck

The project then moved to real Wikispeedia paths, HTML, anchors, paragraphs, and article-body semantics.

Observationally, successful human paths showed strong target-directed semantic progress, and BACK behavior was predominantly short-range. One-step returns followed by replacement branches were associated with semantic correction.

However, deployable causal policies transferred much less cleanly.

AP-RS5, using an equal anchor/context semantic scorer, failed its preregistered +2 pp criterion. At the same time, anchor-only navigation was substantially stronger than the equal anchor/context mixture for explicit-target navigation.

Subsequent trigger experiments sharpened the main problem:

- there can be large oracle opportunity;
- a useful alternative can exist;
- but visible semantic state may still be insufficient to identify the correct intervention reliably.

AP-RS13 is a representative case: a large K=1 oracle opportunity remained, but the visible semantic trigger policy hurt held-out performance.

AP-RS14 selected no intervention under a more robust sparse trigger.

AP-RS15 added body-prefetch information, but observability improved only slightly and the calibrated policy again selected no intervention despite large K=1 oracle opportunity.

The key lesson was therefore an **observability bottleneck** rather than a lack of potential value.

### Stage 5 — Recursive LM-style querying: AP-LM1

The research then generalized from fixed links to recursive queries, where an answer can itself reveal new unresolved items.

A recursive query is therefore not just `q -> answer`; it exposes a subtree of future information-acquisition opportunities.

AP-LM1 compared local/visible policies with a teacher that predicted exact frozen-subtree opportunity value.

Held-out results:

- B=16: recursive learned policy vs visible greedy **+4.765 pp**, CI **[+4.449,+5.078]**, 12/12 seeds positive;
- B=32: **+7.034 pp**, CI **[+6.531,+7.546]**;
- latency-aware prediction added a smaller but clear benefit;
- K=8 was the smallest tested frontier within 1 pp of full at both budgets.

This established that, in a recursive setting, **downstream subtree value** can matter materially beyond immediate visible value.

### Stage 6 — Async/slack-aware querying: AP-LM2

AP-LM2 added the possibility that reading/work can continue while a query is pending.

The relevant cost is then not raw latency alone but the amount of latency that actually blocks useful work.

Held-out results supported:

- slack-aware scheduling over latency-only scheduling;
- recursive-subtree teacher over a matched immediate-value teacher;
- compact frontier approximations;
- adaptive capacity that reduced retained state with little performance loss.

This strengthened the view that query value depends on both downstream information and timing.

### Stage 7 — Grounded natural-language transfer: AP-LM3B

AP-LM3A was treated only as development provenance because an implementation audit found that pre-query latency depended on hidden answer length.

AP-LM3B corrected the cost model so only visible query text determined pre-query cost and then ran a fresh held-out confirmation on grounded natural-language recursive answer trees.

Held-out sample:

- **1,197 missions**;
- **753 target clusters**.

At B=12:

- recursive vs matched immediate: **+3.112 pp**, CI **[+2.292,+3.951]**, 8/8 buckets positive;
- recursive vs visible greedy: **+3.109 pp**, CI **[+2.238,+4.010]**, 8/8 buckets positive.

Capacity audit:

- K=8 was within **0.551 pp** of full;
- K=4 was substantially below full in this natural-language construction.

At B=20, the main policies were essentially saturated near oracle, showing that scheduling differences vanish when the budget is no longer binding.

### Stage 8 — Analytic synthesis

At this point the recurring policy pattern was stable enough that continued broad heuristic search had diminishing value.

The problem can be written directly as an optimal-control problem.

For one unresolved item `q` in state `s`, the key comparison is:

`resolve q now` vs `defer q and continue reading`.

The working stopping rule is:

`ASK(q) iff expected_value(resolve now) > expected_value(defer)`.

For a known recursive query tree with integer costs, the exact full-information problem is a precedence-constrained tree knapsack and can be solved by dynamic programming.

The correct value object is therefore a **budget-conditioned recursive value curve** `V(q,b)`, not one static importance scalar.

The research question has consequently shifted from:

> Which heuristic should we try next?

into:

> **How closely can a bounded, partially observed reader or agent approximate the analytic full-information oracle?**

See `docs/ANALYTIC_THEORY.md`.

## Theory-following sanity check: AP-T1

AP-T1 is not a new discovery experiment. It checks finite-state consequences of the analytic model on 200 small random recursive trees.

It verified:

- monotonicity in budget;
- monotonicity in frontier capacity;
- full oracle >= immediate greedy;
- saturation when the budget permits all useful work;
- zero contradictions in 144 enumerated single-unknown defer/resolve threshold checks.

This is a consistency check of the analytic formulation, not evidence about human cognition.

## Current scientific position

### Supported by the computational program

- Immediate resolution of every uncertainty is not generally necessary for good bounded information acquisition.
- Deferring an option can have positive value because later context changes its estimated usefulness.
- Local relevance and downstream/trajectory value are distinct.
- Recursive answer generation makes downstream subtree value especially important.
- Query scheduling matters most when the information budget is binding.
- Useful full-information actions can exist even when visible state is insufficient to identify them reliably.
- Small bounded frontiers can often approximate much richer history, although the required capacity is environment-dependent.
- Async scheduling should account for effective blocking/slack, not only raw latency.

### Not established

- An optimal human reading strategy.
- A universal human memory capacity such as K=4 or K=8.
- A causal claim that observed human BACK behavior is generated by deferred-option memory.
- A validated human-comprehension benefit from AI highlighting or deferred-frontier UI.
- Reliable handling of hallucinating/stochastic generative answers in the full real-LM setting.

## Interface proposal

The theory suggests a concrete interface direction without claiming that it has already been validated in humans:

- AI estimates which unclear spans have high downstream importance;
- important candidates receive restrained emphasis such as bold/underline/markers;
- explanations remain optional rather than automatically interrupting reading;
- unresolved items are stored in an external frontier and reprioritized as context grows;
- chat is used as a context-conditioned explanation channel;
- new unknowns introduced by an answer return to the frontier rather than forcing depth-first recursive chat;
- explanation depth adapts to downstream importance.

See `docs/INTERFACE_PROPOSAL.md` for the concrete proposal and its validation boundary.

## Current stopping point

The broad computational exploration phase is complete enough to stop without another large parameter sweep.

The main remaining work is theoretical or external-validity work:

- formal properties and approximation guarantees for the analytic control problem;
- optional stochastic/hallucinating-answer extensions;
- if desired, one small human-interface validation study.

These are follow-up directions, not prerequisites for preserving the current results as a completed computational research phase.
