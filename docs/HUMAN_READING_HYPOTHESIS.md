# Human-Reading Hypothesis

## Status

This document translates the computational program into **testable human-reading hypotheses**. It is not a validated prescription.

The early project emphasized a very concrete hyperlink rule: retain a few recent alternatives and reconsider them soon. Later work showed that the more general object is not a particular K value or one fixed reconsideration schedule. The current hypothesis is broader:

> **When an uncertainty appears, do not assume it must be resolved immediately. Continue reading when the expected value of more context exceeds the value of resolving now; resolve it when its downstream importance and expected explanation value become large enough.**

## Why deferral may help

Continuing to read can:

- resolve an unfamiliar term from later context;
- reveal that it is peripheral;
- show that it is a prerequisite for what follows;
- improve the wording and scope of a later question.

Thus `DEFER` is not equivalent to `DISCARD`. It preserves an option to resolve later.

For an unresolved item `q` in context `s`, the qualitative decision is:

`resolve now` versus `defer and keep reading`.

The analytic formulation in `docs/ANALYTIC_THEORY.md` makes this an optimal-stopping/control problem.

## Downstream importance, not difficulty alone

A difficult term is not automatically worth interrupting reading for. A concept can be obscure but irrelevant to the rest of the document. Conversely, a simple-looking concept may be central to every later argument.

A human-facing prediction is therefore:

> The value of clarification should depend strongly on **downstream dependency**—how much later understanding depends on the concept—not only on current subjective difficulty.

## Bounded unresolved state

The computational experiments often found diminishing returns from retaining large frontiers, but the capacity needed depended on the environment:

- the controlled spectral bridge plateaued near K≈4;
- AP-LM1 required about K=8 among the tested values;
- AP-LM2 again reached near-full performance around K=4;
- AP-LM3B needed more capacity than the simplest fixed-link bridge.

These values do **not** estimate a universal human working-memory capacity.

The human hypothesis is only that carrying many unresolved items has a cost, so external memory or AI assistance may help preserve useful unresolved options without forcing them into working memory.

## Relation to observed Wikispeedia behavior

Real successful Wikispeedia paths showed strong target-directed semantic progress, and BACK behavior was predominantly short-range. Short-range return followed by branch replacement was associated with semantic correction.

That pattern is compatible with quick reconsideration, but it is observational. It does not prove that humans were using the computational deferred-option mechanism.

## A conservative human-readable rule

A cautious practical translation is:

> **If something is unclear, first ask whether you need it to understand what comes next. If not, keep reading. If it becomes a prerequisite, clarify it at the minimum depth needed to continue. Keep unresolved but potentially important items externally rather than trying to solve or memorize everything at once.**

This is still a hypothesis, not an established optimal reading method.

## AI-assisted reading hypothesis

The current interface proposal makes an additional, unvalidated prediction: AI can reduce two burdens that unaided readers face.

1. **candidate detection** — noticing which unclear items are likely to matter downstream;
2. **explanation selection** — choosing whether no explanation, a one-line gloss, an example, or a detailed explanation is appropriate.

A minimal interface can therefore:

- sparsely emphasize high-downstream-dependency spans;
- let the reader choose `No explanation / 1 line / Example / Detailed / Later`;
- keep routine supplements in the reading flow;
- learn explanation needs from those choices;
- retain deferred items internally rather than requiring the reader to manage a visible backlog.

See `docs/INTERFACE_PROPOSAL.md`.

## Proposed human validation

A compact first study could compare:

1. **document + ordinary chat**;
2. **document + sparse AI salience + explanation-choice buttons**;
3. **the same interface with personalized explanation defaults learned from prior choices**.

Primary outcomes:

- comprehension/task success;
- information acquired per unit time;
- total reading time;
- number and depth of explanations.

Secondary outcomes:

- context-switch time;
- explanations that later proved unnecessary because context resolved the issue;
- subjective interruption/cognitive load;
- calibration of the AI's importance/explanation predictions.

## Falsifiable predictions

- Immediate explanation of every difficult item should be inefficient when many items later self-resolve or prove irrelevant.
- Downstream-dependency-based salience should be more useful than difficulty-only highlighting.
- Reader-controlled explanation depth should reduce unnecessary explanation cost relative to fixed verbosity.
- Sparse interaction history should allow better personalization than a non-personalized fixed explanation policy.
- There should be a point at which more unresolved/frontier state yields diminishing practical value.

## Boundary

The repository has **not** yet shown that these predictions improve human comprehension, retention, or cognitive load. Navigation performance, semantic similarity, synthetic query reward, and grounded-language proxy utility are not substitutes for direct human outcomes.
