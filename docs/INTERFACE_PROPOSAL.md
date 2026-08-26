# Interface Proposal: Deferred-Query Reading with AI-Guided Salience

## Status

This document is a **design proposal derived from the current theory and computational results**. It is not a validated human-comprehension result.

The core design claim is narrower than “chat is the optimal interface”:

> A reading interface should help the user identify high-downstream-value uncertainties, preserve them without forcing immediate resolution, and make context-conditioned explanation cheap when resolution becomes worthwhile.

The interface therefore externalizes two burdens that ordinary reading leaves to the user:

1. **importance estimation** — which unclear terms are likely to matter later;
2. **deferred-query memory** — which unresolved items should remain available for later reconsideration.

## Proposed layout

### 1. Main reading pane

The document remains primary. The interface should not continuously replace reading with explanations.

AI predicts which spans have high expected downstream dependency and applies restrained emphasis.

Possible presentation:

- **bold** for the highest-priority concepts;
- lighter underline or margin marker for medium-priority concepts;
- no mark for low-priority or likely self-resolving concepts.

The ranking target is not “difficulty” alone. It is closer to expected future value:

`priority(q) ~= unresolvedness * downstream_dependency * (1 - contextual_resolution_probability) + recursive_gain - query_cost`.

The exact score is model-dependent; the design principle is that **important-to-understand-later** is more useful than simply **hard-to-understand-now**.

### 2. Inline query affordance

Selecting or activating a highlighted span should expose a small action menu rather than an automatic popup.

Suggested actions:

- `1-line explanation`
- `Explain in this context`
- `Why does this matter here?`
- `Keep for later`
- `Ask a custom question`

The surrounding text should be supplied automatically to the assistant so the user does not need to reconstruct context manually.

### 3. Deferred unresolved frontier

Unresolved items should be stored in a compact frontier rather than in the user's working memory.

Each item can retain:

- the original span;
- source location;
- current predicted importance;
- why it may matter downstream;
- whether later context has partially or fully resolved it;
- optional explanation history.

The frontier should be dynamically reprioritized as the user reads.

Possible statuses:

- `important now`
- `pending`
- `likely resolved by context`
- `probably unnecessary for current goal`

A small visible top set is preferable to showing a large unresolved backlog. Lower-priority items can remain latent/searchable.

### 4. Chat as an explanation channel, not the main navigation structure

Ordinary chat encourages depth-first recursion:

`question -> answer -> follow-up -> answer -> deeper follow-up ...`

The proposed interface instead keeps three separate objects:

1. the **main reading stream**;
2. the **unresolved frontier**;
3. the **chat/explanation channel**.

When an answer introduces a new unclear concept, that concept should normally be added back to the frontier rather than automatically becoming the next question.

This preserves the user's original task and reduces accidental recursive detours.

### 5. Adaptive explanation depth

The action should be modeled as `(question, explanation depth)`, not just `question`.

Useful depth levels:

- **gloss** — one sentence;
- **short** — enough to continue reading;
- **example** — explanation plus one example;
- **formal** — derivation or technical detail;
- **deep** — tutorial-level treatment.

The default should depend on downstream dependency. A peripheral term may need only a gloss; a concept supporting the rest of the document may justify a deeper explanation.

### 6. Threshold-based proactive guidance

The assistant should not constantly interrupt the user.

Proactive prompts should occur mainly when an item's estimated resolution value crosses a threshold relative to continued reading.

Examples:

- `This concept is now used as a prerequisite in the next section.`
- `You can keep reading; this term is explained shortly.`
- `A one-line definition is probably sufficient here.`

The system should also be allowed to recommend **not asking yet**.

## Intended interaction loop

```text
READ
  |
  +-- unclear span appears
  |       |
  |       +-- AI estimates downstream importance
  |       +-- optional restrained emphasis
  |       +-- add to unresolved frontier
  |
continue reading
  |
  +-- context may resolve / downgrade / upgrade item
  |
reconsideration point
  |
  +-- if resolve-value <= defer-value: keep reading
  |
  +-- if resolve-value > defer-value: surface one-click explanation
                                  |
                                  +-- answer may add new unknowns
                                      back to frontier
```

## Why this differs from common reading assistants

The proposal is not simply “put chat next to a document.”

It adds three control mechanisms:

1. **AI-guided salience** — the system helps decide what may matter;
2. **deferred resolution** — highlighted uncertainty does not imply immediate explanation;
3. **frontier management** — unresolved questions are reprioritized and retired over time.

These are direct consequences of the defer-vs-resolve formulation.

## Minimal product specification

A minimally useful prototype could contain:

- document viewer with three-level AI emphasis;
- click/selection actions for gloss, contextual explanation, or defer;
- side panel showing the top unresolved items;
- automatic priority updates as reading position changes;
- chat anchored to the selected span and local context;
- answer-depth selector;
- automatic return of follow-up unknowns to the frontier.

No autonomous interruption is required for the first prototype.

## Testable interface hypotheses

These remain hypotheses until a human study is run.

### H-UI1 — AI salience reduces candidate-detection cost

AI-ranked emphasis should reduce the effort required to identify which unclear items are worth attention relative to plain text.

### H-UI2 — Deferred access beats forced explanation

Emphasis plus optional/deferred explanation should outperform automatically opening explanations at every difficult term because forced explanations remove contextual self-resolution and add interruption cost.

### H-UI3 — Dynamic reprioritization beats static highlighting

If importance changes with later context, updating salience/frontier rank should outperform static pre-highlighting.

### H-UI4 — Frontier chat beats linear recursive chat

Returning follow-up unknowns to a ranked frontier should reduce unnecessary depth-first exploration compared with standard linear chat.

### H-UI5 — Adaptive depth reduces explanation cost

Selecting answer depth according to downstream dependency should reduce explanation time/tokens without materially reducing task success.

## Minimal validation study

If one human-interface experiment is eventually run, the clean comparison is:

1. plain document + ordinary chat;
2. AI salience highlighting + ordinary chat;
3. AI salience highlighting + deferred frontier + chat.

Primary measurements:

- comprehension/task success;
- total reading time;
- number of queries;
- explanation time/tokens;
- queries that later proved unnecessary because context resolved them;
- interruption/cognitive-load ratings.

The most informative comparison is (2) vs (3): highlighting tests candidate detection, while the frontier tests whether deferred resolution adds value beyond salience guidance.

## Design boundary

The computational work supports the **structure of the proposal**, not its human-factors effectiveness.

In particular, the current evidence does not establish:

- the optimal visual emphasis style;
- how much text should be highlighted;
- the optimal number of visible frontier items for humans;
- whether proactive prompts improve or harm concentration;
- whether this interface improves retention or comprehension.

Those are interface questions, not conclusions already established by the navigation/query experiments.
