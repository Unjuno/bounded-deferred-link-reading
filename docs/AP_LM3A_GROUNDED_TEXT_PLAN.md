# AP-LM3A — Grounded Natural-Language Recursive Answer Trees

## Purpose

Bridge AP-LM2 from latent synthetic query nodes to real natural-language strings before adding generative-LM error. A query is a visible linked concept from Wikispeedia. Issuing the query reveals that concept's real article body as a fixed grounded answer; visible concepts inside that answer become further query candidates.

This is deliberately a **grounded answer surrogate**, not yet a generative LM. The objective is to test whether the recursive-subtree-value mechanism transfers when candidate cues and answers are real text rather than scalar latent variables.

## Development pilot

The first run is development-only and uses only the middle 20% target partition. The held-out confirmatory target partition is not touched during pilot design.

Frozen before pilot:
- real Wikispeedia graph, HTML anchors/paragraph contexts, and plaintext article bodies;
- target-disjoint 40% fit / 20% pilot / 40% held-out target split;
- six initially visible query candidates per task;
- recursive expansion to depth 3, maximum 3 children per answer, maximum 18 nodes;
- answer latency derived deterministically from fixed answer length, clipped to 1–8 units;
- budgets 12 and 20;
- same deployable features for immediate-value and recursive-value Ridge models;
- fit N=700 missions; pilot N=280 missions;
- K sweep 2/4/8 plus full frontier;
- frozen AP-LM2 adaptive top4/top8 compression rule.

## Natural-language information regime

For each candidate query, the deployable policy sees only information available before requesting the answer:
- anchor-to-target semantic similarity;
- containing-context-to-target similarity;
- linked-title-to-target similarity;
- combined visible score;
- answer-length latency estimate;
- recursive depth;
- already-observed parent answer reward;
- remaining budget;
- frontier size and visible-score rank.

The unqueried destination body is hidden from the deployable policy. Once queried, its fixed article body is revealed and its semantic contribution is observed.

## Utility and teachers

Ground-truth answer utility is the squared nonnegative cosine similarity between the revealed answer body and the target article body. This is a semantic target-alignment proxy, not a human-comprehension score.

Two matched Ridge models use exactly the same observable features:
- **immediate teacher:** utility of the candidate answer alone;
- **recursive teacher:** exact best ancestor-closed utility available from the candidate's entire answer-generated subtree under remaining budget.

The recursive teacher therefore tests the same mechanism identified in AP-LM1/2 while removing the synthetic latent-quality generator.

## Pilot contrasts

1. recursive-value vs matched immediate-value policy at budget 12;
2. recursive-value vs visible cue/latency greedy at budget 12;
3. both contrasts at budget 20;
4. K=2/4/8 vs full recursive frontier;
5. AP-LM2 adaptive top4/top8 compression vs fixed K=8.

After the pilot, confirmatory PASS thresholds will be written here **before** the held-out 40% target partition is run. No held-out target will be used to choose those thresholds.

## Interpretation boundaries

- This is real natural-language content but not a generative-LM response distribution.
- Hallucination is absent by construction.
- Latency is a deterministic answer-length proxy, not measured API wall time.
- The target article body is used only to define hidden evaluation/teacher utility; deployable features do not receive unqueried body text.
- Results concern query scheduling/information acquisition, not direct human comprehension.

## Decision branch after pilot

- If recursive teacher has a stable positive effect and the recursive policy clearly beats visible greedy, preregister and run the held-out confirmatory target split.
- If recursive teacher collapses while oracle/greedy gap remains large, diagnose feature observability before touching held-out targets.
- If the recursive policy itself has little advantage over visible greedy, do not claim natural-language transfer; redesign the utility/task on fresh partitions.
