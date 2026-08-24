# AP-RS13 preregistration — K=1 trigger-only visible semantic state

## Motivation

AP-RS12 found a large one-shot oracle opportunity under anchor-only local navigation: K=1 already recovered most of the K=4 oracle opportunity, while AP-RS8's learned K=4 visible gate was near-null. This points to **trigger timing** rather than candidate selection as the primary bottleneck.

AP-RS13 therefore fixes the deferred candidate to the immediately previous page's rank-2 anchor candidate (K=1) and learns only whether to return to it after reading the current page.

## Information regime

No unvisited destination page body, degree, or prefetched content is used.

Three prespecified linear direct-trajectory-utility feature blocks are compared on fit/tune targets only:

1. **V0 minimal anchor trigger** — previous-page option count, rank-2 anchor score relative to the current page's best outgoing anchor, and decision step.
2. **V1 visible navigation state** — V0 plus current-page candidate count, current top-1 anchor score, current top1-top2 gap, and previous rank1-vs-rank2 anchor margin.
3. **V2 visible semantic state** — V1 plus semantic similarity of the already visited current and parent article bodies to the known target title, body semantic progress, current/parent title semantic similarity, the remembered rank-2 link's containing-paragraph similarity, and anchor-vs-paragraph disagreement.

V2 is compatible with a sidebar assistant that remembers and scores information the user has already seen. It does **not** prefetch the rank-2 destination page.

## Data split

- Real Wikispeedia graph and HTML/plaintext corpus.
- Real human source-target mission distribution.
- Target-disjoint fit/tune/test split with a new fixed seed.
- Planned sizes: fit 1,500; tune 800; test 2,400 missions, subject to available missions.
- Full visited-page set retained for cycle avoidance; this is separate from the K=1 deferred-option memory construct.

## Model and teacher

- Anchor-only local navigation.
- Deferred candidate: previous page rank-2 anchor candidate only.
- Direct one-intervention teacher at S@16: `y = success(BACK+rank2 then local) - success(CONTINUE local)`.
- Linear Ridge with standardized features.
- Variant and decision threshold selected on tune targets only, maximizing S@16 gain subject to mean S@32 harm no worse than -0.5 percentage points.
- At most one discretionary return; BACK + alternative traversal costs two actions.

## Primary decision rule

PASS iff the frozen selected model on test targets satisfies all of:

1. test n >= 400;
2. S@16 improvement >= +2.0 percentage points versus anchor-only local;
3. target-cluster bootstrap 95% CI lower bound for S@16 improvement > 0;
4. at least 6/8 target buckets have positive S@16 point effects;
5. mean S@32 effect >= -0.5 percentage points;
6. target-cluster S@32 CI lower bound >= -1.0 percentage point.

Also report:

- non-tie sign AUC for each feature block;
- K=1 oracle opportunity on the same test missions;
- learned/oracle S@16 recovery fraction;
- intervention rate.

## Interpretation

- PASS: trigger timing is recoverable from visible/visited semantic state; a non-prefetch sidebar memory/reranking assistant becomes a plausible implementation path.
- FAIL with high non-tie AUC: state-level utility is identifiable but deployment threshold / trajectory distribution remains the bottleneck.
- FAIL with weak AUC: visible post-click state is insufficient; recovering the oracle gap likely requires different information (possibly prefetch, which would be a separate information regime).

This remains simulated goal-directed navigation, not a human comprehension or retention experiment.
