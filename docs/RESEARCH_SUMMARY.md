# Research Summary

## Research question

The project asks whether an unknown hypertext can be read or navigated efficiently under tight memory and action budgets, and what minimal state is actually needed.

The research program deliberately separates two constructs:

- **Reading / information acquisition:** coverage of latent core information under a reading budget.
- **Goal-directed navigation:** reaching a target under a navigation budget, used later as a mechanism and stress-test environment.

Navigation results are therefore not automatically claims about human comprehension.

## Evolution of the theory

### Stage 1 — Local semantic scoring

Early experiments tested whether a compact local representation was sufficient for information acquisition. The robust direction was a low-dimensional additive score combining information such as:

- current sentence or page relevance,
- anchor/link relevance,
- containing-sentence or paragraph relevance,
- limited novelty information.

A recurring finding was that **context around the link materially improves link-value estimation**. Larger recurrent, nonlinear, and global-memory models frequently improved an intermediate predictor while failing to improve the end-to-end objective.

### Stage 2 — Immediate-parent runner-up memory

Under a controlled distance-like cue, retaining the immediate-parent runner-up was useful. This motivated an initially strong hypothesis:

> A single abandoned alternative may be enough history for efficient navigation.

Direct counterfactual utility calibration gave additional gains under some controlled-cue conditions and independently replicated, including hidden noise-mixture tests. However, those results still depended on a controlled observation channel.

### Stage 3 — Cue-universality test

The one-scalar hypothesis was then tested under a spectral/non-distance bridge derived from the same Wikipedia graph topology but without using shortest-path distance directly as the observation score.

The hypothesis failed. The runner-up policy could become materially worse than local-only navigation. Recalibrating the margin did not rescue it.

This changed the theory from:

> one scalar history is sufficient

into:

> the useful amount and representation of memory depend on the observation/value channel.

### Stage 4 — Frontier oracle and trajectory utility

Oracle experiments showed that richer abandoned-alternative frontiers could contain large latent value. Yet observable score-based frontiers and myopic value models could not recover that value end-to-end.

The key distinction was:

> "this alternative is locally closer/better" is not equivalent to "returning to this alternative and then continuing with the same imperfect policy improves the final outcome."

The teacher was therefore changed from a myopic state-local target to **counterfactual trajectory utility**:

\[
Y = S_B^{\text{BACK}} - S_B^{\text{CONTINUE}}.
\]

This changed the sign of the deployable effect from negative to positive.

### Stage 5 — Bounded-memory compression

The successful richer-history policy was then compressed aggressively.

Key results:

- K=8 could be reduced to **K=4** with almost no early-budget loss.
- Full ancestor history could be reduced to the **immediately previous decision point**.
- K=3 or less failed the preregistered retention/safety criterion in the tested bridge.
- A 6-feature value model could be reduced to **3 features**:
  - candidate degree,
  - origin candidate count,
  - relative score.
- A frozen 3-feature / K=4 / one-shot policy replicated on 12 new seeds × 500 tasks.

This restored an O(1)-in-depth policy, but the constant state is richer than one scalar.

### Stage 6 — Capacity plateau, delay, and adaptive memory

Later tests examined the human-inspired ideas of larger short-term buffers, delayed reconsideration, and load-adaptive capacity.

Results:

- K=7 or K=9 did not materially outperform K=4.
- Several-page delayed reconsideration did not establish an advantage over next-decision-point reconsideration.
- Persistent multi-page pending-link retention was not supported.
- A task-demand K=3/4/5 rule did not materially beat fixed K=4.
- A resource-saving K=5/4/3 rule reduced mean stored candidates by about 20.9% while losing about 0.4 percentage points versus fixed K=4.

The most defensible current interpretation is therefore:

> **Acquire local context first; keep a small and short-lived option buffer; reconsider it quickly and at most once; treat memory capacity as a resource/performance trade-off rather than assuming that more retained links always help.**

## Current compact bridge policy

Under the Chameleon spectral/non-distance bridge:

- local context first,
- history window: immediate previous decision point,
- candidate memory cap: 4,
- features: candidate degree, origin candidate count, relative score,
- model: frozen Ridge counterfactual trajectory utility,
- threshold: 0.05,
- maximum discretionary interventions: 1,
- preferred reconsideration: next decision point / TTL=1.

## Decisive confirmatory result: AP-S43

Frozen after model/feature/cap selection; 12 completely new seeds × 500 tasks = 6,000 tasks.

- S@16: **+3.9167 pp**, 95% cluster+task CI **[+3.133,+4.683] pp**, 12/12 seeds positive.
- S@32: **+1.1500 pp**, 95% CI **[+0.250,+2.067] pp**, 9/12 seeds positive.

## Resource-saving result: AP-S49

12 new seeds × 500 tasks = 6,000 tasks.

Relative to fixed K=4, the high-branching resource-saving cap reduced mean stored alternatives from 2.987 to 2.364 per forward page (about 20.9%) with:

- S@16 change: **-0.433 pp**, CI **[-0.733,-0.150] pp**.
- S@32 change: **-0.400 pp**, CI **[-0.633,-0.183] pp**.

This is a memory/performance trade-off, not a performance improvement.

## Claim boundary

### Supported as computational findings

- Link value should be evaluated with surrounding context rather than anchor text alone.
- More model/state complexity does not automatically improve end-to-end performance.
- One abandoned alternative is not universally sufficient.
- In the tested spectral bridge, a short-lived top-4 buffer is enough to recover the useful deployable signal found by richer frontiers.
- Counterfactual trajectory utility is more appropriate than a myopic local-value teacher for deciding whether to return.
- Repeated discretionary returns and long-lived pending buffers are not supported.

### Not yet established

- That K=4 is a human working-memory constant.
- That the policy improves human comprehension or retention.
- That the same compact policy transfers to real anchor + containing-context multi-hop tasks.
- That this is an optimal human hyperlink-reading strategy.

## Required next external-validity gates

1. At least 400 independent real anchor/context semantic tasks with page/source-target clustering and real multi-hop outcomes where possible.
2. Human experiments comparing immediate-click, straight-through, and bounded-deferred-link conditions on comprehension, retention, reading time, navigation behavior, and cognitive load.
