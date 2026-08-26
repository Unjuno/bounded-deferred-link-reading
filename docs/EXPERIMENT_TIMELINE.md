# Decisive Experiment Timeline

This is a selective timeline of experiments that materially changed the theory. It omits many intermediate ablations while preserving negative results and implementation corrections that affected interpretation.

## 1. Local reading / information acquisition

**Question:** Can compact local semantic state recover useful information without large recurrent memory?

**Result:** Low-dimensional additive scoring was competitive. Anchor/link semantics became more useful when combined with containing context in the information-acquisition construct. Larger recurrent/global models did not show a stable end-to-end advantage.

**Theory update:** Start with compact context-sensitive value; model complexity is not automatically useful.

---

## 2. Immediate-parent runner-up under a controlled cue

**Question:** Is one remembered abandoned alternative enough history?

**Result:** Under a distance-like controlled cue, a recent runner-up was useful and sometimes captured much of richer-history benefit.

**Theory update:** A very small memory representation looked plausible, but only under that observation/value channel.

---

## 3. Spectral/non-distance bridge: one-scalar failure

**Question:** Is the one-runner representation cue-universal?

**Result:** No. Under a spectral/diffusion-style cue, the same runner could materially hurt navigation; recalibration did not rescue the universal claim.

**Decision:** FAIL for universal one-scalar sufficiency.

**Theory update:** Useful memory representation depends on the observation/value channel.

---

## 4. Frontier oracle and trajectory value

**Question:** Is deferred history useless under the spectral cue, or merely difficult to exploit?

**Result:** Richer deferred frontiers contained large oracle value, while local score and myopic-value models failed to recover it reliably.

The central distinction became:

> A locally attractive alternative is not necessarily the action that improves the final trajectory under the future policy.

Trajectory-level counterfactual supervision converted part of the latent frontier value into deployable gain; repeatedly reusing the same gate degraded performance, and short-horizon optimization required long-horizon safety constraints.

**Theory update:** Predict downstream/trajectory utility rather than local attractiveness alone.

---

## 5. Bounded-memory compression and AP-S43

**Question:** How much deferred state is required once the right value target is used?

**Result:** Richer history compressed to a short-lived recent frontier. AP-S43 froze a compact top-4, one-shot trajectory-utility policy and replicated it on 6,000 fresh tasks:

- S@16: **+3.9167 pp**, 95% CI **[+3.133,+4.683]**, 12/12 seeds positive;
- S@32: **+1.1500 pp**, 95% CI **[+0.250,+2.067]**.

Later capacity sweeps found little extra gain from K=7/9 in this bridge. AP-S49 reduced mean retained alternatives by about 20.9% for roughly 0.4 pp loss.

**Boundary:** these K values are environment-specific and are not estimates of human working-memory capacity.

**Implementation correction:** an early bounded-history-window off-by-one result was discarded and rerun on fresh seeds.

---

## 6. Real Wikispeedia observational semantics

**Question:** Does short-range branch correction appear in real human navigation paths?

**Result:** Successful paths showed strong nonterminal target-directed body-semantic progress. First eligible BACK episodes were predominantly short-range; one-step return followed by branch replacement was associated with positive semantic correction on average.

**Boundary:** observational human navigation does not establish a bounded-memory cognitive mechanism or comprehension benefit.

---

## 7. Real visible causal transfer and observability bottleneck

**Question:** Can a deployable bounded-deferred policy exploit real anchor/context semantics?

**AP-RS5:** The preregistered equal anchor/context policy missed its +2 pp transfer criterion; anchor-only local navigation was substantially stronger for explicit-target navigation. An audit also found one graph-assisted feature (`candidate_outdegree`), so RS5 is not strictly visible-only.

Subsequent phases removed or stressed hidden metadata and tested visible triggers, wrong-turn triggers, visited-memory approximations, K=1 oracle opportunity, robust sparse gates, and body prefetch.

Representative result: AP-RS13 retained a large K=1 oracle opportunity, yet the visible semantic trigger policy hurt held-out S@16. AP-RS14 selected no intervention. AP-RS15 body prefetch improved observability only slightly and still selected no intervention despite large oracle opportunity.

**Theory update:** The central real-data problem is often **observability**: a valuable full-information action can exist while visible state is insufficient to identify when to take it.

---

## 8. AP-LM1 — recursive query frontier

**Question:** What changes when an answer can reveal new unresolved questions?

**Result:** Recursive-subtree value prediction beat visible greedy selection:

- B=16: **+4.765 pp**, CI **[+4.449,+5.078]**, 12/12 seeds positive;
- B=32: **+7.034 pp**, CI **[+6.531,+7.546]**.

Latency-aware features also helped. K=8 was the smallest tested frontier within 1 pp of full at both budgets.

**Theory update:** A query should be valued as an entry into a future information subtree, not only by its immediate answer.

---

## 9. AP-LM2 — asynchronous/slack-aware scheduling

**Question:** How should query latency be valued when useful reading/work can continue while a query is pending?

**Result:** Slack-aware recursive scheduling beat raw-latency recursive scheduling. A recursive-subtree teacher beat a matched immediate-value teacher. K=4 was sufficient in this synthetic construction, and adaptive capacity reduced retained state by about 18% with little performance loss.

**Theory update:** Relevant query cost is effective blocking/opportunity cost, not raw latency alone.

---

## 10. AP-LM3A audit and AP-LM3B grounded natural-language confirmation

AP-LM3A bridged to real natural-language answer bodies but an audit found that pre-query latency depended on hidden answer length. It is therefore development provenance only.

AP-LM3B removed that privileged-information path: pre-query cost depended only on visible query text, and the final 40% target partition remained untouched until preregistration.

Held-out confirmation: **1,197 missions / 753 target clusters**.

At B=12:

- recursive vs matched immediate: **+3.112 pp**, CI **[+2.292,+3.951]**, 8/8 buckets positive;
- recursive vs visible greedy: **+3.109 pp**, CI **[+2.238,+4.010]**, 8/8 positive.

K=8 was within **0.551 pp** of full; K=4 was substantially worse in this construction. At B=20, the main policies were essentially saturated near oracle.

**Theory update:** Scheduling/downstream-value advantage survived grounded natural-language transfer when budget was binding, and disappeared as budget became non-binding.

---

## 11. Analytic synthesis

Once the same structure recurred across fixed links, recursive synthetic queries, async scheduling, and grounded language, broad heuristic search had diminishing value.

For one unresolved item, the key comparison is now written directly as:

`resolve now` versus `defer and continue reading`.

Working stopping rule:

`ASK(q) iff E[value(resolve now)] > E[value(defer)]`.

For a fully known recursive query tree with integer costs, the full-information problem is a precedence-constrained tree knapsack solvable by dynamic programming. The relevant theoretical object is a budget-conditioned recursive value curve `V(q,b)`.

**Theory update:** The research target shifted from discovering another heuristic to approximating an analytic oracle under partial observability, bounded memory, latency, and uncertain answers.

---

## 12. AP-T1 — theory-following sanity check

**Question:** Do basic finite-state consequences of the analytic formulation hold in exact small-tree computations?

**Result:** Across 200 random recursive trees AP-T1 verified:

- value monotonicity in budget;
- value monotonicity in frontier capacity;
- full oracle >= immediate greedy;
- policy convergence when budget permits all useful work;
- zero contradictions in 144 enumerated single-unknown threshold checks.

**Boundary:** AP-T1 is a consistency/sanity check, not evidence about humans or real-LM factuality.

---

# Current theory and stopping point

The strongest current statement is no longer a literal K=4 rule. It is:

> **Under a binding information budget, unresolved items can have positive option value. The relevant decision is whether resolving one now has greater expected downstream value than continuing to read and deferring it. Full-information recursive cases can be solved analytically; practical agents must approximate that oracle from limited visible state and bounded resources.**

The broad computational exploration phase is complete enough to stop. Remaining high-value work is theoretical (approximation/threshold results), stochastic-answer extensions, or optional human/interface validation—not another broad parameter sweep.
