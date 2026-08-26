# Analytic Theory of Deferred Querying

## Research phase

The experimental program has now identified a stable policy pattern. The main task is no longer broad heuristic search. It is to formulate the problem as a finite-horizon optimization problem, solve a full-information oracle exactly where possible, and study how closely bounded and partially observed agents can approximate it.

Working principle:

> **Resolve an uncertainty only when the expected value of resolving it now exceeds the expected value of deferring it.**

This unifies hyperlink traversal, document reading, and recursive LM querying.

## State and actions

At time `t` let

- `x_t`: current reading/context state,
- `F_t`: unresolved-query frontier,
- `B_t`: remaining budget,
- optionally `P_t`: in-flight asynchronous queries.

Actions are primarily `READ` and `ASK(q)`; asynchronous systems may also have `CONSUME(q)` when an answer arrives.

A candidate unknown is therefore an **option**, not an immediate command to query.

## Optimal stopping

For unresolved item `q` in state `s`, define

- `A(q,s)`: expected value of resolving `q` now,
- `W(q,s)`: expected value of deferring `q` and continuing to read.

The ideal stopping rule is

`ASK(q) iff A(q,s) > W(q,s)`.

`A(q,s)` includes downstream loss avoided, direct information gain, recursive descendants exposed, urgency, and query cost. `W(q,s)` includes contextual self-resolution, the value of reading other material first, improved future query formulation, and the cost of carrying the unresolved dependency.

`DEFER` is not `DISCARD`; it preserves option value.

## Exact finite-tree solution

Suppose a recursive query tree is fully known. Each node `q` has cost `c_q`, reward `r_q`, and children `Ch(q)`. Descendants become available only after the parent is queried.

Define `V_q(b)` as the maximum reward obtainable from the subtree rooted at `q` with remaining integer budget `b`.

If `b < c_q`, `V_q(b)=0`. Otherwise,

`V_q(b) = max(0, r_q + max_{sum b_j <= b-c_q} sum_{j in Ch(q)} V_j(b_j))`.

Multiple roots are combined with the same budget-allocation convolution. This is a precedence-constrained tree knapsack problem and is exactly solvable by dynamic programming.

The correct theoretical object is therefore not one scalar `V(q)` but the **budget-conditioned recursive value curve** `V(q,b)`.

## Online Bellman form

For current frontier `F` and remaining budget `b`,

`J(F,b) = max(0, max_{q in F, c_q <= b} [r_q + J((F \ {q}) union Ch(q), b-c_q)])`.

This formalizes the empirical distinction between local/immediate value and downstream trajectory/subtree value.

## Why reading first can be optimal

`READ` is itself an information-acquisition action. More context can

- resolve an unknown for free,
- reveal that it is irrelevant,
- change its downstream dependency,
- improve a future query.

Thus an immediate-query policy can be dominated even when the query would eventually be useful.

A practical approximation is

`I(q) ~= unresolvedness * downstream_dependency * (1-contextual_resolution_probability) + recursive_gain - query_cost`.

This is a heuristic approximation, not a universal exact index theorem.

## Bounded frontier

If only `K` unresolved items can be retained, impose `|F_t| <= K`. When new children overflow the frontier, the policy must jointly choose what to query and what to retain/evict.

Let `V*_K(B)` be optimal value under capacity `K`. Then

`V*_K(B) <= V*_{K+1}(B) <= V*_infinity(B)`.

Define

`K_epsilon = min {K : V*_K(B) >= V*_infinity(B) - epsilon}`.

This is the right interpretation of experimental K sweeps: environment-specific memory sufficient to approximate the full-frontier oracle, not a universal human-memory constant.

## Async latency

If query latency is `tau_q` and reading can continue for slack `S_q`, a useful first-order quantity is

`effective_blocking(q)=max(0, tau_q-S_q)`.

Exact optimization adds pending queries and completion times to the state, yielding a finite-horizon semi-Markov decision problem.

## Stochastic / unreliable answers

For a generative LM,

`(answer, children) ~ P(. | q, context)`.

The optimal value becomes an expectation over answer usefulness, correctness, and descendants. With latent reliability the natural generalization is a Bayesian MDP/POMDP.

A practical approximation is

`Q(q) ~= P(useful_and_correct | q,s) * E[V_subtree(q,b) | visible state] - latency_cost - attention_cost - reliability_risk`.

## What the experiments established

### Controlled fixed-link bridge

AP-S43: a short-lived top-4 alternative buffer plus trajectory utility improved the spectral bridge.

- S@16: +3.9167 pp, 95% CI [+3.133,+4.683], 12/12 seeds positive.
- S@32: +1.1500 pp, 95% CI [+0.250,+2.067].

The lesson is bounded deferred options plus downstream value prediction, not the literal number four.

### Real Wikispeedia

Visible causal transfer was much harder despite large oracle opportunity. This exposed an **observability bottleneck**: the full-information optimal action can exist without enough visible information to identify it reliably.

This motivates separating:

1. full-information optimal control, from
2. statistical estimation of the value function from visible state.

### AP-LM1

Recursive subtree-value prediction beat visible greedy:

- B=16: +4.765 pp, CI [+4.449,+5.078], 12/12 positive.
- B=32: +7.034 pp, CI [+6.531,+7.546].

Latency awareness also helped, and K=8 was the smallest tested capacity within 1 pp of full at both budgets.

### AP-LM2

In an asynchronous/slack-aware synthetic setting:

- slack awareness: about +2.886 pp;
- recursive-subtree teacher vs matched immediate teacher: about +1.995 pp;
- K=4 was sufficient in that construction;
- adaptive capacity reduced retained state by about 18% with little performance loss.

### AP-LM3B

Grounded natural-language recursive answers confirmed the same mechanism on 1,197 held-out missions / 753 target clusters:

- recursive vs matched immediate at B=12: +3.112 pp, CI [+2.292,+3.951], 8/8 buckets positive;
- recursive vs visible greedy: +3.109 pp, CI [+2.238,+4.010], 8/8 positive;
- K=8 was within 0.551 pp of full;
- K=4 was 17.23 pp below full;
- at B=20 the full policies were essentially saturated near oracle.

This supports the prediction that policy differences shrink when budget is large enough to resolve almost everything.

## Current interpretation

The research problem has shifted from **discovering a heuristic** to **approximating an analytic oracle under partial observability, bounded memory, latency, and stochastic answers**.

Operationally:

1. keep reading instead of resolving every unknown immediately;
2. retain a bounded unresolved frontier;
3. update importance as context accumulates;
4. prioritize budget-conditioned downstream value rather than local relevance alone;
5. query only when the best resolution action dominates continued reading.

## Next analytic tasks

1. Prove monotonicity/threshold results for the single-unknown stopping problem.
2. Characterize conditions under which the tree DP admits greedy/index approximations.
3. Test submodularity/adaptive-submodularity-like conditions for approximation guarantees.
4. Bound `K_epsilon` as a function of branching, budget, and value concentration.
5. Extend the Bellman model to async pending queries and stochastic answer quality.

These are now more valuable than another broad sweep of ad hoc policy variants.
