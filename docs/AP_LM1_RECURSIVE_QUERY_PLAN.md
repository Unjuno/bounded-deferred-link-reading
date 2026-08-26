# AP-LM1 — Recursive Deferred-Query Frontier

## Purpose

Generalize bounded deferred-link reading from fixed hypertext destinations to LM-like queries whose answers can themselves contain unresolved concepts that generate further queries.

The controlled AP-LM1 experiment deliberately sets hallucination to zero. The first target is the optimization problem created by recursive query generation and response latency, not answer correctness.

## Research program

1. **AP-LM1 — serial recursive-query control (this preregistration).** Perfect answers, answer-generated unknowns, variable serial latency, exact oracle, learned subtree-value ranking, and bounded deferred-frontier capacity sweep.
2. **AP-LM2 — asynchronous latency/slack.** Permit queries to remain in flight while reading continues; optimize effective latency rather than raw latency and compare prefetch/scheduling rules.
3. **AP-LM3 — real LM answers.** Replace the synthetic answer tree with fixed LM-generated explanations and automatically extracted follow-up unknown spans; separately audit correctness/hallucination.
4. **AP-LM4 — human/agent study.** Compare unaided reading, immediate ask-on-unknown, deferred question buffer, and value-ranked query assistance on comprehension/information-acquisition outcomes.

## H — hypotheses

### H1: recursive value ranking

A policy that predicts the **useful-information value of the entire recursive subtree reachable through a query** will outperform a strong global visible-cue/latency greedy policy under a tight latency budget.

Primary confirmatory criterion at budget 16:

- mean oracle-normalized gain >= **+2.0 percentage points**;
- seed-cluster bootstrap 95% CI lower bound > 0;
- >= **10/12** held-out test seeds positive;
- long-budget (32) mean gain >= **-1.0 pp**.

### H2: latency information matters

Under variable query latency, adding latency features to the learned subtree-value model will improve oracle-normalized performance relative to the otherwise matched no-latency model. Confirmatory evidence requires a positive budget-16 seed-cluster CI and nonnegative mean effect at budget 32.

### H3: bounded frontier capacity is not assumed universal

K is swept over **1, 2, 4, 8, 16**. We do **not** preregister that the earlier hypertext K≈4 result transfers. The reported capacity statistic is the smallest K whose mean oracle-normalized performance is within **1 pp** of the full learned frontier at **both** budgets 16 and 32.

This directly tests whether recursive answer-generated unknowns require a larger deferred frontier than fixed-link reading.

## T — task/environment

Each episode is a synthetic base document with six initially unresolved spans. Querying a span:

1. consumes a serial response latency of 1–5 time units;
2. yields positive useful-information reward;
3. reveals 0–3 unresolved spans inside that answer;
4. may therefore create deeper queries up to depth 4.

Branch quality is correlated across generations so that evidence from an answer can inform the expected value of its follow-up questions. The visible cue for a candidate is noisy, so local relevance is imperfect.

### Frozen generator

- roots: 6
- maximum query depth: 4
- parent/child latent-quality correlation: 0.92
- reward: `exp(0.85*z + Normal(0, 0.25))`
- visible cue: `z + Normal(0, 1.45)`
- latency: rounded/clipped `exp(Normal(0.72, 0.45))`, range 1–5
- child count: `min(3, Poisson(1.0 + 0.6*sigmoid(z)))`

## D — policies and data split

### Baselines

- **recursive DFS:** when an answer reveals new unknowns, immediately pursue those nested questions before older alternatives;
- **global visible greedy:** among all currently known candidates, choose the largest visible cue / latency score.

### Learned policy

Training target for each candidate is the exact useful-information value obtainable from that candidate's **recursive subtree** under the remaining budget. The deployable model sees only observable state:

- candidate visible cue;
- depth;
- observed parent-answer reward;
- remaining-budget fraction;
- frontier size and cue rank;
- plus latency and cue/latency features in the latency-aware variant.

Model: standardized Ridge regression, alpha 10, fixed without test tuning.

### Oracle

A clairvoyant dynamic program computes the exact best ancestor-closed subset of recursive queries under the serial latency budget. It knows all latent rewards and latencies and is used only as an upper-bound denominator/teacher source, never as deployable information.

### Split

Development-only pilot simulation was used to choose this experiment family and freeze the above design. It is not part of confirmatory evidence.

Confirmatory settings are frozen before running the held-out seeds:

- fit RNG seed: `26082601`
- fit episodes: 1,200
- held-out test seeds: `910001..910012`
- 300 test episodes per seed
- budgets: 16 and 32
- 10,000 seed-cluster bootstrap replicates

No parameter, threshold, K, generator coefficient, or model feature will be changed after reading held-out results.

## C — primary contrasts

1. learned full latency-aware vs global visible greedy at budget 16;
2. same contrast at budget 32 for safety;
3. learned full latency-aware vs learned full no-latency;
4. K capacity sweep relative to learned full frontier;
5. recursive DFS as the direct analogue of 'follow every new unknown inside the answer immediately'.

## U — interpretation / uncertainty boundaries

- AP-LM1 does **not** test hallucination; answer correctness is fixed perfect.
- AP-LM1 models serial latency. Query/read overlap and slack are reserved for AP-LM2.
- Synthetic reward is useful-information acquisition, not a direct human comprehension measure.
- A result that K>4 is needed would not contradict AP-S43; recursive query generation creates a different frontier-growth process.
- If learned full beats greedy but small K fails, the bottleneck is frontier capacity/compression rather than value estimation.
- If latency-aware fails to beat no-latency, latency may be adequately absorbed by remaining-budget dynamics or the current latency variation may be too weak.

## Decision branching

- **H1 PASS, small-K close to full:** proceed to AP-LM2 with the compact frontier fixed.
- **H1 PASS, K>=8 needed:** AP-LM2 should test adaptive frontier capacity as well as asynchronous scheduling.
- **H1 FAIL but oracle gap large:** improve observable query-value features / teacher before adding real LM noise.
- **Oracle gap small:** the generator/task is not sufficiently discriminative and must be redesigned on fresh seeds, without recycling this test set.
