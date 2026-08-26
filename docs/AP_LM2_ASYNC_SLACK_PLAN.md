# AP-LM2 — Asynchronous Slack-Aware Recursive Query Scheduling

## Purpose

Test whether the AP-LM1 recursive-query result survives a more realistic asynchronous setting in which reading continues while a single LM query is in flight. Unknown spans are released as reading progresses; answers can reveal further unknowns; each answer has a task-relevant need time, so raw response latency and effective latency are distinct.

The controlled environment still fixes hallucination to zero. AP-LM2 isolates scheduling, recursive value estimation, and bounded query-frontier memory.

## Final preregistration status

This file was finalized **before any held-out seed 920001..920012 was run**. Development-only seeds 280001..280005 were used to debug and freeze the environment. A development implementation audit found that an early capped-tree generator could exhaust the node cap before creating all six roots; the final generator therefore creates all roots first and expands descendants second. The held-out set has not been used for any design choice.

## H — preregistered hypotheses

### H1: slack-aware scheduling

A recursive branch-value policy given a noisy observable slack/dependency cue will outperform the otherwise matched recursive branch-value policy that sees raw latency but not slack.

Primary confirmatory criterion at horizon 24:
- mean oracle-normalized gain >= **+1.0 pp**;
- seed-cluster bootstrap 95% CI lower bound > 0;
- >= **9/12** held-out seeds positive;
- horizon-40 mean gain >= **0.0 pp**.

The +1 pp threshold is an incremental-feature criterion: AP-LM2 compares two otherwise matched learned recursive policies rather than a learned policy against a simple greedy baseline.

### H2: recursive-value teacher vs immediate-value teacher

With the deployable feature set held fixed, training on recursive branch value rather than immediate answer utility will improve tight-horizon performance.

Criterion at horizon 24:
- mean oracle-normalized gain >= **+1.0 pp**;
- seed-cluster bootstrap 95% CI lower bound > 0;
- >= **9/12** held-out seeds positive;
- horizon-40 mean gain >= **-1.0 pp**.

This is the matched teacher ablation requested after AP-LM1: same slack-aware observable features, same model class, different target only.

### H3: frontier capacity and adaptive compression

K is swept over **2, 4, 8, 16**, with a full-frontier policy as reference. We do not assume AP-LM1's K=8 transfers unchanged.

Capacity statistic: smallest K whose mean oracle-normalized performance is within **1 pp** of full at both horizons.

Adaptive rule is frozen before held-out testing: among the top 8 predicted candidates, keep only top 4 when those four account for at least 90% of the summed positive predicted value; otherwise keep top 8. Adaptive compression is considered resource-successful if:
- performance is no more than **0.5 pp** below fixed K=8 at either horizon; and
- mean retained frontier size is at least **10% lower** than fixed K=8.

## T — asynchronous task/environment

One episode is a synthetic reading task with a single LM query server.

- Reading time advances on the wall clock while a query is in flight.
- Root unknowns become observable at release times as the base document is read.
- All six roots are generated before descendant expansion, so the node cap cannot remove initial unknowns.
- Only one LM query can be in flight at a time.
- A query occupies the LM server for its latency, but does not stop the reading clock from advancing.
- When the answer completes, its useful-information reward is obtained and its child unknowns become observable.
- Every node has a latent task-relevant need time. Completing after that need time exponentially discounts utility.
- The deployable slack-aware policy sees a **noisy need-time cue**, not the true need time.

Frozen generator:
- horizons: 24 and 40;
- roots: 6;
- maximum nodes: 18;
- maximum recursive depth: 3;
- parent/child latent-quality correlation: 0.90;
- reward: `exp(0.8*z + Normal(0, 0.25))`;
- visible semantic cue: `z + Normal(0, 1.35)`;
- latency: rounded/clipped `exp(Normal(1.0, 0.55))`, range 1–8;
- root release time: integer 0–10;
- root true slack: integer 2–10;
- child need-time increment: integer 1–6;
- observed need-time cue: true need time + `Normal(0, 1.5)`, rounded/clipped;
- child count: `min(3, Poisson(0.35 + 1.45*sigmoid(z)))` until node/depth cap;
- late utility multiplier: `exp(-lateness / 1.0)`.

All deployable policies are non-idling when an eligible query exists. If no query is currently eligible, reading advances to the next root release. The oracle is exact within this same non-idling one-server policy class.

## D — policies, teachers, and split

### Baselines

- **recursive DFS:** prioritize newly exposed deeper answer-unknowns;
- **visible/raw-latency greedy:** cue / latency;
- **visible/effective-latency greedy:** cue divided by `1 + max(0, latency - observed_slack)`.

### Learned policies

Model class for every learned variant: standardized Ridge regression, alpha 10.

**Raw-latency recursive model** sees:
- visible cue;
- latency;
- recursive depth;
- observed parent answer reward;
- remaining horizon;
- current frontier size;
- cue/latency;
- cue rank.

**Slack-aware recursive model** adds:
- observed slack fraction;
- observed effective-lateness ratio;
- predicted completion margin to observed need time.

**Slack-aware immediate model** receives exactly the same features as the slack-aware recursive model. Only its training target changes.

Teacher targets:
- immediate: timely utility of the candidate answer if started now;
- recursive: exact best non-idling utility obtainable from that candidate's answer subtree if the candidate is started now.

### Bounded-frontier implementation

Once a candidate is removed by K-compression, it remains discarded and is not silently reintroduced on the next decision. This was explicitly checked during development because an early pilot implementation re-derived discarded descendants from the queried-parent state.

### Oracle

A clairvoyant memoized scheduler knows true rewards, true need times, latencies, releases, and the full recursive tree. It computes the exact best reward achievable in the frozen non-idling one-server class and is used only as an evaluation denominator.

### Split

Development-only pilot seeds `280001..280005` were used to debug the environment, implementation, and preregistration. They are excluded from confirmatory evidence.

Frozen confirmatory split:
- fit RNG seed: `26082602`;
- fit episodes: **700**;
- held-out test seeds: `920001..920012`;
- **200** test episodes per seed;
- horizons: 24 and 40;
- bootstrap replicates: 10,000.

No generator coefficient, feature, teacher, threshold, K rule, seed, or PASS criterion will be changed after reading held-out results.

## C — primary contrasts

1. slack-aware recursive vs raw-latency recursive at horizon 24 (H1);
2. same contrast at horizon 40 for long-horizon safety;
3. slack-aware recursive vs slack-aware immediate at horizon 24 and 40 (H2);
4. slack-aware recursive vs simple effective-latency greedy;
5. K=2/4/8/16 and adaptive compression vs full recursive slack-aware policy;
6. mean retained frontier size for fixed and adaptive memory policies.

## U — interpretation boundaries

- Hallucination/correctness remains fixed perfect.
- Slack is represented by a noisy synthetic dependency cue; AP-LM2 does not establish that real readers or agents can estimate need time equally well.
- Only one LM query can be in flight; multi-query concurrency is not tested here.
- Policies are non-idling when a candidate exists; strategic deliberate idling is outside this experiment.
- Synthetic timely information utility is not a direct human comprehension measure.
- A lower K than AP-LM1 would not contradict AP-LM1: asynchronous release/deadline structure changes frontier competition.

## Overall decision label

- **PASS:** H1 PASS and H2 PASS.
- **PARTIAL:** H1 PASS and H2 FAIL.
- **FAIL:** H1 FAIL.

## Decision branching

- **H1 PASS + H2 PASS:** proceed to AP-LM3 real fixed LM-answer trees, keeping slack-aware recursive value as the primary policy family.
- **H1 PASS + H2 FAIL:** retain slack-aware scheduling but weaken the claim that recursive-value teaching is necessary; test real LM transfer with both teachers.
- **H1 FAIL:** do not add hallucination yet; first diagnose whether noisy slack observability, scheduler class, or utility definition removes the AP-LM1 latency effect.
- **Adaptive resource success:** carry adaptive frontier compression into AP-LM3; otherwise keep the smallest fixed K within 1 pp of full.
