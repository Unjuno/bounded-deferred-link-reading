# Preregistration — AP-LM3B grounded natural-language confirmation

Committed after the AP-LM3B development pilot and BEFORE evaluating the reserved final 40% target partition.

## Frozen scientific question
Does the recursive-subtree-value teacher improve finite-budget grounded natural-language query scheduling when all pre-query features and costs are genuinely visible at query time?

## Data and split
Use the same Wikispeedia graph, HTML anchors, plaintext bodies, frozen MiniLM model, and deterministic target ordering from AP-LM3B development seed 20260912.

Target-disjoint partitions are unchanged:
- 40% fit;
- 20% development pilot (already observed);
- final 40% confirmatory reserve (not previously evaluated).

Confirmatory sample: up to 1200 missions whose target is in the final 40% reserve.

## Frozen construct and visibility
Unchanged from AP-LM3B pilot:
- 6 roots;
- max depth 3;
- max 3 children per answer;
- max 18 unique article nodes;
- grounded answer = fixed destination article body;
- reward = squared nonnegative body cosine to target body;
- visible score = 0.80 anchor-to-target cosine + 0.20 destination-title-to-target cosine;
- query cost = clip(1 + floor((visible_anchor_words - 1)/3), 1, 4);
- candidate answer text, answer length, true reward, and children are hidden until the candidate is queried;
- same StandardScaler + Ridge(alpha=10) model family;
- same matched features for immediate and recursive teachers.

The fit set may be regenerated deterministically from the frozen fit partition. No confirmatory labels are used for fitting or calibration.

## Implementation correction frozen before confirmatory run
For bounded-K capacity audits only, candidates whose visible query cost exceeds the current remaining budget are removed before compression. Remaining budget never increases, so such candidates can never become feasible later. This correction prevents permanently infeasible candidates from consuming bounded frontier slots.

This change does NOT alter the full-frontier primary recursive, immediate, greedy, or DFS policies.

## Primary budget
B = 12.

The development pilot showed meaningful policy separation at B=12. At B=20 all full-frontier policies were essentially oracle-saturated, so B=20 is a safety/saturation check rather than a second primary endpoint.

## Primary comparison H1
`recursive full frontier - matched immediate-value full frontier` at B=12, measured in oracle-normalized semantic utility.

H1 PASS requires all of:
1. at least 800 evaluated confirmatory missions;
2. mean gain >= +1.0 percentage point;
3. target-cluster bootstrap 95% CI lower bound > 0;
4. at least 6 of 8 deterministic target buckets have positive mean gain.

## Secondary external baseline H2
`recursive full frontier - visible greedy` at B=12.

H2 is supported if:
1. mean gain > 0;
2. target-cluster bootstrap 95% CI lower bound > 0;
3. at least 5 of 8 target buckets are positive.

H2 is secondary and is not required for the primary AP-LM3B PASS if H1 passes, but it is required for the stronger claim that recursive value improves over the visible local heuristic as well as over the matched immediate teacher.

## Safety / saturation check
At B=20, require recursive full frontier not to be meaningfully worse than immediate full frontier:
- mean recursive-minus-immediate >= -0.25 pp;
- target-cluster CI lower >= -0.50 pp.

This is a safety criterion, not an expected positive-effect criterion, because the development pilot was saturated near oracle utility at B=20.

## Capacity audit
At B=12, evaluate K in {2,4,8} after dropping permanently infeasible candidates before compression.

Resource-success criterion:
- K=8 is within 1.0 pp of full recursive policy at B=12.

K=4 is exploratory. The AP-LM2 adaptive top4/top8 rule is also exploratory in this natural-language transfer; it is not part of AP-LM3B scientific PASS because the development pilot showed a material performance cost.

## Overall decision
AP-LM3B scientific PASS iff H1 PASS and the B=20 safety criterion passes.

Report H2 and capacity independently.

## Bootstrap and buckets
- 10,000 target-cluster bootstrap replicates for confirmatory CIs.
- Eight deterministic target buckets from SHA-256(target) modulo 8.
- No threshold, feature, teacher, split, budget, or K rule may be changed after the first confirmatory result is observed.

## Claim boundaries
A PASS supports transfer of bounded recursive downstream-value scheduling from synthetic query trees to retrieval-backed natural-language answer trees under visible query-side cost.

A PASS does NOT establish:
- performance with a generative LM;
- robustness to hallucination;
- measured wall-clock API latency;
- human comprehension or metacognitive unknown detection.
