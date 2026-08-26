# AP-LM3B — visible-query-cost grounded recursive text pilot

## Status
Development-only pilot. The held-out confirmatory 40% target partition MUST NOT be evaluated until a separate confirmatory preregistration is committed after this pilot.

## Motivation
AP-LM3A established feasibility of grounded natural-language recursive answer trees, but its pilot exposed a design flaw before held-out evaluation: exact query latency was derived from the hidden destination answer-body length and was visible to the policy before the query. That is privileged information.

AP-LM3B removes that leak. Query cost is a deterministic function of the visible anchor string only. Hidden answer length is never used in policy features, eligibility, compression, greedy ranking, or the oracle cost model.

## Construct
- Real Wikispeedia graph, HTML anchors, and plaintext article bodies.
- A query candidate is a visible linked concept.
- Issuing the query reveals that concept's fixed article body as a grounded answer and then exposes linked concepts in that answer as new recursive query candidates.
- Grounded article text is a retrieval-backed answer surrogate, not a generative LM response.
- Hallucination is absent by construction.

## Visibility discipline
Before querying candidate q, the policy may use only:
- anchor semantic similarity to the known target title;
- candidate-title similarity to target title;
- a fixed blend of those two visible scores;
- visible anchor-derived query cost;
- depth;
- remaining budget;
- frontier size and rank;
- the reward of the already-revealed parent answer, if any.

The policy may NOT use:
- candidate answer-body text or embedding;
- candidate answer length;
- candidate children before the candidate is queried;
- candidate true reward;
- future subtree reward.

## Visible query cost
Let w be the whitespace-token count of the visible anchor string selected for the edge. The cost is fixed before any pilot result is observed:

`cost = clip(1 + floor((w - 1) / 3), 1, 4)`.

Thus 1–3 word anchors cost 1, 4–6 cost 2, 7–9 cost 3, and >=10 cost 4. This is a controlled visible prompt-cost proxy; it is not claimed to be measured API latency.

## Tree construction
- 6 initial roots from the source page.
- breadth-first expansion after each hypothetical answer;
- max depth 3;
- max 3 children per queried answer;
- max 18 unique article nodes per task.
- Children are structurally computed for scoring/teacher construction but are exposed to a deployed rollout only after their parent is queried.

## Semantic reward
For a queried node, reward is squared nonnegative cosine similarity between that grounded answer body and the target article body using frozen `sentence-transformers/all-MiniLM-L6-v2` embeddings.

This is a computational semantic-alignment reward, not human comprehension.

## Models and matched teacher ablation
Same visible feature vector, same StandardScaler + Ridge(alpha=10) for both models.

- Immediate teacher: the queried answer's own semantic reward.
- Recursive teacher: exact dynamic-programming value recoverable from that query's revealed subtree under the remaining budget.

Primary development comparison: recursive teacher vs immediate teacher.
Secondary: recursive teacher vs visible greedy score/cost.

## Capacity
Evaluate recursive policy with full frontier and K in {2,4,8}. Also audit the AP-LM2 adaptive top4/top8 compression rule (top4 predicted positive-value mass >= 0.90 => K=4 else K=8).

## Split and samples
Deterministic target-disjoint split using seed 20260912:
- first 40% target titles: fit;
- next 20%: development pilot;
- final 40%: untouched confirmatory reserve.

Pilot sample sizes:
- fit: up to 700 missions;
- pilot: up to 400 missions;
- budgets: 12 and 20.

No confirmatory threshold is declared in this file. Pilot results will be used only to choose a defensible primary budget and freeze confirmatory thresholds in a NEW preregistration before the final 40% partition is evaluated.

## Development success signal
The pilot is considered promising enough to preregister a fresh confirmation if at least one budget shows:
1. recursive-vs-immediate mean > 0;
2. target-cluster bootstrap CI lower bound > 0;
3. recursive-vs-greedy mean > 0;
4. at least K=8 is within 1 pp of full frontier.

This is a development gate, not a scientific PASS claim.

## Boundaries
- retrieval-backed grounded answers, not generated LM responses;
- no hallucination;
- visible prompt-cost proxy, not wall-clock latency;
- semantic target alignment, not human comprehension;
- pilot and confirm partitions are target-disjoint;
- AP-LM3A pilot is diagnostic only because of the hidden-answer-length latency leak.
