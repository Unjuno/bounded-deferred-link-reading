# AP-RS15 preregistration — sidebar-agent K=1 destination prefetch

## Motivation

AP-RS13 found a large K=1 oracle opportunity but a harmful broad visible-only trigger. AP-RS14 then used a conservative four-fold calibration rule and found **no stable visible-only intervention threshold**, while the K=1 oracle S@16 ceiling remained about +8 pp on a fresh split. AP-RS15 tests whether the missing variable is information about the **unvisited deferred destination itself**.

## Information-regime change

This phase deliberately changes the information regime to represent a sidebar agent that can prefetch one deferred destination while the reader/agent is on the next page.

The buffer remains K=1: the immediately previous page's rank-2 anchor candidate only.

Local navigation ordering is unchanged: visible anchor MiniLM cosine to the known target title.

The trigger model receives the frozen AP-RS14 visible V0 features plus exactly two prefetch features:

1. MiniLM cosine between the rank-2 deferred destination's **article-body prefix** and the known target title;
2. that prefetched-body similarity minus the already-visited current page's body similarity to the target title.

No graph shortest-path distance, future rollout result, target body, or other oracle information is available to the policy.

## Split

Seed: `20260903`.

Targets are shuffled once and partitioned 40% fit / 20% calibration / 40% test. Mission samples are target-disjoint within the phase:

- fit: up to 1,600 missions;
- calibration: up to 1,000 missions;
- test: up to 2,600 missions.

## Teacher and model

Budget-16 one-intervention counterfactual utility teacher:

`y = success16(BACK to rank-2 then anchor-local) - success16(CONTINUE anchor-local)`.

Fit StandardScaler + Ridge(alpha=1) on fit targets. Ties remain in regression.

Primary feature vector:

- `log1p(origin candidate count)`;
- rank-2 anchor score minus current-page top-1 anchor score;
- step / 16;
- prefetched rank-2 destination body similarity to target title;
- prefetched rank-2 body similarity minus current visited-page body similarity.

A visible-only V0 model is fit on the same fit teacher only as a post-test identifiability comparator; it is not eligible for policy selection.

## Robust sparse threshold selection

Candidate margins:

`[-0.05, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20, 999.0]`.

Calibration targets are assigned to four deterministic folds. A margin is eligible only if:

1. mean intervention rate <= 25%;
2. mean S@16 effect > 0;
3. S@16 effect > 0 in at least 3 of 4 target folds;
4. S@32 effect >= -0.5 pp in every target fold.

Among eligible margins select the largest mean calibration S@16 effect; ties: lower intervention rate, then higher margin. If none are eligible, select 999.0 (no intervention).

## Confirmatory PASS rule

All required:

1. test n >= 400;
2. S@16 gain >= +2.0 pp versus anchor-local;
3. target-cluster bootstrap 95% CI lower bound for S@16 > 0;
4. >=6 of 8 deterministic target buckets positive at S@16;
5. mean S@32 effect >= -0.5 pp;
6. target-cluster bootstrap 95% CI lower bound for S@32 >= -1.0 pp.

Otherwise FAIL.

## Secondary diagnostics after threshold freeze

- held-out non-tie sign AUC of prefetch model and visible-only V0 comparator;
- actually triggered-state oracle teacher help/tie/harm fractions;
- same-test never-harm K=1 oracle ceiling;
- recovered fraction of oracle S@16 opportunity.

## Interpretation boundary

A PASS would show that **one-candidate destination preview can make bounded reconsideration deployable in this simulated navigation construct**, whereas visible-only state did not. It would support a sidebar-agent role as an information-acquisition aid, not prove human comprehension benefit. A FAIL with a persistent large oracle ceiling would indicate that body similarity preview alone is still insufficient to identify beneficial reconsideration states.