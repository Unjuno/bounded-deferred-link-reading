# AP-RS14 preregistration — robust sparse K=1 trigger

## Question

AP-RS13 showed a large same-distribution K=1 oracle opportunity at S@16 (+8.125 pp), but its visible-state trigger intervened on about 66% of missions and reduced S@16 by 1.625 pp. AP-RS14 tests the narrow hypothesis that this failure is primarily an **over-intervention / threshold-instability** problem rather than lack of K=1 rescue opportunity.

## Frozen construct

- Real Wikispeedia hyperlink graph and mission distribution.
- Candidate ordering uses **visible anchor text only**, MiniLM cosine to the known target title.
- Deferred memory is **K=1**: only the immediately previous page's rank-2 anchor candidate.
- No deferred-destination body prefetch, no destination summary, and no unvisited-page semantic features.
- Trigger features are frozen to AP-RS13 V0 only:
  1. `log1p(origin candidate count)`;
  2. remembered rank-2 anchor score minus current-page top-1 anchor score;
  3. current step / 16.
- One discretionary return at most; BACK + deferred alternative costs two actions.

## Split

Seed: `20260902`.

Targets are shuffled once and partitioned 40% fit / 20% calibration / 40% test. Mission samples are then drawn only from their corresponding target sets.

- fit: up to 1,600 missions;
- calibration: up to 1,000 missions;
- test: up to 2,600 missions.

The test set is not used for model or threshold selection.

## Fit

Fit one StandardScaler + Ridge(alpha=1) model to the budget-16 one-intervention counterfactual utility teacher on fit targets:

`y = success16(BACK to remembered rank-2 then local) - success16(CONTINUE local)`.

Ties are retained in regression. Non-tie sign AUC is diagnostic only and is not a decision criterion.

## Robust sparse threshold selection

Candidate margins are frozen to:

`[-0.05, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20, 999.0]`.

Calibration targets are deterministically assigned to four target folds. A margin is eligible only if all of the following hold on calibration data:

1. mean intervention rate <= 25%;
2. mean S@16 effect > 0;
3. S@16 effect > 0 in at least 3 of 4 target folds;
4. S@32 effect >= -0.5 pp in **every** target fold.

Among eligible margins, choose the one with largest mean calibration S@16 effect; ties are broken by lower intervention rate, then higher margin. If no margin is eligible, select `999.0` (no discretionary intervention).

This rule is fixed before test outcomes are inspected.

## Confirmatory test decision

PASS requires all:

1. test n >= 400;
2. S@16 gain >= +2.0 pp versus anchor-local;
3. target-cluster bootstrap 95% CI lower bound for S@16 > 0;
4. at least 6 of 8 deterministic target buckets have positive S@16 effect;
5. mean S@32 effect >= -0.5 pp;
6. target-cluster bootstrap 95% CI lower bound for S@32 >= -1.0 pp.

Otherwise FAIL.

## Secondary diagnostics, not selection criteria

After the confirmatory policy test is fixed, report:

- held-out test non-tie sign AUC;
- among actually triggered test states, oracle-local budget-16 teacher outcome proportions (help / tie / harm);
- same-test K=1 never-harm oracle ceiling;
- recovery fraction of oracle S@16 opportunity.

These diagnostics must not be used to alter the AP-RS14 test threshold.

## Interpretation boundary

A PASS would support a sparse, conservative visible-only trigger in simulated goal-directed Wikispeedia navigation. It would not establish human comprehension or retention benefit. A FAIL with a still-large K=1 oracle ceiling would strengthen the conclusion that **visible-state value identification, not buffer capacity, is the bottleneck**, motivating a separate sidebar-agent information-regime test with destination prefetch.