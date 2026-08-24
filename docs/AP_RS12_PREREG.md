# AP-RS12 preregistered diagnostic criteria

AP-RS12 is a post-AP-RS8 internal mechanism diagnostic using the same real Wikispeedia universe and visible anchor-only local scorer.

Before observing AP-RS12 outcomes, the following diagnostic rules are fixed:

1. **Material K=4 buffer opportunity:** oracle K=4 improves S@16 by at least +2.0 percentage points versus anchor-only local, and the target-cluster bootstrap 95% CI lower bound is > 0.
2. **K=4 capacity plateau:** oracle K=8 minus oracle K=4 at S@16 is <= +0.5 percentage points.
3. The oracle is one-shot, uses only the immediately previous page's top-K abandoned alternatives, charges two actions for BACK + alternative traversal, and never intervenes when local continuation would already succeed within the remaining budget.
4. K is swept over {1,2,4,8}. The visited-page set remains full, matching AP-RS8, because AP-RS10 showed that cycle-avoidance memory is a separate construct from deferred-alternative memory.

Interpretation is fixed as follows:

- Material K=4 oracle opportunity + near-null AP-RS8 learned gate => value-identification / triggering bottleneck.
- Weak K=4 oracle opportunity => little deferred-option value remains under strong anchor-only local navigation in this goal-directed construct.
- Oracle results are not deployable performance and do not establish human comprehension benefit.
