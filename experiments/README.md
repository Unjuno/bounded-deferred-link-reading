# Experiment index

The experiment directory preserves both headline experiments and negative/diagnostic phases that materially changed the theory.

## Current core

- `ap_t1_analytic_sanity.py` — exact finite-state sanity checks for the analytic model.
- `ap_lm1_recursive_query_frontier.py` — recursive synthetic query frontier.
- `ap_lm2_async_slack.py` — asynchronous/slack-aware recursive scheduling.
- `ap_lm3b_confirm.py` — held-out grounded natural-language confirmation.

## Real-data bridge and observability studies

- `ap_rs4_article_body_semantics.py` — article-body semantic progress and human BACK diagnostics.
- `ap_rs5_real_anchor_context_policy.py` — real anchor/context causal transfer gate.
- `ap_rs6_visible_only_policy.py` through `ap_rs15_sidebar_prefetch_k1.py` — visible-policy, memory, oracle-capacity, trigger, and observability diagnostics.

## Why negative experiments remain

Several experiments failed their preregistered criteria. They are intentionally retained because they ruled out stronger claims, exposed implementation/information leaks, or motivated the shift from heuristic search to an analytic oracle/observability formulation.

## Running experiments

See the repository-level `REPRODUCIBILITY.md`. Raw external datasets should remain outside version control; generated downloads/artifacts are covered by `.gitignore`.
