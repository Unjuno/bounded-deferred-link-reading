# Documentation

This directory separates the **current research interpretation** from **experiment-specific records**.

## Start here

Read these in roughly this order:

1. [`RESEARCH_SUMMARY.md`](RESEARCH_SUMMARY.md) — the research trajectory, decisive findings, negative results, and claim boundaries.
2. [`ANALYTIC_THEORY.md`](ANALYTIC_THEORY.md) — the current Bellman / tree-knapsack / optimal-stopping formulation.
3. [`EXPERIMENT_TIMELINE.md`](EXPERIMENT_TIMELINE.md) — a selective chronology of the experiments that materially changed the theory.
4. [`INTERFACE_PROPOSAL.md`](INTERFACE_PROPOSAL.md) — the reading-interface direction derived from the theory; this remains an unvalidated design proposal.
5. [`HUMAN_READING_HYPOTHESIS.md`](HUMAN_READING_HYPOTHESIS.md) — cautious, testable translation from computational results to human reading.

## Experiment-specific records

Plans and preregistrations are preserved because they document what was fixed before evaluation and how the experiment sequence evolved. The most important confirmatory preregistration is `PREREG_AP_LM3B_CONFIRM.md`; AP-RS14 and AP-RS15 preregistrations are also retained because the corresponding source files explicitly reference those paths.

Earlier short phase summaries and the RS5 audit are collected in [`archive/`](archive/) so they remain available without competing with the current synthesis documents.

## Code, results, and reproduction

- [`../experiments/`](../experiments/) — executable experiment code and an experiment index.
- [`../results/`](../results/) — selected machine-readable results and a result index.
- [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) — environment and rerun instructions.
- [`../NOTICE.md`](../NOTICE.md) — external-resource provenance, licensing boundaries, and audit corrections.

## Status labels

When reading historical files, distinguish:

- **confirmatory / preregistered** — evaluated after a fixed plan;
- **diagnostic / exploratory** — useful for mechanism discovery or debugging, not a fresh confirmation;
- **superseded** — preserved for provenance after a later implementation or visibility correction;
- **interface hypothesis** — derived design implication, not direct human-comprehension evidence.

The repository-level `README.md` is the shortest current statement of what is and is not supported.
