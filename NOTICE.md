# Data, Licensing, Reproducibility, and Claim Notice

## Scope

This repository is a public computational research record on deferred reading/querying under limited time, attention, memory, and observability.

It contains:

- project-authored experiment code, preregistrations, analyses, and summaries;
- selected small machine-readable result files;
- references to third-party datasets and model weights downloaded at runtime.

It does **not** redistribute the large third-party corpora or model weights used by the later real-data experiments.

## External resources used by the code

The later Wikispeedia experiments reference public upstream resources including:

- Stanford SNAP Wikispeedia paths/graph archive;
- Stanford SNAP Wikispeedia article HTML archive;
- Stanford SNAP Wikispeedia article plaintext archive;
- a pinned public path file from the EPFL ADA Wikispeedia project used by AP-RS4;
- `sentence-transformers/all-MiniLM-L6-v2` model weights for semantic embeddings.

These resources remain under their own upstream terms. Their presence as download URLs in this repository does not relicense them under MIT.

Before redistributing an upstream corpus, model, or derived resource, check the corresponding source license/terms directly.

## Repository license

Unless a file states otherwise, project-authored code and text in this repository are released under the MIT License in `LICENSE`.

The MIT license applies only to material the repository has the right to license. It does not override third-party dataset/model terms.

## Reproducibility status

The repository preserves fixed seeds, experiment scripts, preregistration documents for major confirmatory phases, and selected result outputs.

Headline preserved results include:

- AP-S43 compact spectral-bridge replication;
- AP-S49 adaptive memory/resource trade-off;
- real Wikispeedia observational and causal/diagnostic phases;
- AP-LM1 recursive query scheduling;
- AP-LM2 asynchronous/slack-aware scheduling;
- AP-LM3B held-out grounded natural-language confirmation;
- AP-T1 analytic sanity checks.

See `REPRODUCIBILITY.md` for commands and environment notes.

Historical GitHub Actions workflows used during active experimentation are intentionally not active in the public release tree. Some of them had repository-write permissions to commit result artifacts. Their historical definitions remain available through git history; reproduction should now be explicit rather than automatically triggered by changes to the public repository.

## Important implementation/audit corrections

The research record intentionally preserves corrections and negative results.

Notable examples:

- an early bounded-history-window off-by-one implementation was discarded and rerun on fresh seeds;
- AP-RS5 included `candidate_outdegree`, which uses graph metadata from an unvisited candidate, so RS5 is described as graph-assisted rather than strictly visible-only;
- AP-LM3A used hidden answer length to determine pre-query latency and is therefore treated as development provenance only;
- AP-LM3B removed that privileged-information path and ran a fresh held-out confirmation using visible query text for pre-query cost.

These corrections are part of the scientific interpretation rather than hidden implementation details.

## Claim boundary

The repository supports computational claims about bounded control, deferred options, recursive query scheduling, observability, and resource/performance trade-offs in the tested environments.

It does **not** establish:

- an optimal human reading strategy;
- a universal human working-memory capacity such as K=4 or K=8;
- a causal cognitive explanation of observed human Wikispeedia BACK behavior;
- improved human comprehension/retention from the proposed reading interface;
- robustness to hallucinating real generative-LM answers.

Navigation success, semantic similarity, and synthetic recursive-query reward are proxy/task outcomes, not direct measurements of human comprehension.

## Recommended high-level description

A cautious summary is:

> Across controlled hypertext, real Wikispeedia diagnostics, synthetic recursive queries, and grounded natural-language query trees, the project found recurring value in deferring some uncertainties and evaluating actions by downstream/budget-conditioned value rather than immediate local relevance alone. The resulting full-information problem can be formulated analytically; the remaining difficulty is approximating that oracle under partial observability, bounded memory, latency, and uncertain answers. Human-reading and interface benefits remain hypotheses for external validation.
