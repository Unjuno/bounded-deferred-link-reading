# Reproducibility

This repository preserves the code, preregistrations, and selected machine-readable results for the computational research program. Raw third-party corpora and model weights are **not** redistributed.

## Environment

The later experiments were run with Python 3.12. A practical local environment is:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` gives compatible dependency ranges rather than a bit-for-bit historical lock file. Exact numerical reproduction can therefore depend on platform, BLAS, PyTorch, and model-library versions. The fixed seeds and original experiment scripts are preserved.

## Fast analytic sanity check

The smallest self-contained check is AP-T1:

```bash
python experiments/ap_t1_analytic_sanity.py
```

It writes `results/AP_T1_ANALYTIC_SANITY.json` and checks consequences of the analytic finite-state model. It is a consistency check, not a human-comprehension experiment.

## Synthetic recursive-query experiments

AP-LM1:

```bash
python experiments/ap_lm1_recursive_query_frontier.py
```

AP-LM2:

```bash
python experiments/ap_lm2_async_slack.py
```

These use synthetic recursive trees with fixed seeds and do not require external corpora.

## Grounded natural-language confirmation

AP-LM3B uses real Wikispeedia graph/HTML/plaintext resources and the `sentence-transformers/all-MiniLM-L6-v2` embedding model:

```bash
PYTHONPATH=. python experiments/ap_lm3b_confirm.py
```

The script downloads external data/model resources at runtime. It can be substantially slower than AP-T1/LM1/LM2 and requires network access and local storage.

The confirmatory result committed to this repository is:

- `results/AP_LM3B_CONFIRM_RESULTS.json`

Its preregistration is:

- `docs/PREREG_AP_LM3B_CONFIRM.md`

## Real Wikispeedia analyses

The `experiments/ap_rs*.py` scripts cover the real-data validation and diagnostics. Important entry points include:

- `ap_rs4_article_body_semantics.py` — article-body semantic progress and human BACK diagnostics;
- `ap_rs5_real_anchor_context_policy.py` — first causal real anchor/context policy gate;
- `ap_rs13_k1_trigger_semantic_state.py` — visible semantic-state K=1 trigger test;
- `ap_rs15_sidebar_prefetch_k1.py` — body-prefetch observability diagnostic.

These scripts download Wikispeedia resources at runtime. See `NOTICE.md` for data provenance and claim boundaries.

## Result files

`results/` contains selected result JSON/Markdown outputs. Not every exploratory intermediate artifact is committed. See `results/README.md` for the headline files and `docs/EXPERIMENT_TIMELINE.md` for the role of each decisive experiment.

## Historical GitHub Actions

During active experimentation, GitHub Actions workflows were used to launch monitored/preregistered runs and, in several early phases, to commit result artifacts back to the repository. Those workflows are preserved in git history but are intentionally **not active in the public release tree**. Reproduction should be explicit and local/manual rather than triggered automatically by repository changes.

## Data and model downloads

The code references third-party resources, principally:

- Stanford SNAP Wikispeedia path/graph, article HTML, and article plaintext archives;
- a pinned public path file from the EPFL ADA Wikispeedia project used by AP-RS4;
- `sentence-transformers/all-MiniLM-L6-v2` model weights.

These resources are not relicensed by this repository. Check the upstream terms before redistribution.

## Scientific boundary

Reproducing a reported navigation, semantic-similarity, or synthetic-query result does not turn that result into a human-comprehension finding. The repository intentionally separates computational control results, observational human-navigation evidence, and unvalidated interface hypotheses.
