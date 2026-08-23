# Data, Licensing, and Reproducibility Notice

## Scope of this public repository

This repository is a public release of a computational and hypothesis-generating research program on bounded-memory hypertext reading/navigation.

It intentionally distinguishes:

- **results and analysis authored for this project**, which are released under the repository license unless otherwise stated;
- **third-party datasets and graph resources**, which remain under their original licenses and are not automatically relicensed by this repository.

## External data

The research program has used or investigated Wikipedia-derived resources, including Wikipedia article-network data and public dataset mirrors. To avoid accidental relicensing or redistribution problems, the public repository should prefer:

1. source links and dataset identifiers;
2. deterministic download/preprocessing instructions;
3. hashes or metadata for locally generated intermediates;
4. small project-authored result summaries rather than copied third-party corpora.

Do not commit a third-party dataset simply because it was available during an experiment. Verify its original license first.

## Reproducibility status

### Reproduced in frozen independent test blocks

The compact spectral-bridge policy was frozen after selection and independently replicated in AP-S43 on 12 new seeds × 500 tasks. The resource-saving memory variant was then tested in AP-S49 on another 12 new seeds × 500 tasks.

Selected machine-readable results are available in `results/`.

### Not yet externally validated

The following are explicit missing gates:

- 400+ independent tasks using real anchor text + containing context semantics with appropriate source/page clustering;
- direct human experiments on comprehension, delayed retention, navigation behavior, reading time, and cognitive load.

Accordingly, statements such as "four links is the human working-memory optimum" or "this is the optimal human hyperlink-reading strategy" are not supported by the current evidence.

## Known implementation corrections

The research log includes negative findings and corrected implementation mistakes. In particular, an off-by-one error in an early bounded-history-window implementation was identified, the affected result was discarded, and the experiment was rerun on fresh seeds with the corrected definition.

This repository treats such corrections as part of the scientific record rather than hiding them.

## Recommended citation language

When describing the present result, use wording similar to:

> In a Wikipedia-derived spectral/non-distance navigation bridge, a short-lived buffer of approximately four alternatives from the immediately previous decision point, combined with a compact one-shot counterfactual trajectory-utility rule, produced a reproducible early-budget gain. Human reading optimality remains untested.
