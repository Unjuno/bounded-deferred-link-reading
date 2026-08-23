# Research Summary

## Research question

The project asks whether an unknown hypertext can be read or navigated efficiently under tight memory and action budgets, and what minimal state is actually needed.

The research program deliberately separates two constructs:

- **Reading / information acquisition:** coverage of latent core information under a reading budget.
- **Goal-directed navigation:** reaching an explicit target under a navigation budget, used later as a mechanism and stress-test environment.

Navigation results are therefore not automatically claims about human comprehension.

## Evolution of the theory

### Stage 1 — Local semantic scoring for information acquisition

Early experiments supported a compact additive representation using current-text relevance, anchor/link relevance, containing-sentence or paragraph relevance, and limited novelty. In the reading/information-acquisition construct, **containing context materially improved link-value estimation**. Larger recurrent, nonlinear, and global-memory models often improved intermediate predictors without improving the end-to-end objective.

### Stage 2 — Immediate-parent runner-up memory

Under a controlled distance-like cue, retaining the immediate-parent runner-up was useful. Direct counterfactual utility calibration also replicated under controlled hidden-noise mixtures. This motivated, but did not ultimately support universally, a one-scalar-history hypothesis.

### Stage 3 — Cue-universality failure

Under a spectral/non-distance bridge, immediate-parent runner-up memory could materially hurt. Margin recalibration did not rescue it. The theory changed from “one scalar history is sufficient” to **memory representation depends on the observation/value channel**.

### Stage 4 — Frontier oracle and trajectory utility

Richer abandoned-alternative frontiers contained large oracle value, but score-based and myopic value models could not recover it. The key distinction became:

> “this alternative is locally closer/better” is not equivalent to “returning to this alternative and then continuing with the same imperfect policy improves the final outcome.”

The teacher was therefore changed to counterfactual trajectory utility:

\[
Y = S_B^{\mathrm{BACK}} - S_B^{\mathrm{CONTINUE}}.
\]

This changed the deployable effect from negative to positive in the spectral bridge.

### Stage 5 — Bounded-memory compression

The richer-history policy was compressed to the immediately previous decision point, K=4 abandoned alternatives, one discretionary reconsideration, and a 3-feature Ridge model. AP-S43 froze this policy and replicated it on 12 new seeds × 500 tasks = 6,000 tasks:

- S@16: **+3.9167 pp**, 95% CI **[+3.133,+4.683] pp**, 12/12 seeds positive.
- S@32: **+1.1500 pp**, 95% CI **[+0.250,+2.067] pp**, 9/12 seeds positive.

K=7 or K=9 did not materially improve over K=4. Multi-page delayed retention was not supported. A resource-saving adaptive cap reduced candidate memory by about 20.9% at a cost of about 0.4 pp.

### Stage 6 — Real human Wikispeedia semantics

The next phase moved from surrogate policy environments to real human Wikispeedia paths.

AP-RS3 showed strong target-directed progress using complete article-title embeddings. AP-RS4 upgraded the outcome space to **actual Wikispeedia article-body semantics** and excluded the terminal target transition.

AP-RS4 findings:

- 28,182 successful human paths showed strong positive nonterminal body-semantic progress; **96.36%** of paths had positive mean progress.
- In 4,301 first eligible nonterminal BACK episodes, returning to the ancestor itself moved away from the target on average, but the replacement branch moved **+0.0291** in body-semantic similarity relative to the abandoned branch; the preregistered average-effect criterion passed.
- The episode-level effect is heterogeneous; the raw positive fraction was only **51.08%**.

Exploratory AP-RS4c/4e analyses sharpened the mechanism:

- **85.1%** of first eligible BACK episodes were one-step returns; **96.0%** were within two steps.
- One-step replacement correction averaged **+0.0348**, whereas two-or-more-step correction averaged approximately zero/negative; target-matched sensitivity retained a positive one-step advantage.
- When the immediately preceding forward click was itself a body-semantic regression, the subsequent replacement correction was very large (**+0.0927** on average). When the preceding click had already improved target similarity, replacement-vs-abandoned was slightly negative on average.

These are observational human-navigation results. They support a **short-range branch-correction pattern**, not a human working-memory mechanism.

### Stage 7 — First causal real anchor/context gate: AP-RS5

AP-RS5 used real Wikispeedia HTML, real hyperlink topology, and mission pairs from the human task distribution. Candidate actions were scored using a frozen MiniLM mixture of **0.5 anchor text + 0.5 containing paragraph**. Fit/tune/test targets were disjoint, and test contained 1,200 missions.

The preregistered bounded-deferred policy **FAILED**:

- equal anchor/context local S@16: **0.7425**
- bounded policy S@16: **0.7475**
- delta: **+0.50 pp**, target-cluster CI approximately **[-0.43,+1.45] pp**
- equal local S@32: **0.8275**
- bounded policy S@32: **0.8325**
- delta: **+0.50 pp**

The +2 pp early-budget threshold, positive-CI threshold, and 6/8 target-bucket threshold all failed. Long-horizon safety passed.

A major construct-specific scorer result also emerged:

- **anchor-only local:** S@16 **0.7908**, S@32 **0.8458**
- equal anchor/context local: S@16 0.7425, S@32 0.8275
- context-only local: S@16 0.3808, S@32 0.5333

Thus, on **explicit target navigation**, containing paragraph context diluted a very strong anchor-target signal. This does **not** overturn the earlier reading/information-acquisition result that context is useful; it strengthens the need to keep the two constructs separate.

Implementation audit: one AP-RS5 gate feature (`candidate_outdegree`) uses metadata from an unvisited candidate page, so RS5 is best described as **graph-assisted real-semantic** rather than strictly human-visible. AP-RS6 was specified before reading the RS5 outcome to remove that feature.

## Current scientific position

### Supported

- In reading/information-acquisition experiments, anchor semantics are more useful when combined with containing context.
- In explicit target navigation, the best local semantic channel can differ; AP-RS5 strongly favored anchor-only scoring.
- One abandoned alternative is not universally sufficient.
- In the spectral bridge, a short-lived top-4 buffer plus one-shot trajectory utility is a reproducible compact policy.
- Real successful human Wikispeedia paths show strong body-semantic target progress.
- Human BACK behavior is predominantly short-range, and short-range BACK followed by branch replacement is associated with semantic correction.
- A generic real-semantic bounded-deferred policy using the frozen equal anchor/context scorer did **not** meet the causal transfer criterion in AP-RS5.

### Not established

- That K=4 is a human working-memory constant.
- That bounded deferred links improve human comprehension or retention.
- That the AP-S43 compact policy transfers to real visible semantics.
- That context should always be added to anchor text for explicit target navigation.
- That observed human BACK correction is caused by a bounded option buffer.
- That this is an optimal human hyperlink-reading strategy.

## Active external-validity tests

- **AP-RS6:** strictly visible-only equal anchor/context gate; removes unvisited candidate metadata.
- **AP-RS7:** target-matched finished-vs-unfinished human BACK body-semantic correction.
- **AP-RS8:** visible anchor-only local baseline plus bounded one-shot reconsideration; tests whether deferred memory adds value when the local semantic channel is already strong.

## Human and agent-assistance frontier

A later human/agent experiment should separate at least four information regimes: unaided reading, external short-term buffer only, agent reranking of retained links, and agent prefetch/summarization of destination pages. Prefetch changes the information regime and should not be conflated with simple memory offloading.
