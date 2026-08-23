# Decisive Experiment Timeline

This is a compact timeline of the experiments that materially changed the theory. It is intentionally selective; many intermediate ablations are omitted.

## A–AG2 — Local reading / information acquisition

**Question:** Can a compact local policy recover core information without large recurrent state?

**Result:** Low-dimensional additive local scoring was robust. Anchor relevance became substantially more useful when combined with the containing sentence/context. Larger recurrent/nonlinear/global-memory variants frequently failed to improve the end-to-end objective.

**Theory update:** Start with context-aware local scoring; do not assume model complexity is necessary.

---

## AH–AO — Small real-Wikipedia semantic pilots

**Question:** Does the anchor+context direction survive contact with real article text?

**Result:** In a very small seven-page pilot, equal anchor+context weighting gave the strongest mean page AUC among the simple tested scorers. The independent sample size was far below the planned external-validity requirement.

**Decision:** UNCERTAIN.

**Theory update:** Real-semantic signal is promising but unvalidated at scale.

---

## AK / AP-S0–S1 — Immediate-parent runner-up under controlled cue

**Question:** Is one remembered abandoned alternative enough history?

**Result:** Under a distance-like controlled cue on Wikipedia-derived topology, immediate-parent runner-up memory improved navigation and often captured most of the benefit of larger frontier memory.

**Theory update:** One-scalar history looked plausible — but only under this observation channel.

---

## AP-S4–S11 — BACK calibration and counterfactual utility

**Question:** Can the agent learn when a return is actually useful?

**Result:** Simple validity classification was not enough. Direct counterfactual utility produced navigation gains; the original strict AP-S6 preregistration failed on an auxiliary AUC threshold, but the navigation effect independently replicated in AP-S11.

**Theory update:** Optimize decision/trajectory utility rather than an intermediate validity label.

---

## AP-S12–S21 — Noise/calibration robustness

**Question:** Is the utility-gated runner a robust single deployment policy?

**Result:** It was robust to several hidden noise/calibration mixtures, but its incremental benefit over the simple runner became small under joint OOD. Reliability-fallback rules did not provide a robust additional gain.

**Theory update:** Additional calibration complexity offers limited practical gain; external validity matters more than further controlled-cue feature engineering.

---

## AP-S22–S24 — Spectral/non-distance bridge

**Question:** Is the one-scalar runner cue-universal?

**Result:** No. Under a spectral/diffusion similarity cue that did not directly use shortest-path distance, the runner could be materially worse than local-only navigation. Margin recalibration did not rescue it.

**Decision:** FAIL for universal one-scalar sufficiency.

**Theory update:** Useful memory representation depends on the observation/value channel.

---

## AP-S25–S29 — Frontier oracle and identifiability

**Question:** Is history useless under the spectral cue, or merely hard to exploit?

**Result:** Oracle frontier selection had very large latent gains, increasing strongly with retained candidate count. Observable score-based frontiers and myopic value models still failed end-to-end. A myopic classifier could identify locally better frontier states reasonably well, revealing a mismatch between local value and final trajectory utility.

**Theory update:** The central target should be counterfactual trajectory utility.

---

## AP-S30–S35 — Counterfactual trajectory utility

**Question:** Can trajectory-level supervision turn the latent frontier value into deployable gain?

**Result:** Yes, partially. A one-shot trajectory-utility intervention changed the effect from negative to positive. Reusing the same gate repeatedly degraded performance. Optimizing the short horizon alone harmed long-horizon success; adding a long-horizon safety constraint restored useful early gains without major long-horizon harm.

**Theory update:** Use one deliberate intervention and explicitly protect long-horizon outcomes.

---

## AP-S36–S38 — Memory compression

**Question:** How much abandoned-alternative memory is actually required?

**Result:** K=8 compressed to K=4 with almost no loss. Full ancestor history compressed to the immediate previous decision point. K<=3 failed the predefined retention/safety criterion in the tested bridge.

**Theory update:** Rich history can be compressed to a bounded O(1)-in-depth buffer, but not to one scalar.

**Implementation note:** An off-by-one history-window bug was detected in the first AP-S37 implementation. The affected result was discarded and the phase was rerun on fresh seeds with the corrected definition.

---

## AP-S39–S43 — Feature compression and confirmatory replication

**Question:** Can the decision model also be made compact?

**Result:** After correcting a threshold-calibration confound, the model compressed to three features: candidate degree, origin candidate count, and relative score. The frozen compact policy replicated on 6,000 new tasks.

**AP-S43 confirmatory result:**

- S@16: +3.9167 pp, 95% CI [+3.133,+4.683], 12/12 seeds positive.
- S@32: +1.1500 pp, 95% CI [+0.250,+2.067].

**Decision:** PASS for the compact spectral-bridge policy.

---

## AP-S44 — Capacity plateau 1..9

**Question:** Does a larger short-term buffer (7–9 candidates) materially help?

**Result:** No material additional gain over K=4.

**Theory update:** More pending links are not automatically better; performance plateaus around four in this bridge.

---

## AP-S45–S48 — Load, ambiguity, and delayed reconsideration

**Question:** Should the buffer persist across several pages, or be dynamically delayed until more context accumulates?

**Result:** Structural load suggested different task-side capacity demands, but simple ambiguity gating did not solve the problem. Reconsideration after several pages and persistent multi-page remapping did not establish an advantage over quick next-decision-point reconsideration.

**Theory update:** Prefer a short-lived option buffer over long-lived deferred tabs.

---

## AP-S49 — Adaptive shared-memory budget

**Question:** Can candidate memory be reduced under high branching while preserving most performance?

**Result:** A resource-saving K=5/4/3 mapping reduced mean stored candidates by ~20.9% versus fixed K=4, with small losses (~0.4 pp) in S@16/S@32. A task-demand K=3/4/5 mapping did not materially outperform fixed K=4.

**Decision:** Resource-saving trade-off supported; performance-improving adaptation not established.

---

# Current theory

The strongest current computational statement is:

> **Under the tested spectral/non-distance bridge, efficient navigation does not require all-history memory, but one abandoned alternative is insufficient. A short-lived buffer of about four alternatives from the immediately previous decision point, evaluated with a compact one-shot trajectory-utility rule, provides a reproducible early-budget gain without long-horizon harm.**

This remains a computational bridge. Real anchor/context multi-hop validation and human-reading experiments are still required.
