# AP-RS5 implementation audit

AP-RS5 is preserved exactly as run and remains a preregistered **FAIL**.

## Visibility correction

The semantic action score itself uses only real Wikispeedia **anchor text + containing paragraph**. However, one of the three trajectory-gate features is `log1p(candidate_outdegree)`, computed from the unvisited candidate page's outgoing links.

That feature is not normally available to an unaided human reader before visiting the candidate page. Therefore AP-RS5 should be interpreted as a **graph-assisted real-semantic causal policy test**, not as a strictly visible-only human-information-regime test.

This audit does not alter the AP-RS5 result or decision. AP-RS6 was specified before reading the AP-RS5 result and removes unvisited candidate metadata entirely.

## AP-RS5 result retained

- Decision: **FAIL**
- target-disjoint test tasks: 1,200
- equal anchor/context local S@16: 0.7425
- bounded policy S@16: 0.7475
- delta S@16: +0.50 pp, target-cluster CI approximately [-0.43, +1.45] pp
- equal local S@32: 0.8275
- bounded policy S@32: 0.8325
- delta S@32: +0.50 pp

A separate and important scorer result was that anchor-only local navigation was materially stronger than the frozen equal anchor/context scorer on this goal-directed Wikispeedia construct (S@16 0.7908 vs 0.7425). This should not be generalized back to reading/information-acquisition, where containing context was previously beneficial.