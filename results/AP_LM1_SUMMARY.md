# AP-LM1 Summary

Decision: **PASS**

## Primary confirmatory comparison
- learned full latency-aware vs global visible greedy, B=16: **+4.765 pp** oracle-normalized
- seed-cluster bootstrap 95% CI: **[+4.449, +5.078] pp**
- positive seeds: **12/12**
- B=32 delta: **+7.034 pp**

## Latency ablation
- aware minus no-latency, B=16: **+1.425 pp**, CI **[+1.301, +1.547] pp**
- aware minus no-latency, B=32: **+1.125 pp**, CI **[+1.015, +1.250] pp**

## Bounded frontier capacity
- K=4 gap from full: B=16 **1.556 pp**, B=32 **7.846 pp**
- K=8 gap from full: B=16 **0.093 pp**, B=32 **0.136 pp**
- minimal K within 1 pp of full learned frontier at both budgets: **8**

## Interpretation
Recursive subtree-value ranking beats global visible cue/latency greedy in this controlled answer-generated-unknown environment, and explicitly modeling latency adds independent value. Recursive query generation requires a larger deferred frontier than the earlier fixed-link controlled result (K≈4 vs K≈8 here).

## Boundary
This is a controlled recursive-query model with perfect answers and serial latency. It tests query-frontier control, not hallucination, asynchronous read/query overlap, or human comprehension.
