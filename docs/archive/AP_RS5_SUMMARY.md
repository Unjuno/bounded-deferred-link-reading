# AP-RS5 — Real anchor/context causal policy

**Decision: FAIL**

- test tasks: 1200 (target-disjoint)
- visible edge coverage: 0.9673
- selected margin: 0.02
- local equal S@16 / S@32: 0.7425 / 0.8275
- bounded policy S@16 / S@32: 0.7475 / 0.8325
- delta S@16: +0.500 pp, target-cluster CI [-0.004340561085726719, 0.014529455777364404]
- delta S@32: +0.500 pp, target-cluster CI [-0.00317707115579456, 0.013158165475416934]
- positive S@16 target buckets: 5/8
- intervention rate: 0.1967

## Local scorer comparison
{
  "anchor": {
    "S16": 0.7908333333333334,
    "S32": 0.8458333333333333
  },
  "context": {
    "S16": 0.38083333333333336,
    "S32": 0.5333333333333333
  },
  "equal": {
    "S16": 0.7425,
    "S32": 0.8275
  }
}

## Claim boundary
Real HTML anchor/context and real graph are used, but the outcome is simulated navigation, not human comprehension.
