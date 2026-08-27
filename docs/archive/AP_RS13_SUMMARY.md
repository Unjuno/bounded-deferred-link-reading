# AP-RS13 — K1 trigger-only visible semantic state

**Decision: FAIL**

- selected: V0, margin=0.0
- test missions: 2400
- local S@16 / S@32: 0.7817 / 0.8492
- policy S@16 delta: -1.625 pp, CI [-0.028394305897186795, -0.004610178035468033], buckets + 1/8
- policy S@32 delta: -0.083 pp, CI [-0.00774526678141136, 0.006316995288390677]
- intervention rate: 0.6629
- same-test oracle K1 S@16: +8.125 pp, CI [0.0675379945292423, 0.09672094996449362]
- learned/oracle recovery: -0.2
- fit AUC: {'V0': 0.7238476872478344, 'V1': 0.7311019274528175, 'V2': 0.7367236648373711}
- held-out state AUC: {'V0': 0.7039530153456225, 'V1': 0.7041234039381064, 'V2': 0.689138792157865}

## Boundary
No deferred destination prefetch. V2 uses only already-visible semantic state; navigation is simulated and is not a comprehension study.
