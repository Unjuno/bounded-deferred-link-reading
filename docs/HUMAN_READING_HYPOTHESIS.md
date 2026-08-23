# Human-Reading Hypothesis

## Status

This document translates the computational findings into a **testable human-reading hypothesis**. It is not a validated prescription.

## Bounded Contextual Deferred-Link Hypothesis

When hyperlink value is uncertain, a human reader may benefit from treating links as **deferred options** rather than immediate commands to leave the current text.

A practical version is:

> Read enough of the current passage to build context. Keep only a few promising links as short-lived options. Reconsider those options soon after additional context has been acquired, and follow at most one if it now appears more useful than continuing locally.

The hypothesis has four parts.

### 1. Context-first valuation

The meaning and usefulness of a link depend on the sentence, paragraph, and current reading goal. Therefore the reader should first understand why the link appears in the current context.

### 2. Small bounded option buffer

Do not attempt to remember every possible branch. Keep only a small number of promising alternatives. In the current computational bridge, performance plateaus around four retained candidates, while 7–9 provide little additional gain.

**Important:** the computational value K≈4 is not evidence that human working-memory capacity is exactly four links.

### 3. Short deferral, not indefinite postponement

The current experiments favor reconsideration at the next decision point. Retaining the same pending alternatives across several pages did not improve the tested navigation outcome.

Thus the hypothesis is better described as a **short-lived option buffer** than as "save links and come back much later."

### 4. One deliberate reconsideration

Repeated discretionary returns degraded performance in the computational bridge. The current hypothesis therefore favors one deliberate reconsideration over repeated back-and-forth navigation.

## Human-readable rule

> **Do not click every link immediately. Read the surrounding context, keep a few promising links in mind or in a note, and at the next natural stopping point reconsider which one—if any—would improve your understanding most. Then choose one and avoid repeated hopping.**

## Why this might work

Let the reader's current context representation be \(C_t\), and let \(l\) be a pending link. The reader can only estimate its utility:

\[
\hat U(l \mid C_t).
\]

After reading more locally, the context changes to \(C_{t+1}\), allowing a revised estimate:

\[
\hat U(l \mid C_{t+1}).
\]

Deferral is useful when the expected value of better-informed choice exceeds the cost of continuing to read locally.

Conceptually:

\[
\text{defer if } \operatorname{E}[\text{better decision after more context}] > \text{cost of waiting}.
\]

The computational results further suggest that the relevant decision target is not "is this link locally attractive?" but:

> **If I switch to this option now and continue using the same imperfect reading policy, will the final outcome improve?**

That is a trajectory-level, not purely local, utility question.

## Working-memory interpretation

A possible human extension is that context, goal, and pending links compete for limited working-memory resources. This motivates a shared-budget view:

\[
M = M_{context} + M_{goal} + M_{pending\ links}.
\]

The computational AP-S49 result is compatible with a resource/performance trade-off: using fewer pending options reduced candidate memory by about 21% for a small performance loss. However, AP-S49 used graph branching as a task-load proxy, not measured human cognitive load.

Therefore the human prediction is deliberately weaker:

> Under higher cognitive load, readers may prefer fewer pending links to preserve resources for context representation, even if this sacrifices a small amount of search performance.

This prediction requires direct human testing.

## Proposed human experiment

### Conditions

1. **Immediate-click:** participants may follow links as soon as they appear.
2. **Straight-through:** participants are instructed to minimize link traversal during the initial passage.
3. **Bounded deferred-link:** participants read local context first, retain up to a small number of promising links, and reconsider them at the next natural boundary.
4. Optional capacity arms: pending-link cap 1, 2, 4, and 7.

### Outcomes

Primary:

- comprehension score,
- delayed retention,
- information acquisition per unit time.

Secondary:

- reading/navigation time,
- number of link traversals,
- backtracking count,
- self-reported cognitive load,
- confidence calibration,
- proportion of pending links eventually selected.

### Key falsifiable predictions

- A bounded deferred-link condition should outperform immediate indiscriminate traversal when link value is uncertain.
- Increasing the pending-link buffer beyond a modest size should show diminishing returns.
- Repeated reconsideration/backtracking should not provide proportional benefit.
- The optimal buffer may shrink under high cognitive load if context representation competes with pending-link memory.

## Japanese summary

計算実験を人間向けに翻訳すると、仮説は次のようになります。

> **リンクを見つけてもすぐ飛ばず、まず周囲の文章を読んで文脈を作る。重要そうなリンクだけを少数保留し、次の自然な区切りで一度だけ再評価する。その時点で今の流れより価値が高いリンクがあれば一つ選び、何度も行ったり来たりしない。**

現時点では、これは検証可能な読解仮説であり、人間に対して最適だと実証された方法ではありません。
