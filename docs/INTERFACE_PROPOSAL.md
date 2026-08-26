# Interface Proposal: Low-Friction Adaptive Reading

## Status

This is a **design proposal derived from the analytic model and computational findings**. It is not a validated human-comprehension result.

The strongest design implication is narrower than “chat is the optimal interface”:

> A reading interface should help identify high-downstream-value uncertainties without forcing the reader to stop, and should make the right amount of explanation cheap when it becomes useful.

The proposal therefore separates three problems:

1. **salience** — what is likely to matter later;
2. **resolution timing** — should it be explained now or left alone;
3. **explanation depth** — if explained, how much is enough.

## V1: the smallest useful interface

The first version should be deliberately simple.

### 1. Keep the document primary

The main surface is still ordinary text. The system does not replace every difficult passage with generated explanation.

AI may apply **sparse, restrained emphasis** to concepts that are likely to have high downstream dependency.

The target is not “hard word detection.” A difficult term that never matters again may deserve no mark; a simple-looking concept that the rest of the argument depends on may deserve emphasis.

A rough internal priority is:

`priority(q) ~= unresolvedness * downstream_dependency * (1 - contextual_resolution_probability) - interruption_cost`

Recursive information value can be added when a concept is likely to open useful downstream inquiry.

### 2. Emphasis must not imply interruption

Highlighted text should remain readable as ordinary text. No explanation needs to open automatically.

This preserves the value of continuing to read: later context may explain the concept for free, show that it is irrelevant, or make a later explanation easier to target.

### 3. One small explanation-choice control

Activating an emphasized span can expose a compact set of actions such as:

- `No explanation`
- `1 line`
- `Example`
- `Detailed`
- `Later`

A custom chat question can remain available, but it does not need to be the default interaction.

These choices have two roles at once:

1. they let the reader control interruption and explanation depth;
2. they provide training signals about what this reader already knows and how they prefer explanations.

### 4. Put short explanations back into the reading flow

For `1 line` or `Example`, the lowest-friction presentation is an inline supplement immediately below/adjacent to the relevant text, followed by the original document.

The interaction should feel like:

```text
original text
short contextual supplement
original text continues
```

rather than:

```text
original text -> separate chat -> answer -> manually recover reading position
```

Chat remains useful for custom or deep questions, but routine clarification need not cause a context switch.

## Personalization without requiring a complete personal-data model

The interface does not need a perfect model of the reader before it becomes useful.

Each choice supplies an observation of the form:

`(concept, local context, document goal) -> chosen explanation mode`

Over time the system can estimate:

`P(explanation mode | concept/context, reader history)`.

Examples of learnable tendencies:

- mathematical prerequisites usually need no explanation;
- domain-specific terminology often needs a one-line gloss;
- mechanism questions are better with examples;
- some readers prefer formal derivations, others prefer minimal continuation-oriented explanations.

The useful user state is therefore closer to a **minimal knowledge-and-explanation model** than a broad collection of unrelated personal data.

## Hidden deferred frontier

The analytic model benefits from a frontier of unresolved items, but the interface does not have to expose a large frontier panel.

The system can internally retain items that were:

- noticed but not explained;
- explicitly marked `Later`;
- introduced by an explanation;
- likely to become important downstream.

As reading continues, the system can silently:

- retire items resolved by context;
- lower items that turn out to be peripheral;
- raise items that become prerequisites.

A visible “saved for later” list can be optional. The user should not inherit the system's bookkeeping burden.

## Progressive automation

A sensible progression is:

### Stage A — manual depth choice
AI marks sparse high-value candidates; the reader chooses `No explanation / 1 line / Example / Detailed / Later`.

### Stage B — personalized defaults
The system preselects the likely explanation mode but waits for the reader to activate it.

### Stage C — conservative automatic supplements
When confidence is high and interruption cost is low, the system may insert a very short supplement automatically. The reader can disable or undo this behavior.

### Stage D — continuous adaptive reading stream
In the long-run interface, the system may continuously decide what text to show next: original material, a short prerequisite, an example, a compressed known section, or a deeper explanation.

The information stream can continue without requiring explicit question/answer turns. Internally, the system still solves a defer-versus-resolve scheduling problem.

Crucially, “continuous” means **the system need not block waiting for a query cycle**. It does not mean the reader cannot pause, slow down, inspect the source, or take control.

## Role of chat

Chat is still useful, but its role becomes clearer:

- custom questions;
- deep explanation;
- comparison or synthesis across passages;
- user correction of the system's assumptions.

For routine clarification, chat can be an underlying capability rather than the main navigation structure.

The long-run direction may therefore be less “better linear chat” and more **chat capabilities embedded into an adaptive reading stream**.

## Why this follows from the research

The proposal maps directly to the defer-vs-resolve formulation:

- sparse emphasis reduces the cost of detecting potentially valuable unknowns;
- optional explanation preserves contextual self-resolution;
- the `Later` option preserves query option value;
- explanation-depth controls trade information gain against time/attention cost;
- interaction history improves observability of the reader's knowledge state;
- a hidden frontier prevents recursive depth-first detours from becoming the default UI.

## Minimal prototype

A first prototype only needs:

1. a document viewer;
2. sparse AI emphasis on predicted high-downstream-dependency spans;
3. the five explanation actions above;
4. inline contextual explanation;
5. logging of choices for personalization;
6. a normal custom-chat escape hatch.

It does **not** require a visible frontier panel, autonomous popups, or a complete long-term personal profile.

## Testable interface hypotheses

These remain hypotheses until directly tested with readers.

### H-UI1 — downstream salience beats difficulty-only salience
Emphasizing concepts based on expected downstream dependency should be more useful than highlighting merely difficult/rare terms.

### H-UI2 — optional explanation beats forced explanation
Sparse emphasis plus reader-controlled explanation should preserve more contextual self-resolution and impose less interruption than automatic popovers at every difficult item.

### H-UI3 — explanation-mode choices support useful personalization
A small history of `No explanation / 1 line / Example / Detailed / Later` choices should predict later explanation needs better than a non-personalized fixed verbosity policy.

### H-UI4 — inline supplements reduce context-switch cost
Short explanations embedded in the reading flow should reduce navigation/interruption overhead relative to moving every clarification into a separate linear chat.

### H-UI5 — conservative automation can approach a continuous adaptive stream
Once the model has enough evidence about the reader, selectively auto-inserting high-confidence short supplements should reduce manual interaction without materially increasing unwanted explanations.

## Minimal validation study

If one human-interface experiment is later run, a clean progression is:

1. document + ordinary chat;
2. sparse AI salience + explanation-choice buttons;
3. the same interface with personalization from prior choices.

Possible outcomes:

- comprehension/task success;
- total reading time;
- number and depth of explanations;
- explanations later shown to have been unnecessary;
- time lost to context switching;
- subjective interruption/cognitive load.

A future fourth condition could test conservative automatic inline supplements, but it is not needed for the first validation.

## Design boundary

The computational work supports the **structure of the optimization problem**, not the human-factors superiority of this exact UI.

It does not yet establish:

- the best visual emphasis style or density;
- the optimal button labels;
- how accurately AI can infer individual knowledge from sparse interaction;
- whether visible or hidden frontiers work better for humans;
- when automatic supplements become helpful rather than distracting;
- whether a continuous adaptive stream improves comprehension or retention.

Those are design hypotheses derived from the theory, not results already demonstrated by the navigation/query experiments.
