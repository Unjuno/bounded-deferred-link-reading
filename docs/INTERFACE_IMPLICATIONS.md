# Interface Implications: From Chat to Deferred-Query Control

## Why interface design now follows from the theory

The optimization model does not imply that users should manually estimate the importance of every unknown. A better interface should help expose likely high-value unknowns and reduce the cost of asking about them.

The key product consequence is:

> **The interface should externalize both importance estimation and deferred-query memory.**

The user should be able to keep reading while the system highlights, retains, and reprioritizes unresolved items.

## 1. Highlight likely high-value dependencies

Instead of asking the reader to decide from scratch what matters, the system can visually mark terms or passages predicted to have high downstream dependency.

Possible signals:

- bold text or subtle emphasis,
- underline / margin markers,
- one-click "explain this" affordances,
- stronger emphasis when a concept becomes a prerequisite for later material.

The goal is not to highlight every difficult phrase. The target is **expected downstream value**.

A rough ranking target is

`priority(q) ~= unresolvedness * downstream_dependency * (1-contextual_resolution_probability) + recursive_gain - query_cost`.

## 2. Do not force immediate explanation

A highlighted item should not automatically interrupt reading.

Preferred interaction:

1. mark a possibly important unknown;
2. let the reader continue;
3. keep the item in a visible or latent deferred frontier;
4. lower its priority if later context resolves it;
5. raise its priority if later text depends on it;
6. surface an explanation when resolution value exceeds deferral value.

This preserves the central benefit of context-first reading.

## 3. Add a deferred-query frontier to chat

Ordinary chat is linear and tends to encourage recursive depth-first questioning:

`question -> answer -> follow-up -> answer -> deeper follow-up ...`

The theory instead suggests three concurrent objects:

- **main reading stream**,
- **unresolved frontier**,
- **chat/explanation channel**.

When an answer introduces new unknowns, they should return to the frontier rather than automatically becoming the next question.

The assistant should be able to recommend:

- "ask this now",
- "keep this pending",
- "this was resolved by later context",
- "this is probably not needed for your current goal".

## 4. Make question formulation nearly frictionless

If the system has the selected span plus surrounding context, it can precompose the query.

Instead of the user typing a generic question such as

`What is entropy?`

an interface can offer

`Explain entropy only to the depth needed for the argument in these paragraphs.`

This reduces query cost and increases context conditioning.

## 5. Optimize explanation depth, not only query timing

The action should be modeled as `(q, depth)` rather than only `q`.

Possible depths:

- one-line gloss,
- short explanation,
- example,
- formal derivation,
- extended tutorial.

A low-dependency unknown may deserve only a gloss. A concept supporting the rest of the document may justify a deeper explanation.

Thus the interface should optimize:

1. **what** to ask,
2. **when** to ask,
3. **how deeply** to answer.

## 6. External working memory

Humans are bad at retaining many unresolved questions while reading. The interface can store:

- the unresolved span,
- where it appeared,
- why it may matter,
- current predicted importance,
- whether later context has partially or fully resolved it.

This turns chat into external working memory rather than only an answer box.

## 7. A likely future reading UI

A plausible interface has:

### Main pane
The document remains primary. High-value dependencies receive restrained emphasis.

### Frontier pane
A compact list of unresolved items, dynamically ranked and automatically retired when resolved.

### Chat pane
Explanations are generated only when needed. New unresolved concepts from answers are returned to the frontier.

### Optional assistant prompts
The system may occasionally surface one high-value suggestion such as:

- "This concept now appears to be required for the next section."
- "You can keep reading; this term is explained shortly."
- "A one-line definition is probably sufficient here."

## 8. Interface hypotheses derived from the theory

These are testable predictions rather than established human-factors results.

### H-UI1 — Importance highlighting
AI-ranked highlighting should reduce the effort required to identify which unknowns deserve attention relative to unassisted reading.

### H-UI2 — Deferred highlighting beats forced popovers
Highlighting plus deferred access should outperform automatically opening explanations whenever a difficult term appears, because forced popovers destroy contextual self-resolution and impose interruption cost.

### H-UI3 — Dynamic reprioritization beats static highlights
If importance changes as later context arrives, highlights/frontier ordering should update. Static pre-highlighting should be inferior when downstream dependency is not locally observable.

### H-UI4 — Frontier chat beats linear recursive chat
A chat interface that returns follow-up unknowns to a ranked frontier should reduce unnecessary depth-first exploration compared with a standard linear conversation.

### H-UI5 — Adaptive answer depth beats fixed verbosity
Selecting explanation depth based on downstream dependency should reduce reading time/attention cost without materially reducing comprehension.

## 9. Minimal product experiment

A minimal human/agent study could compare four interfaces on the same documents and questions:

1. plain text,
2. plain text + ordinary chat,
3. AI importance highlighting + ordinary chat,
4. AI importance highlighting + deferred frontier + chat.

Measure:

- comprehension / task success,
- reading time,
- number of queries,
- explanation tokens/time consumed,
- number of unnecessary queries later resolved by context,
- subjective interruption/cognitive load.

The critical comparison is (3) vs (4): highlighting alone helps candidate detection; the frontier tests whether deferred resolution adds value beyond salience guidance.

## 10. Product interpretation

The strongest interface claim supported by the theory is not "chat is the optimal UI". It is:

> **Chat is a particularly compatible execution surface for a deferred-query policy, especially when augmented with AI-ranked salience and an explicit unresolved frontier.**

The likely evolution is therefore from

`chat as answer box`

toward

`chat as context-aware query scheduler + external working memory`.
