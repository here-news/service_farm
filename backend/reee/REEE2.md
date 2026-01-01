# REEE2: Inquiry Product Epistemology (Rigor, Scaffolding, Incentives)

This note extends `backend/reee/REEE1.md` into the *product layer*.

- `REEE1` defines the epistemic field and view operators (incident/case), plus invariants.
- `REEE2` defines how a user-facing **Inquiry** system should wrap those views to support:
  - “Is this true?” questions,
  - participation and evidence gathering,
  - auditable plausibility updates,
  - and safe incentives.

The design goal is *epistemic quality under real-world ambiguity*, not rhetorical certainty.

---

## 1) Core Separation: ProtoInquiry vs Inquiry

### 1.1 ProtoInquiry (system-emitted epistemic object)

ProtoInquiries emerge from weaving and meta-claims. They are **not** user contracts.

Derived from:
- typed surfaces (L2 truth maintenance) and/or view clusters (incident/case),
- tension signals (conflict, high uncertainty, single-source-only, suspected contamination),
- stable scope signatures (entity sense + time window when available).

ProtoInquiry payload (minimum):
- `target`: proposition schema (question_key + domain)
- `scope_signature`: entities + time window + incident/case view identifier
- `evidence_refs`: surface IDs / incident IDs + snapshot pointer
- `belief_summary`: posterior MAP + entropy + credible set (when typed)
- `tensions`: meta-claims that justify “this is worth asking”
- `priority_score`: used for homepage surfacing

ProtoInquiries can power the product with zero users (homepage and task queue).

### 1.2 Inquiry (contractual wrapper)

An Inquiry is a persisted product object that adds:
- stake/bounty economics and participation workflow,
- explicit resolution policy (what counts as “settled”),
- optional deadlines and adjudication resolvers (forecasts),
- moderation rules and provenance requirements.

**Invariant:** Inquiry cannot “rewrite truth”. It can only:
- append new L0 claims (new evidence),
- propose versioned parameter updates,
- trigger recompute/snapshots.

---

## 2) Rigor Ladder: A, B, C (and D)

User questions arrive under-specified. The product should *scaffold* them into well-posed forms.
Rigor is a property of the inquiry specification and available evidence, not a moral judgment.

### 2.1 Rigor “Wrapping” (composition)

Many real user questions aren’t a single atomic proposition; they are *built out of* smaller, more adjudicable propositions.

- **Rigor A** is “atomic”: one typed target, one scope, one adjudication rule.
- **Rigor B** is a *bundle/aggregation of A-units* where the typed target exists but calibration/adjudication is incomplete.
- **Rigor C** is a *bundle/aggregation of B-units (and A-units)* where the user’s term is contested or inherently multi-factor; output is an explicit index/scorecard unless upgraded.

One shorthand:
- `B ≈ Wrap(A, A, A, …)` (typed, but blocked by missing priors/resolvers or sparse evidence)
- `C ≈ Wrap(B, B, …, A, A, …)` (definition contested; must decompose into proxies/hypotheses)

This matters because REEE’s “superpower” is not magical certainty; it is that it can:
1) **decompose** high-level questions into explicit sub-propositions,
2) **trace** how each sub-proposition is supported/refuted (with provenance),
3) **re-compose** them into an index or decision rule that the user explicitly accepts.

### Rigor A: Typed + adjudicable

Definition:
- target is a typed proposition with clear scope,
- admissible evidence is well-defined,
- resolution rule exists (primary source or deterministic resolver).

Examples:
- “Did Reuters report p?” (report-truth)
- “Current death toll of incident X?” (monotone count with timestamps/inequalities)
- “What is the verdict/status?” (categorical with docket/official record)

Output promise:
- posterior is meaningful and replayable,
- “Resolved ≥ 95%” badge is allowed under policy.

### Rigor B: Typed, but incomplete calibration / incomplete adjudication

Definition:
- typed target and scope exist,
- but evidence is incomplete or model calibration is weak (generic priors, missing time),
- resolution may be blocked by tasks (“need primary source”, “time missing”).

Examples:
- breaking casualty count with conflicting reports and no official statement yet,
- forecast before a resolver is mature.

Output promise:
- posterior is meaningful as “best belief given evidence” with explicit uncertainty,
- resolution badge is conservative and policy-gated.

### Rigor C: Exploratory / index-based (definition still contested)

Definition:
- question is not a single typed variable,
- inquiry must be decomposed into proxies (a scorecard) or competing hypotheses,
- output is an **index** unless/until the user defines a typed target.

Examples:
- “Is Trump’s base collapsed?”
- “Is the US trustable?”

Output promise:
- evidence map + trace + tasks,
- an index/score only if its formula and inputs are explicit,
- no “Resolved ≥ 95%” badge unless upgraded to A/B.

### Rigor D: Meta/system inquiry

Definition:
- the target is a measurable property of REEE itself (calibration, false-merge rate, stability),
- evidence is evaluation runs and postmortems, not world facts.

Example:
- “Is REEE reliable on death_toll inquiries (last 30 days)?”

### 2.2 Examples (what users ask vs how REEE scaffolds)

**Example 1: “Is Trump’s base collapsed?” (Rigor C → can upgrade parts)**
- Problem: “base” and “collapsed” are undefined; there is no single decisive observation.
- Scaffold into explicit proxies (each can be Rigor A/B):
  - `approval_rating(t)` (time series; “collapsed” threshold rule)
  - `primary_turnout_share(t)` (resolver = election results)
  - `donation_rate(t)` (if available)
  - `GOP_self_id_as_MAGA(t)` (survey instrument; resolver = poll dataset)
  - `elite_defections(t)` (typed count of defections; evidence = named statements)
- Output: a **scorecard/index** + trace, not “true/false” by default.

**Example 2: “Did the US send humans to the moon?” (Rigor A/B depending on contract)**
- If the inquiry contract is “Is there primary-source evidence that Apollo missions occurred?”, it can be **Rigor A**:
  - targets: existence of specific artifacts (mission logs, telemetry datasets, photographs with provenance, named participants, launch records)
  - resolver: curated primary archives + cross-corroboration requirements
- If the contract is “did it really happen?” in a metaphysical sense, it is **Rigor C** unless the user accepts concrete adjudicators.

**Example 3: Forecast: “Will GPT‑5 be released before July 2025?” (Rigor B → A via resolver)**
- There may be *no direct claims* until release; evidence is often indirect.
- Make the contract explicit:
  - resolver: “OpenAI publishes a model named GPT‑5 on date ≤ deadline” (or a specific API/model-card identifier)
  - absence-of-event: at deadline, if resolver artifact doesn’t exist, resolve “no” (decidable without someone saying “it won’t”)
- REEE can still accumulate precursor evidence as sub-inquiries (Rigor B):
  - `credible_leak(model_name, date_range)` (report-truth)
  - `official_announcement(date)` (categorical)
  - `beta_access_reports` (count + provenance)

**Example 4: “Is the US trustable?” (Rigor C by default)**
- Ask “trustable for what?” and scaffold into domains:
  - treaty compliance, fiscal reliability, human rights, statistical reliability, wartime claims, etc.
- Each domain becomes a scorecard with typed sub-inquiries; the overall “trust” score is a user-accepted aggregation, not a hidden editorial output.

### 2.3 Glossary (UI-facing)

- **MAP estimate**: the *most probable* hypothesis/value under the current posterior (argmax).
- **B³ (B-cubed)**: a clustering evaluation metric used in experiments; it measures precision/recall of cluster assignments by scoring each item against its cluster memberships (useful when “ground truth events” are known).

---

## 3) Scaffolding: Converting User Intent Into a Well-Posed Inquiry

Most users will ask natural language questions that are ambiguous. The product should:

1) **Offer templates** (high rigor, one-click)
   - monotone count, categorical, report-truth, quote-authenticity, deadline forecast.

2) **Ask for scope** (minimal, defaults allowed)
   - entities (with sense IDs when possible),
   - time window,
   - jurisdiction/case identifiers if available.

3) **Ask for a collapse/threshold rule** (for “is it X?”)
   - example: “collapse = drop ≥ 15 points for ≥ 8 weeks”.

4) **Generate initial tasks** (meta-claim seeding)
   - “need primary source”
   - “need second independent corroboration”
   - “suspected scope contamination”
   - “need timestamped evidence”

5) **Allow upgrade**
   - C → B → A as definitions and primary evidence arrive.

---

## 4) Resolution Policy (What “Resolved” Means)

Resolution is a *contract rule*, not a vibe.

Suggested policy:
- “Resolved” requires posterior ≥ 0.95 for ≥ 24h,
- and no blocking tasks remain (primary source, scope contamination),
- and stability under recent evidence arrival (no large oscillations).

### Deadline / forecast inquiries

Forecasts require a resolver:
- at deadline, check for an admissible “occurred” artifact (official release, filing, election result),
- if not present, resolve “did not occur” under the contract (absence-of-event is decidable).

This is not “someone reported no”; it is adjudication under a declared rule.

---

## 5) Incentives: Reward Epistemic Work, Not Posterior Manipulation

If incentives depend on “moved the posterior”, the system will be gamed by novelty injection.

### 5.1 What to reward immediately
- completing a meta-claim task (primary source, disambiguation, scope split),
- adding high-provenance evidence with verifiable links,
- reducing contamination (finding bridges, removing mis-scoped evidence).

### 5.2 What to reward later (proper scoring)
- use proper scoring rules when outcomes are adjudicated (resolved inquiries),
- reward contributions that improved predictive log-score / calibration,
- penalize repeated low-quality contributions over time (reputation decay).

---

## 6) Second-Order Information: Report-Truth vs World-Truth

Modern text often states:
- `Said(A, p)`
- `Reported(B, p)`
- `p` (world proposition)

The product should default to representing attribution as separate propositions:
- high confidence in `Reported(B, p)` does not automatically imply high confidence in `p`,
- inquiry pages should explicitly show where uncertainty lives (origin vs transmission vs world state).

---

## 7) Data Model and Persistence (Product vs Epistemic Field)

Canonical split:
- **Epistemic field** (claims + surfaces + views) is derived, versioned, replayable.
- **Product objects** (inquiries, stakes, tasks, contributions) are persisted as user state.

### Suggested storage split (implementation detail, not required)
- Postgres: Inquiry, Contribution, Stake, Task, Resolution snapshots
- Graph/warehouse (optional): claim topology read-model for traversal/visualization

**Invariant:** inquiry traces must pin a snapshot:
- `(snapshot_id, params_version, view_scale)` so “why” is replayable.

---

## 8) Feasibility: What REEE Can and Cannot Guarantee

REEE can guarantee:
- provenance, replayability, and contestability,
- structured conflict (contradictions co-located),
- calibrated uncertainty for typed variables under declared assumptions,
- percolation resistance when using bridge-resistant views.

REEE cannot guarantee:
- a single “true” event boundary at all scales,
- resolution of ill-posed questions without user definition,
- perfect extraction without error (hence guardrails and contestability).

The product should make these boundaries explicit and use tasks + scaffolding to improve quality over time.
