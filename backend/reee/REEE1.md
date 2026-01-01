# REEE1: Views, Not Layers (Multi‑Scale Epistemic Weaving)

This note refines the REEE architecture to resolve a recurring tension:

- We *need* principled invariants (or the graph percolates and becomes un-auditable).
- We *don’t* want to hard-freeze the world into arbitrary “L1–L5” boxes.

The resolution: **REEE is a single append‑only epistemic field** that can be **projected into multiple emergent views at multiple scales**, then “harvested” according to purpose (inquiries, feeds, summaries).

---

## 1) The Field: One Reality, Many Projections

REEE maintains one underlying state:

- **L0 Claims** (append-only, provenance-bearing observations)
- **Derived relations** between claims (costly, cached, versioned)
- **Computed features** (embeddings, entities, timestamps, extractors, etc.)

Everything else is a *projection* of that field:

- “Surfaces”, “events”, “narratives”, “frames” are not metaphysical layers.
- They are **views** produced by running a weave operator with a particular **semantics** and **scale**.

Formally:

```
view = Weave(field, operator, params, snapshot_id)
```

Where:
- `field` is the immutable evidence substrate (plus cached expensive decisions)
- `operator` specifies semantics (identity vs aboutness vs discourse)
- `params` selects scale (thresholds, time windows, hub penalties, etc.)
- `snapshot_id` pins an auditable worldline (“what did we believe then?”)

---

## 2) The Non‑Negotiables (Invariants Still Matter)

Even if we stop worshiping “layers”, we cannot drop these invariants without breaking epistemic correctness:

### 2.1 Identity vs Aboutness are different kinds of relations

- **Identity (same fact)** behaves like an equivalence‑like relation for merging and truth maintenance.
  - It supports safe operations like connected components for “same proposition”.
  - Claim relations: `CONFIRMS`, `REFINES`, `SUPERSEDES`, `CONFLICTS`, `UNRELATED/NOVEL`.

- **Aboutness (same incident/case)** is a *soft association*.
  - It is not reliably transitive and is vulnerable to bridges (percolation).
  - You must not treat it as an equivalence relation.

This is not a stylistic preference. It is the mathematical reason topic blobs happen.

### 2.2 L0 immutability and replayability

- Claims are append-only. No deletion, no mutation.
- Any “fix” happens by adding new evidence or changing parameters with provenance.
- Any derived object must be reconstructible from `(L0, params@version, snapshot_id)`.

### 2.3 Derived state purity

Operational actions can only influence outputs by:
- appending L0 claims (new evidence)
- updating parameters (versioned, audited)
- triggering recompute

No silent edits to surfaces/events as “truth”.

---

## 3) The Weave Operator as a Scale‑Family

Instead of declaring “L3 = event” universally, we run **families of weaves** at different scales and harvest whichever is stable and useful.

### 3.1 Scale is not duration; it is resolution

“Incident vs case vs saga” is not only about time. It is about the resolution at which we compress the topology:

- *Incident view*: tight temporal compatibility; strong discriminative anchors; bridge‑resistant.
- *Case view*: looser time; stronger reliance on entity relations; still bridge‑resistant.
- *Saga view*: long horizon; narrative/causal edges dominate; summary-oriented.

Time is a powerful prior for incident views, but it is not “what an event is”.

### 3.2 Stability selects “natural shapes”

To avoid arbitrary partitions, measure which clusters persist under perturbations:

- shuffle arrival order
- subsample claims/surfaces
- sweep thresholds within a small band

Clusters that are stable across perturbations are candidates for promotion (proto‑inquiries, feeds, summaries).

This is the operational meaning of “natural emergence”.

---

## 4) Proto‑Inquiries: Questions Emerge From Weaving

An Inquiry is a product contract. But the **question** often exists objectively in the evidence field.

### 4.1 Proto‑Inquiry (epistemic object)

A proto‑inquiry is emitted when a typed proposition surface (or surface family) exhibits tension:

- conflicting typed values (`typed_value_conflict`)
- unresolved contradictions (`unresolved_conflict`)
- single-source-only under high stakes
- high epistemic entropy (value uncertainty), not just semantic dispersion

Proto‑inquiry contains:

- `target`: typed variable (`question_key` + domain spec)
- `scope_signature`: incident/entity/time signature inferred from the field
- `evidence_refs`: surface IDs / claim IDs used
- `belief_summary`: posterior MAP + entropy + credible set
- `tensions`: meta-claims that justify “this is worth asking”
- `priority_score`: why this should be surfaced on the homepage

Proto‑inquiries can power the system even with zero users: the weaver “asks” what the evidence is struggling to settle.

### 4.2 Inquiry (contractual wrapper)

When a user adopts a proto‑inquiry, the product adds:

- stake/bounty rules
- resolution policy (≥95% for 24h, blocking tasks, etc.)
- deadlines and resolvers (for forecasts)
- participation workflow and moderation

The wrapper must not rewrite the evidence field—only add evidence or parameter updates.

---

## 5) Why Fragmentation and Percolation Both Happen (and the Remedy)

### 5.1 Fragmentation

High precision / low recall at “event emergence” usually means:

- L2 surfaces are mostly singletons (identity is strict, correct)
- L3 aboutness has insufficient binding signals for the chosen scale

Remedy is **not** to lower L2 until it becomes aboutness.
Instead:

- improve typed extraction coverage (more propositions become comparable)
- add scale‑appropriate binding signals (time compatibility, discriminative anchors, entity relations)
- use bridge‑resistant clustering (no raw connected-components on weak edges)

### 5.2 Percolation (mega-events)

Even a small ratio of bridge edges can collapse everything if L3 uses connected components.

Remedy:

- bridge‑resistant event builders (core/periphery, community detection, cut/bridge penalties)
- strict temporal compatibility for incident views
- discriminative anchor requirements (not just “shared celebrity name”)
- treat publishers as metadata, not binding entities

---

## 6) “Is this true?” and Second‑Order Information

Modern information is often “claims about claims”:

- `Said(A, p)`
- `Reported(B, p)`
- `p` (world proposition)

REEE should represent these as distinct typed propositions, each with its own surfaces and belief states.

This yields the right behavior:

- high confidence that “B reported p” does not automatically imply high confidence in `p`
- source plausibility can be learned from outcomes rather than hardcoded
- inquiry outputs can report *where uncertainty lives* (origin vs transmission vs world state)

---

## 7) Practical Product Harvesting

Different product surfaces harvest different projections:

- **Homepage**: high‑priority proto‑inquiries + newly resolved contractual inquiries
- **Search**: retrieval over inquiries + entities + surfaces/events (as views)
- **Inquiry page**: posterior + trace + tasks (meta‑claims) + contributions
- **Resolver worker**: binds inquiry ↔ surfaces in a pinned snapshot, evaluates resolution policy, emits tasks

REEE remains the epistemic constitution; the product chooses which views to expose and incentivize.

---

## 8) Concrete View Definitions

Based on empirical investigation, we define two primary event views:

### 8.1 IncidentEventView (cohesion operator)

**Purpose**: "These facts are about the same happening"

**Semantics**:
- Tight temporal window (default: 7 days)
- Requires discriminative anchor (high-IDF shared entity)
- 2-of-3 signal gate: anchor overlap + semantic similarity + entity overlap
- Bridge-resistant clustering (core/periphery, not connected components)
- Global hub penalty (entities appearing in >N surfaces get IDF=0)

**Example**: "Wang Fuk Court fire" - all claims about deaths, rescue, cause within the incident timeframe.

**What it rejects**: Background context ("building was constructed in 1980"), related but separate incidents ("another fire last month").

### 8.2 CaseView (narrative operator)

**Purpose**: "These happenings are part of the same story"

**Semantics**:
- Loose temporal window (months to years)
- Entity relation backbone as primary signal (Do Kwon ↔ Terraform Labs)
- Local hubness: dispersion-based (hub anchors suppressed, backbones bind)
- Shared topic/domain as secondary signal
- Bridge-resistant clustering (core/periphery, not connected components)

**Bridge-resistance implementation**:

CaseView uses core/periphery clustering to prevent mega-case formation:

1. **Core formation**: Only strong edges form cores
   - Score >= `case_core_threshold` (default 0.4)
   - Signals >= `case_core_min_signals` (default 2)
   - Connected components on this filtered strong graph define case cores

2. **Periphery attachment**: Incidents not in cores attach to best core
   - Score >= `case_periphery_threshold` (default 0.2)
   - Periphery incidents attach but never merge cores
   - Incidents with no strong connection remain singletons

3. **Hub suppression**: Before scoring, hub entities are filtered out
   - Hub anchors can't contribute to anchor overlap (Signal 1)
   - Hub entities in backbone pairs are blocked (Signal 2)
   - Hub entities in entity overlap are blocked (Signal 3)
   - Backbones (low dispersion) contribute to binding

This ensures: one weak edge (or one hub entity) cannot collapse unrelated cases.

**Example**: "Do Kwon / Terraform Labs saga" - founding (2018), TerraUSD launch (2020), collapse (2022), arrest (2023), sentencing (2025).

**What it rejects**: Other crypto cases (FTX, Mt. Gox) unless explicit causal/narrative link.

### 8.3 Hubness is Contextual (Dispersion-Based)

A key insight: the same entity can be:
- A **hub** in incident view (Do Kwon appears in 10+ surfaces → not discriminative)
- A **backbone** in case view (Do Kwon is the central figure of his own story)

**Implementation (dispersion-based, not IDF)**:

The problem with global IDF: it penalizes *any* high-frequency entity, but backbones like
"Do Kwon" or "Jimmy Lai" *should* bind their respective clusters. What makes "Hong Kong"
a hub is not frequency alone, but that it appears across *unrelated contexts*.

**The dispersion measure**:

For each anchor entity `a`, compute:
1. **Frequency**: In how many incidents does `a` appear?
2. **Cohesion of co-anchors**: For the other anchors that co-occur with `a`, do they
   also co-occur with each other?

If `a` appears with anchors that form a cohesive cluster (they co-occur with each other),
then `a` is a **backbone** - it binds related incidents.

If `a` appears with anchors that don't co-occur with each other (disjoint contexts),
then `a` is a **hub** - it bridges unrelated incidents and should be suppressed.

**Dispersion formula**:
```
co_anchors = {b : b co-occurs with a in some incident}
cohesion = |{(b1,b2) : b1,b2 co-occur in some incident}| / |all pairs in co_anchors|
dispersion = 1 - cohesion
```

**Classification**:
- `freq < threshold` → **neutral** (not enough signal)
- `freq >= threshold AND dispersion < 0.7` → **backbone** (binds)
- `freq >= threshold AND dispersion >= 0.7` → **hub** (suppressed)

**Application**:
- Incident view: global DF-based hub penalty (simple, works for tight time windows)
- Case view: dispersion-based local hubness (principled, captures context switching)

**Example**:
- "Hong Kong" (freq=7, dispersion=0.84) → HUB: its co-anchors (Jimmy Lai, Do Kwon,
  Tai Po, Time Magazine) don't co-occur with each other
- "Do Kwon" (freq=3, dispersion=0.40) → BACKBONE: its co-anchors (Terraform Labs,
  Luna, Montenegro) frequently co-occur in crypto incidents
- "Jimmy Lai" (freq=3, dispersion=0.33) → BACKBONE: its co-anchors (Apple Daily,
  Hong Kong) co-occur consistently in HK democracy incidents

### 8.4 View Selection

An inquiry or product surface chooses a view:
- "How many died in Wang Fuk Court fire?" → incident view
- "What is the Do Kwon / Terraform Labs story?" → case view
- "Will GPT-5 be released by July?" → case/forecast view

The resolver pins which view produced the trace, ensuring epistemic replayability.

### 8.5 Evaluation Alignment

Each view must be evaluated against appropriate ground truth:
- Incident view: incident-level GT (split legacy events by time gaps + anchor shifts)
- Case view: case-level GT (legacy events, storylines)

Measuring incident view against case GT produces false "low recall" signals.

---

## 9) Summary: The Refined Position

- Keep REEE invariants (identity/aboutness separation, L0 immutability, versioned parameters).
- Treat "L1–L5" as *useful default projections*, not an ontology.
- Let structure emerge by weaving at multiple scales; select "natural shapes" by stability.
- Let proto‑inquiries emerge from tensions; wrap them into user contracts in the product layer.
- **Explicitly produce IncidentEventView and CaseView as separate artifacts with distinct builders, params, and traces.**
- **Evaluate each view against its appropriate ground truth.**

