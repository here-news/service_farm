# Recursive Epistemic Emergence Engine (REEE)

REEE is the working name for our universal epistemic engine: a recursive, multi-level system that ingests **atomic claims** and lets higher-order structure (facts, surfaces, events, narratives) **emerge** from evidence geometry and provenance.

The key design choice is a strict separation between:

- **Identity (same fact)**: hard, mergeable relations used for truth maintenance.
- **Aboutness (same event/case)**: soft, graded associations used for narrative/event formation.

This separation prevents the two dominant failure modes seen in news graphs:
1) **Topic blobs** (everything merges via hubs), and 2) **Permanent fragmentation** (nothing ever merges).

---

## Layer Model (L0 → L5)

- **L0 ClaimObservation**: append-only stream of observations with provenance.
- **L1 Proposition**: deduplicated facts with version chains and explicit conflicts.
- **L2 Surface**: bundles of claims/propositions connected by **identity edges only**.
- **L3 Event**: groups surfaces via **aboutness links only** (same event, different aspects).
- **L4 Narrative**: temporal/causal/discourse edges between events (interpretive, attributed).
- **L5 Meaning**: frames/stakes/questions (interpretive, audience-dependent).

REEE is “recursive” because the same *pattern* repeats at higher levels:
compute features → build a relation graph → cluster/group → interpret → emit meta-observations.

---

## Core Invariants (Non-Negotiable)

1) **L0 immutability**
   - Claims are append-only: never deleted, never modified.
   - Cold storage is allowed, but lineage must be preserved.

2) **Identity/Aboutness separation**
   - L2 surfaces use **identity edges only**: `CONFIRMS/REFINES/SUPERSEDES/CONFLICTS`.
   - L3 events use **aboutness links only** (between surfaces, not claims).
   - Never mix identity and aboutness into a single connected-component clustering step.

3) **Derived state purity**
   - L1–L5 are deterministic functions of `(L0, params@version)`.
   - Operational actions influence outcomes only via new L0 claims or versioned parameter updates.

4) **Stable claim relation vocabulary**
   - Claim-level relations remain: `CONFIRMS, REFINES, SUPERSEDES, CONFLICTS, NOVEL/UNRELATED`.
   - Domain differences come from extraction (e.g., q1/q2 question keys), not from adding relations.

5) **Meta-claims are observations**
   - Meta-claims describe the epistemic state (tension, gaps, contradictions).
   - They are not world facts and must not be re-injected as L0 claims.

---

## Tier‑1: Identity Graph (Truth Maintenance)

### Goal
Decide when two claims are about the **same underlying fact/question**, so we can:
- corroborate (increase support),
- refine (increase precision),
- supersede (update a value), or
- record contradictions (same fact, incompatible values).

### q1/q2 (Question Key) as the Indexing Primitive
REEE uses the q1/q2 pattern by extracting a `question_key` for a claim:
- `"13 dead"` → `question_key=death_count`, `value=13`
- `"fire started at 3am"` → `question_key=start_time`, `value=3am`

Claims only compete for identity *within the same question bucket*.
This improves both correctness (fewer false matches) and efficiency (fewer comparisons).

### LLM use at Tier‑1
LLM is used conservatively:
- to extract `question_key` when rules don’t match,
- and as fallback identity classification when a claim cannot be bucketed.

Embedding similarity is used only as a **gate** (candidate generation), never as identity truth.

---

## L2 Surfaces (Identity Components)

A **Surface** is an identity-connected sub-topology:
- claims that are the same fact (including conflicts about that fact),
- plus the internal relation edges that justify the grouping.

**Important:** `CONFLICTS` is still an identity relation. Contradictory claims belong together because they disagree about the *same* fact.

---

## Tier‑2: Aboutness and Event Formation

### Why “aboutness” exists
News events contain many different facts that should not be merged as “same fact”:
- conditions, charges, diplomatic pressure, advocacy statements, etc.

These are related by **aboutness**: “different aspects of the same incident/case”.

### Why connected-components is dangerous at Tier‑2
Aboutness is not transitive and is highly vulnerable to “bridges”:
- one generic entity (“Hong Kong”, “Donald Trump”) can glue unrelated stories.

So Tier‑2 needs:
- **multi-signal evidence** (not single overlap),
- **hub/bridge suppression** (generic/context entities must be weak),
- and ideally **core/periphery** membership (not hard merges everywhere).

### The “spring” intuition
Think D3 force layout:
- within-event links should be “stiff springs”,
- incidental references should be “weak springs” that don’t bind clusters.

Practical rule: **single-signal overlaps must be capped** so they can’t merge events.

---

## Self-Observation via Meta-Claims

REEE emits meta-claims when the topology is stressed, for example:
- `high_entropy_surface`: surface is semantically dispersed (likely mixed/aspect-unclear).
- `unresolved_conflict`: same-fact contradiction needs adjudication.
- `single_source_only`: corroboration task candidate.
- `bridge_node_detected`: potential “hub glue” connecting otherwise distinct regions.

Operational layers consume meta-claims to:
- request new evidence (append L0 claims),
- adjust parameters (versioned),
- trigger local recomputation (“healing”).

---

## Evaluation & Experiments

The project uses:
- **B³ precision/recall/F1** for clustering quality (handles fragmentation).
- **Completeness** to measure “event split” severity.
- **Gate recall vs LLM recall** to identify bottlenecks.
- **Bridge / hub diagnostics** to detect percolation risks.

The recurring empirical lesson:
- Increasing recall by lowering thresholds tends to create percolation unless bridge signals are capped or hub-like entities are downweighted.

---

## Practical Usage (Current Reference Implementation)

The REEE implementation lives in `backend/reee/`:

```
backend/reee/
├── REEE.md              # This constitution
├── __init__.py          # Public API exports
├── types.py             # Core data types (Claim, Surface, Event, Parameters)
├── engine.py            # Main orchestrator (Engine/EmergenceEngine)
├── interpretation.py    # LLM interpretation for surfaces/events
├── identity/            # L0 → L2: Identity linking
│   ├── linker.py        # Two-path identity linking
│   └── question_key.py  # Question key extraction
├── aboutness/           # L2 → L3: Aboutness scoring
│   └── scorer.py        # Multi-signal aboutness with hub suppression
└── meta/                # Meta-claims and tension detection
    └── detectors.py     # Tension types: high_entropy, single_source, conflicts
```

Typical flow:
```python
from reee import Engine, Claim, Parameters

engine = Engine(params=Parameters())
await engine.add_claim(claim)        # L0 → build identity edges
engine.compute_surfaces()             # L2: identity components
engine.compute_surface_aboutness()    # L3: candidate links
engine.compute_events()               # L3: event grouping
engine.detect_tensions()              # Meta-claims
```

---

## Glossary

- **Identity**: “same fact” (mergeable, truth-maintenance semantics).
- **Aboutness**: “same event/case” (soft association, narrative semantics).
- **Surface**: identity-connected component (a proposition + its dispute/update structure).
- **Event**: aboutness grouping of surfaces (multi-aspect incident/case).

