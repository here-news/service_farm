# Epistemic Core Architecture

## Overview

The epistemic core is a principled system for truth emergence from converging evidence. It maintains strict separation between **epistemic semantics** (what we know and how confidently) and **operational infrastructure** (storage, federation, task generation).

```
┌─────────────────────────────────────────────────────────────┐
│                    OPERATIONAL LAYER                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Attention   │  │ Federation  │  │ Task Generator      │ │
│  │ Decay       │  │ Summaries   │  │ (from meta-claims)  │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │            │
│         ▼                ▼                     ▼            │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              VERSIONED SNAPSHOTS                        ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                              │ new L0 claims / parameter updates
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    EPISTEMIC CORE (Pure)                    │
│                                                             │
│  L5 Meaning    │ frames, stakes, questions                  │
│  L4 Narrative  │ temporal/causal/discourse edges            │
│  L3 Event      │ aboutness clustering (surfaces → events)   │
│  L2 Surface    │ identity components (propositions bundled) │
│  L1 Proposition│ deduplicated facts, version chains         │
│  L0 Claim      │ append-only, immutable, provenance         │
│                                                             │
│  ═══════════════════════════════════════════════════════   │
│  Emits: Meta-claims (observations about epistemic state)    │
└─────────────────────────────────────────────────────────────┘
```

---

## The 6-Layer Stack

### L0: ClaimObservation (Atomic)
Raw claims with provenance. **Append-only, never modified.**

```python
@dataclass
class Claim:
    id: str
    text: str
    source: str
    embedding: Optional[List[float]]
    entities: Set[str]
    anchor_entities: Set[str]  # PERSON, ORG - high specificity
    timestamp: Optional[datetime]

    # Question Key (q1/q2 pattern) — L1 indexing primitive
    question_key: Optional[str]      # "death_count", "origin_location", etc.
    extracted_value: Optional[Any]   # 13, "Floor 8", etc.
    value_unit: Optional[str]        # "people", "floors", etc.
    has_update_language: bool        # "rises to", "now at", "updated"
    is_monotonic: Optional[bool]     # True for counts that only increase
```

**Question Key (q1/q2 pattern):**
- `question_key` identifies WHICH QUESTION the claim answers
- Claims only relate if they answer the SAME question
- This is the indexing primitive for L1 proposition formation

Examples:
- "13 dead in fire" → `question_key="death_count"`, `value=13`
- "Death toll rises to 17" → `question_key="death_count"`, `value=17`, `has_update=true`
- "Fire started on Floor 8" → `question_key="origin_floor"`, `value=8`

### L1: Proposition (Deduplicated)
Claims that assert the same fact are grouped. Version chains track evolution.

**Identity Relations** (claim → claim):
- `CONFIRMS`: Same fact, different source
- `REFINES`: Adds detail to same fact
- `SUPERSEDES`: Updates/corrects with temporal marker
- `CONFLICTS`: Contradicts (still same fact—disagreement recorded)
- `UNRELATED`: Different facts

### L2: Surface (Proposition Bundle)
Connected component of claims via **identity edges only**.

```python
@dataclass
class Surface:
    id: str
    claim_ids: Set[str]

    # Computed from claims
    centroid: Optional[List[float]]
    entropy: float
    sources: Set[str]
    entities: Set[str]
    anchor_entities: Set[str]

    # IDENTITY edges (formed this surface)
    internal_edges: List[Tuple[str, str, Relation]]

    # ABOUTNESS edges (soft links to other surfaces)
    about_links: List[AboutnessLink]
```

### L3: Event (Surface Cluster)
Groups surfaces by **aboutness edges** (soft, graded).

Aboutness signals (2-of-3 required):
1. **Strong anchor overlap** (IDF-weighted, hub penalty)
2. **Semantic similarity** (centroid cosine > 0.5)
3. **Entity overlap** (IDF-weighted)

```python
@dataclass
class AboutnessLink:
    target_id: str
    score: float
    evidence: Dict  # breakdown of signals
```

### L4: Narrative (Not Yet Implemented)
Edges between events:
- **Temporal**: BEFORE / AFTER / OVERLAPS (rule-based)
- **Causal**: CAUSES / ENABLES (rule/LLM-sparse)
- **Discourse**: BACKGROUND / ELABORATES (LLM)

### L5: Meaning (Not Yet Implemented)
Interpretive layer—never merges facts:
- **Frame**: human rights, public safety, geopolitics...
- **Stake**: harm, rights, money, legitimacy...
- **Question**: "What happened?", "Who is responsible?"

---

## The 6 Invariants

### Invariant 1: L0 Immutability
> Claims are append-only, never deleted, never modified.

- Cold storage allowed, but lineage preserved
- Source/provenance is sacred

### Invariant 2: Parameter Versioning
> Parameters are append-only with full provenance.

```python
@dataclass
class ParameterChange:
    id: str
    timestamp: datetime
    parameter: str
    old_value: Any
    new_value: Any
    actor: str              # "system:tension_detector", "human:operator@xyz"
    trigger: Optional[str]  # meta-claim ID that prompted this
    rationale: str
    topology_version: str
    affects_layers: List[str]

@dataclass
class Parameters:
    version: int = 1
    identity_confidence_threshold: float = 0.5
    hub_max_df: int = 3
    aboutness_min_signals: int = 2
    aboutness_threshold: float = 0.35
    high_entropy_threshold: float = 0.6
    changes: List[ParameterChange]
```

**Key insight**: `(L0, params@version)` → deterministic L1-L5. Full audit trail.

### Invariant 3: Identity/Aboutness Separation
> L2 uses identity edges. L3 uses aboutness edges. Never mixed.

| Layer | Edge Type | Purpose | Can Merge? |
|-------|-----------|---------|------------|
| L2 Surface | Identity | Same fact | Yes (connected components) |
| L3 Event | Aboutness | Same event | No (soft clustering only) |

**The one rule**: If you violate this, you'll either percolate into topic blobs or fragment forever.

### Invariant 4: Derived State Purity
> L1-L5 = f(L0, parameters). No external mutation.

- Operational layers can only affect derived state through:
  1. Appending new L0 claims
  2. Updating parameters (with provenance)
  3. Triggering versioned recompute
- No direct mutation of L1-L5

### Invariant 5: Stable Core Relations
> {CONFIRMS, REFINES, SUPERSEDES, CONFLICTS, UNRELATED}

- Domain differences handled via extraction, not new relations
- Entity sense IDs for disambiguation (not domain-specific relations)

### Invariant 6: Meta-Claims Are Observations
> Emitted about epistemic state, never injected as world-claims.

```python
MetaClaimType = Literal[
    "high_stakes_low_evidence",  # → verification bounty
    "unresolved_conflict",        # → adjudication task
    "single_source_only",         # → corroboration request
    "high_entropy_surface",       # → investigation prompt
    "bridge_node_detected",       # → potential split candidate
    "stale_event",                # → decay candidate
]

@dataclass
class MetaClaim:
    id: str
    type: MetaClaimType
    target_id: str
    target_type: str  # "claim", "surface", "event"
    evidence: Dict
    generated_at: datetime
    params_version: int
    resolved: bool
    resolution: Optional[str]
```

---

## Dataflow

```
┌─────────────────────────────────────────────────────────────┐
│                    OPERATIONAL LAYER                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                │                ▼
 ┌─────────────────┐       │       ┌─────────────────────────┐
 │ New L0 Claims   │       │       │ ParameterChange         │
 │ (append-only)   │       │       │ (versioned, attributed) │
 └────────┬────────┘       │       └────────────┬────────────┘
          │                │                    │
          ▼                │                    ▼
┌─────────────────────────────────────────────────────────────┐
│                     EPISTEMIC CORE                          │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ L0: Claims (append-only log)                        │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Parameter Log (append-only, versioned)              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  L1-L5 = f(L0, parameters@version)                         │
│                                                             │
│  Any (L0, params_version) pair → deterministic topology    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ emits
                           ▼
                  ┌─────────────────┐
                  │  Meta-claims    │
                  │  (observations  │
                  │   about state)  │
                  └─────────────────┘
```

---

## Implementation

### Core Classes

| Class | Layer | Purpose |
|-------|-------|---------|
| `Claim` | L0 | Atomic observation with provenance |
| `Surface` | L2 | Identity component (same-fact bundle) |
| `AboutnessLink` | L2→L3 | Soft edge between surfaces |
| `Event` | L3 | Aboutness cluster of surfaces |
| `Parameters` | - | Versioned computation parameters |
| `ParameterChange` | - | Audit record for parameter updates |
| `MetaClaim` | - | Observation about epistemic state |
| `EmergenceEngine` | - | Orchestrates L0 → L2 → L3 |

### Key Methods

```python
class EmergenceEngine:
    # L0 → L2 (identity edges only)
    async def add_claim(claim: Claim) -> Dict
    def compute_surfaces() -> List[Surface]

    # L2 → L3 (aboutness edges only)
    def compute_surface_aboutness() -> List[Tuple]
    def compute_events() -> List[Event]

    # Meta-claims
    def detect_tensions() -> List[MetaClaim]
    def get_unresolved_meta_claims() -> List[MetaClaim]
    def resolve_meta_claim(id, resolution, actor) -> MetaClaim
```

### Candidate Generation (Clean Two-Path Design)

```
┌─────────────────────────────────────────────────────────────┐
│                      add_claim()                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ PATH A: Claims WITH question_key                     │   │
│  │                                                      │   │
│  │   question_key bucket → rule-based classification   │   │
│  │                                                      │   │
│  │   Same value → CONFIRMS                             │   │
│  │   Different + update language → SUPERSEDES          │   │
│  │   Different, no update → CONFLICTS                  │   │
│  │                                                      │   │
│  │   (No LLM needed)                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ PATH B: Claims WITHOUT question_key                  │   │
│  │                                                      │   │
│  │   embedding gate → LLM classification               │   │
│  │                                                      │   │
│  │   Embedding is ONLY for gating (reducing pairs)     │   │
│  │   Classification always via LLM                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key principle**: Embedding is for GATING, question_key is for INDEXING.
Never use embedding similarity for classification decisions.

### Hub Penalty (IDF Weighting)

Prevents common anchors (Trump, Hong Kong) from gluing unrelated events:

```python
anchor_weight(a) = 0 if df(a) > hub_max_df else log(1 + N/df(a))
```

### 2-of-3 Aboutness Constraint

For event-level links, require 2 signals among:
1. Strong anchor evidence (IDF-weighted > 0.3)
2. Semantic similarity (centroid cosine > 0.5)
3. Entity overlap (IDF-weighted > 0.3)

---

## Operational Layer Concerns (Outside Core)

These wrap the core but do NOT leak into L0-L5 semantics:

| Concern | Approach |
|---------|----------|
| **Forgetting** | Attention/activation decay on L1-L5; L0 cold-stored, lineage preserved |
| **Multi-domain** | Better extraction + entity sense IDs; core relations stay stable |
| **Distribution** | Each shard runs same L0-L5 locally; global = federated summaries with provenance |
| **Active seeking** | Meta-claims → tasks/bounties; outside truth-maintenance semantics |

---

## Validation

### Jaynes Alignment
Single-source claims have higher entropy than multi-source:
```python
entropy(n) = max(0.15, 0.8 / sqrt(n))
```

### Test Suite
25 tests validating all 6 invariants:
```bash
docker exec herenews-app python -m pytest /app/test_eu/test_invariants.py -v
```

### Metrics to Track
- **Hub bleed rate**: % of cross-event merges from top-k anchors
- **Stability under shuffle**: cluster assignments across random orderings
- **Gate recall / LLM recall**: of true same-event pairs, how many are candidates?

---

## File Structure

```
test_eu/
├── core/
│   ├── epistemic_unit.py    # Core implementation (Claim, Surface, Event, etc.)
│   ├── kernel.py            # LLM-backed classification
│   └── topology.py          # Pure mathematical structures
├── test_invariants.py       # 25 tests for all 6 invariants
└── EPISTEMIC_CORE_ARCHITECTURE.md  # This document
```

---

## Summary

The epistemic core provides a clean contract:

1. **L0 is sacred** — append-only, immutable provenance
2. **Parameters are versioned** — full audit trail for reproducibility
3. **Identity ≠ Aboutness** — never mix these graphs
4. **Derived state is pure** — L1-L5 = f(L0, params)
5. **Relations are stable** — domain differences via extraction
6. **Meta-claims are observations** — emit, don't inject

This separation enables a 24/7 epistemic engine where:
- Truth emerges from convergent evidence geometry
- Operational concerns (forgetting, federation, seeking) stay outside semantics
- Full audit trail from any snapshot back to source claims
