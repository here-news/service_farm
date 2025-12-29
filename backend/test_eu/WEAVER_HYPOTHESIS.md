# Epistemic Weaver Hypothesis

## Vision

The kernel evolves into a **Weaver** - a recursive force that operates at all scales:
- Claims → Nodes (beliefs)
- Nodes → Surfaces (coherent clusters)
- Surfaces → Events (real-world incidents)

Same operation at each level: `classify(a, b) → {relation, weight}`

## Validated Architecture: 2-Tier Topology

Based on epistemology/KR analysis, we adopt a **2-tier topology**:

### Tier 1: Claim Graph (Truth Maintenance)
**Purpose:** Same-fact version control
**Relations:** CONFIRMS, REFINES, SUPERSEDES, CONFLICTS, NOVEL
**LLM task:** "Are these about the same fact?"

### Tier 2: Event Graph (Narrative Structure)
**Purpose:** Group claims by real-world incident/case
**Relations:** ABOUT (claim → event), BEFORE/AFTER/OVERLAPS (event → event)
**Method:** Entity coref + embeddings + temporal rules (minimal LLM)

```
┌─────────────────────────────────────────────────────────┐
│                    EVENT GRAPH                          │
│  ┌─────────┐   BEFORE   ┌─────────┐   CAUSES           │
│  │ Event A │───────────►│ Event B │──────────►...      │
│  └────┬────┘            └────┬────┘                    │
│       │ ABOUT                │ ABOUT                    │
│  ┌────┴────────────┐    ┌────┴────────────┐            │
│  │   CLAIM GRAPH   │    │   CLAIM GRAPH   │            │
│  │ ┌───┐   ┌───┐   │    │ ┌───┐   ┌───┐   │            │
│  │ │ C │───│ C │   │    │ │ C │───│ C │   │            │
│  │ └───┘   └───┘   │    │ └───┘   └───┘   │            │
│  │ CONFIRMS/REFINES│    │ CONFIRMS/REFINES│            │
│  └─────────────────┘    └─────────────────┘            │
└─────────────────────────────────────────────────────────┘
```

## Why 2-Tier Works

| Concern | Solution |
|---------|----------|
| Narrative coherence | Event nodes group related claims |
| Temporal chains | Allen's interval algebra at event level |
| Causal/implication | Rule-based (arrested→detained), not LLM |
| LLM complexity | LLM only does "same fact?" classification |
| Orthogonality | Truth ⟂ Narrative ⟂ Temporal |

## Current Epistemic Relations (Tier 1)

| Relation | Meaning | Edge Effect |
|----------|---------|-------------|
| CONFIRMS | Same fact, additional source | Strengthen node, add attestation |
| REFINES | Same fact, more precision | Update node text, keep history |
| SUPERSEDES | Same fact, newer value | Replace node value, mark old |
| CONFLICTS | Same fact, contradictory | Create conflict record |
| NOVEL | Different fact | Create new node |

**Verdict:** These 5 are **complete for same-fact truth maintenance**.
Narrative/temporal/causal belong in Tier 2, not here.

## Event Graph Relations (Tier 2)

| Relation | Level | Method |
|----------|-------|--------|
| ABOUT | Claim → Event | Entity coref + embedding clustering |
| BEFORE/AFTER/OVERLAPS | Event → Event | Temporal extraction (rule-based) |
| CAUSES/ENABLES | Event → Event | KB rules or sparse LLM |

## Domain Model Integration

### Kernel → Domain Model Mapping

| Kernel Concept | Domain Model | Storage |
|----------------|--------------|---------|
| `Node` | `Claim` | Neo4j (node) + PostgreSQL (embedding) |
| `Edge` | CORROBORATES/REFINES/CONFLICTS | Neo4j (relationship) |
| `Surface` | Event claim group | Event-[INTAKES]->Claim |
| `Conflict` | CONFLICTS relationship | Neo4j claim-claim edge |

### Repository Interface

The kernel should accept repository interfaces for database operations:

```python
class KernelRepository(Protocol):
    """Interface for kernel persistence operations."""

    async def find_similar_claims(
        self,
        embedding: List[float],
        limit: int = 10,
        exclude_ids: List[str] = None
    ) -> List[Tuple[Claim, float]]:
        """pgvector similarity search - returns (claim, similarity) pairs."""
        ...

    async def create_relation(
        self,
        source_claim_id: str,
        target_claim_id: str,
        relation: str,  # CONFIRMS, REFINES, SUPERSEDES, CONFLICTS
        confidence: float
    ) -> None:
        """Create claim-claim relationship in Neo4j."""
        ...

    async def get_claim_with_entities(
        self,
        claim_id: str
    ) -> Claim:
        """Hydrate claim with entities from Neo4j."""
        ...
```

### Existing Repository Methods (Already Available)

| Operation | Method | Location |
|-----------|--------|----------|
| Find similar claims | `find_similar(embedding, limit, exclude_ids)` | `ClaimRepository` |
| Store embedding | `store_embedding(claim_id, embedding)` | `ClaimRepository` |
| Create corroboration | `create_corroboration(claim_id, target_id, sim)` | `ClaimRepository` |
| Get entities | `get_entities_for_claim(claim_id)` | `ClaimRepository` |
| Event by embedding | `get_candidate_events_by_embedding(emb, threshold)` | `EventRepository` |

### Guardrails (From External Review)

1. **ABOUT linking**: Asymmetric (claim → event), gate merges with hard entity overlap
2. **Event identity**: `Event := (entities, time_window, location)` as case anchor
3. **Temporal rules**: Extract at claim level, Allen relations only event-to-event
4. **pgvector**: Candidate generator only - still validate with Tier-1 LLM

### Validated Thresholds

```python
# Tier 1 (Kernel) - kernel.py
SIM_THRESHOLD = 0.85       # Skip LLM only for very high similarity
WEAK_SIM_THRESHOLD = 0.50  # LLM with hint for medium similarity

# Tier 2 (Weaver) - event_weaver.py
ENTITY_OVERLAP_THRESHOLD = 0.1   # At least 1 shared entity
EMBEDDING_SIM_THRESHOLD = 0.4    # Embedding similarity gate
TIME_WINDOW_DAYS = 14            # Temporal proximity
```

Quality check results:
- 18 distinct facts from 30 claims (conservative, accurate)
- CONFIRMS only for truly same facts (e.g., "128 fire trucks" ↔ "200 fire trucks")
- REFINES/SUPERSEDES for updates (e.g., death toll revisions)

## Implementation Path

### Phase 1: Strengthen Tier 1 ✓
- [x] 5 relations for same-fact comparison
- [x] Confidence threshold gating
- [x] pgvector for similarity (replace in-kernel O(n))
- [x] Inject ClaimRepository for persistence

### Phase 2: Build Tier 2 ✓
- [x] Event node creation (EventWeaver with entity clustering)
- [x] ABOUT linking (claim → event via entity overlap)
- [x] Temporal relation extraction (Allen's interval algebra)
- [x] Post-merge step for shared-entity events

### Phase 3: Recursive Weaver
- [ ] Same weave() operation at claim and event levels
- [ ] Surfaces emerge from claim graph
- [ ] Meta-surfaces emerge from event graph
- [ ] Integrate with EventRepository for persistence

## References

- **NLI:** Entailment/contradiction/neutral for same-fact logic
- **RST/SDRT:** Discourse relations (elaboration, background, cause)
- **Allen's Interval Algebra:** Temporal relations
- **Event Calculus / FrameNet:** Event modeling

## Validated Results (130 claims from Wang Fuk Court Fire)

### Tier 1: Kernel (Same-Fact)
```
Claims processed: 130
Nodes created:    4
Compression:      32.5x
LLM calls:        3 (97% skipped via pgvector)
Coherence:        0.98
```

### Tier 2: Weaver (Same-Event)
```
Claims processed: 130
Event clusters:   16 (after merging)
Main cluster:     109 claims (84% captured)
Compression:      8.1x
```

### Combined Architecture
```
Input: 130 claims about Hong Kong fire
  │
  ├─► Tier 1 (Kernel): 4 distinct facts
  │     - "Fire kills N people"
  │     - "Firefighters deployed"
  │     - "Building was undergoing renovation"
  │     - "Investigation launched"
  │
  └─► Tier 2 (Weaver): 16 event clusters
        - Main: Wang Fuk Court Fire (109 claims)
        - Side: Other mentions (21 claims)
```

## Success Criteria

- [x] 5 relations complete for same-fact (validated)
- [x] Event nodes group narratively related claims (84% capture rate)
- [x] Temporal chains emerge at event level (Allen relations working)
- [ ] System supports: narrative gen, Q&A, fact-check
