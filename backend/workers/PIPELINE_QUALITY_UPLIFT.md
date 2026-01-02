# Pipeline Quality Uplift: KnowledgeWorker → WeaverWorker

## Problem Statement

The current pipeline loses information at critical handoff points, blocking downstream analysis:

| Lost Signal | Where | Impact |
|-------------|-------|--------|
| `pub_time` | KnowledgeWorker doesn't propagate to claims | Temporal coverage ~48% |
| Full claim text | ClaimRepository truncates to 500 chars | Typed extraction blocked |
| Entity roles | `who`/`where` stored as mention_ids, not resolved | Generic anchors cause bridging |
| Typed values | No `question_key` extraction | Jaynes inference blocked |

---

## Critical Bug: Anchor ID/Name Mismatch

**This is the single biggest cause of "weak binding" in the current system.**

Current flow:
```
weaver_worker.py:250  → _extract_anchors() returns {canonical_name strings}
weaver_worker.py:309  → Surface.anchor_entities = names (from _extract_anchors)
surface_repository.py:400 → find_candidates_by_anchor() uses claim.entity_ids (UUIDs)
surface_repository.py:344 → Matches against s.anchor_entities (names)
                          → IDs never match names → anchor retrieval returns nothing
```

**Result**: Anchor-based candidate retrieval is effectively broken. Surfaces only match via embedding similarity, not shared anchors.

**Fix**: Anchors must be entity_ids everywhere, not canonical_names.

---

## Design Principles

### 1. Data flows forward, not reconstructed
- Capture at source, propagate downstream
- Don't rely on joins to reconstruct what was known earlier

### 2. Storage matches query patterns
- Neo4j: Graph traversals, relationships, lightweight nodes
- PostgreSQL: Full content, embeddings, typed values

### 3. Fallback chains are explicit
- `event_time` → `reported_time` → `created_at`
- Document which signal is authoritative vs fallback

### 4. Minimal schema changes, maximum impact
- Focus on columns that unlock blocked analysis paths
- Don't add fields "just in case"

### 5. Representations stay consistent across stores
- Entity identity = `entity_id`, not `canonical_name`, everywhere in matching
- Names are for display; IDs are for binding
- **Violated by**: `weaver_worker.py:250` (anchors as names)

### 6. Source diversity = publisher, not page
- Corroboration requires independent sources
- Multiple pages from same publisher ≠ independent confirmation
- `Surface.sources` should be publisher entity_ids or domains, not page_ids

### 7. Time carries basis + precision
- Store whether time came from `event_time` (fact) vs `reported_time` (publication)
- Store precision: hour, day, month, approximate
- Prevents false mode-splits from noon placeholders

### 8. Derived fields are versioned
- Typed extraction and role resolution are derived computations
- Store `extractor_version` to enable deterministic recomputation
- Enables epoch comparison after algorithm improvements

---

## Schema Change: `core.claims`

The existing `core.claims` table needs these columns:

```sql
-- Temporal: reported_time enables fallback when event_time missing
ALTER TABLE core.claims ADD COLUMN reported_time TIMESTAMPTZ;

-- Typed extraction: unlocks Jaynes inference
ALTER TABLE core.claims ADD COLUMN topic_key VARCHAR(100);
ALTER TABLE core.claims ADD COLUMN extracted_value JSONB;

-- Role-aware anchors: reduces bridging pathology
ALTER TABLE core.claims ADD COLUMN who_entity_ids TEXT[];
ALTER TABLE core.claims ADD COLUMN where_entity_ids TEXT[];
```

**Backfill**: `reported_time` can be populated from `pages.pub_time` for existing claims.

---

## Code Changes

### 1. ClaimRepository: Hybrid Storage

**Current**: Neo4j only, text truncated to 500 chars
**Change**: Write to both Neo4j (snippet) and PostgreSQL (full)

```python
# claim_repository.py::create()

# Neo4j: snippet for graph display
c.text = $text[:200]

# PostgreSQL: full content for analysis
INSERT INTO core.claims (id, text, reported_time, ...)
VALUES ($1, $full_text, $reported_time, ...)
```

**Principle**: Neo4j handles graph traversals, PostgreSQL handles content analysis.

### 2. KnowledgeWorker: Propagate Time + Roles

**Current**:
- `event_time` from LLM (often missing)
- Entity roles stored as mention_ids (not resolved)

**Change**:
```python
# _create_claims()

# Always set reported_time from page.pub_time
claim.reported_time = page['pub_time']

# Resolve roles to entity UUIDs
who_entity_ids = [identification.get_entity_id(m) for m in claim_data['who']]
where_entity_ids = [identification.get_entity_id(m) for m in claim_data['where']]
```

**Principle**: Capture resolved IDs at extraction time, not at query time.

### 3. WeaverWorker: Anchor ID Fix + Time Fallback

**Current**:
- `_extract_anchors()` returns names → breaks matching (Principle 5 violation)
- Uses `event_time` only (often NULL)
- All entities become anchors

**Change**:
```python
# _extract_anchors(): Return entity_ids, not names
def _extract_anchors(self, claim: Claim) -> Set[str]:
    # Get role-based IDs from claim metadata
    who_ids = set(claim.metadata.get('who_entity_ids', []))

    anchors = set()
    for entity in claim.entities:
        # Prefer 'who' entities (discriminative)
        if str(entity.id) in who_ids:
            anchors.add(str(entity.id))
        # Include rare entities (high IDF)
        elif entity.mention_count and entity.mention_count < 10:
            anchors.add(str(entity.id))
    return anchors

# _get_claim_time(): Explicit fallback chain
return claim.event_time or claim.reported_time or claim.created_at
```

**Principle**: IDs for binding, names for display. Discriminative anchors reduce bridging.

### 4. Typed Extraction (Optional Stage)

**Current**: No `topic_key` extraction
**Change**: Rule-based extraction after claim creation

```python
# Patterns: death_count, injury_count, money_amount
# Store in PostgreSQL: topic_key, extracted_value (JSONB)
```

**Principle**: Start with high-precision rules (counts), expand later.

---

## Expected Impact

| Metric | Before | After | Mechanism |
|--------|--------|-------|-----------|
| Anchor matching | Broken (0%) | Working | ID consistency fix |
| Temporal coverage | ~48% | ~90%+ | `reported_time` fallback |
| Bridge ratio | High | Lower | Role-aware anchors (IDs) |
| Typed coverage | 0% | 5-30% | Rule-based extraction |
| Full text available | No | Yes | PostgreSQL storage |

---

## Implementation Order (Priority)

### P0: Must-do (unlocks temporal + semantic + anchoring)

1. **Fix anchor ID consistency** (WeaverWorker + SurfaceRepository)
   - `_extract_anchors()` returns entity_ids, not names
   - `Surface.anchor_entities` stores entity_ids
   - `find_candidates_by_anchor()` matches IDs consistently

2. **Propagate reported_time** (KnowledgeWorker + ClaimRepository)
   - Claims get `reported_time = page.pub_time`
   - Weaver uses fallback: `event_time → reported_time → created_at`

3. **Verify centroid loading** (Analysis paths)
   - Centroids exist in `content.surface_centroids`
   - Ensure analysis code joins them (not a storage issue)

### P1: High leverage (reduces bridging)

4. **Resolve entity roles as IDs** (KnowledgeWorker)
   - `who_entity_ids`, `where_entity_ids` stored on claims
   - Enables discriminative anchor selection

5. **Source = publisher, not page** (WeaverWorker)
   - `Surface.sources` = publisher entity_ids or domains
   - Fixes inflated "independent source" counts

### P2: Unlocks Jaynes

6. **Typed extraction** (Hybrid approach)
   - **Inline**: Cheap rules (counts) in KnowledgeWorker
   - **Separate worker**: LLM-based extraction (later)
   - Don't block weaver on typed extraction

7. **Migration + Backfill**
   - Add columns to `core.claims`
   - Backfill `reported_time` from `pages.pub_time`

---

## Typed Extraction: Hybrid Approach

**Inline (KnowledgeWorker)**:
- High-precision rules only: `death_count`, `injury_count`, `money_amount`
- No LLM calls, no throughput impact
- Gets typed coverage from 0% to 5-30%

**Separate Worker (later)**:
- LLM-based extraction for richer schemas
- Normalization, disambiguation
- Doesn't block claims → surfaces pipeline

**Proto-inquiry emission**:
- Depends on typed extraction (or emits `typed_coverage_zero` blocker)
- Weaver doesn't depend on it

---

## Open Questions

1. What's the right snippet length for Neo4j? (200 vs 500 chars)
2. Should surfaces store `time_basis` ('event' vs 'reported')?
3. Version string format for `extractor_version`?
