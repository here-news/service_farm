# Migration Plan: case_builder → story_builder

**STATUS: COMPLETE** (2026-01-05)

## Results

- 89 stories formed from 597 incidents
- No mega-cases (largest: 32 incidents, previously 500+)
- All 361 tests pass
- case_builder.py deprecated (kept for EntityCase type)

## Overview

Migrate `canonical_worker.py` from `PrincipledCaseBuilder` to `StoryBuilder`, archive legacy tests, and retest with real topology.

## Phase 1: Prepare StoryBuilder for Production

### 1.1 Fill Implementation Gaps

**story_builder.py gaps identified:**

| Gap | Current State | Action |
|-----|---------------|--------|
| Blocked facet detection | TODO (empty) | Implement schema-driven detection |
| Constraint ledger loading | TODO line 554 | Wire up ledger persistence |
| CompleteStory → Case conversion | Missing | Add `to_case()` method |

### 1.2 Add Conversion Method

```python
# In story_builder.py, add to CompleteStory:
def to_entity_case(self) -> EntityCase:
    """Convert CompleteStory to EntityCase for API compatibility."""
    return EntityCase(
        anchor_entity=self.spine.primary_entity,
        anchor_entity_kind=self.spine.entity_kind,
        incident_ids=self.core_a_ids | self.core_b_ids,
        periphery_ids=self.periphery_incident_ids,
        companion_entities=self._extract_companions(),
        hub_entities=self.hub_entities,
    )
```

## Phase 2: Migrate canonical_worker.py

### 2.1 Import Changes

```python
# OLD:
from reee.builders import (
    PrincipledCaseBuilder,
    CaseCore,
    EntityCase,
)

# NEW:
from reee.builders.story_builder import (
    StoryBuilder,
    CompleteStory,
)
from reee.builders import EntityCase  # Keep for API compatibility
```

### 2.2 Builder Usage Changes

```python
# OLD (line ~814):
builder = PrincipledCaseBuilder(incidents, constraint_ledger)
cases = builder.build_from_incidents()

# NEW:
builder = StoryBuilder(incidents, session)
stories = builder.build_stories()
cases = [story.to_entity_case() for story in stories]
```

### Key Behavioral Differences

| Aspect | case_builder | story_builder |
|--------|--------------|---------------|
| Core membership | k=2 motif recurrence | Spine + 2 structural witnesses |
| Hub threshold | 30% | 20% |
| Decision audit | constraint_ledger | membrane_decisions |
| Facet tracking | None | presence/coverage/noise/blocked |

## Phase 3: Archive Legacy Tests

### 3.1 Files to Archive (42 tests)

Move to `backend/reee/tests/archive/`:

- `test_case_builder_hub_suppression.py` (11 tests)
- `test_case_builder_chain.py` (10 tests)
- `test_case_builder_motif_recurrence.py` (10 tests)
- `test_entity_case.py` (11 tests)

### 3.2 Critical Assertions to Port

These assertions must exist in story_builder tests:

1. **Hub suppression**: Entities >20% cannot create core edges
2. **Anti-percolation**: No mega-case formation from hub linkage
3. **Spine membership**: Core-A is automatic, Core-B needs 2 witnesses
4. **Decision audit**: All decisions recorded with provenance

## Phase 4: Flush & Retest

### 4.1 Flush Neo4j Topology

```cypher
// Clear L4 cases
MATCH (c:Case) DETACH DELETE c;

// Clear case-incident relationships
MATCH ()-[r:BELONGS_TO_CASE]->() DELETE r;
```

### 4.2 Rebuild and Run

```bash
./rebuild_workers.sh
# or
docker-compose build canonical_worker
docker-compose up -d canonical_worker
```

### 4.3 Trigger Real Weave

```bash
# Queue canonical weave job
docker exec herenews-app python -c "
from workers.canonical_worker import CanonicalWorker
worker = CanonicalWorker()
worker.run_full_weave()
"
```

## Phase 5: Deprecate case_builder.py

After successful migration and testing:

1. Remove `case_builder.py`
2. Update `__init__.py` exports
3. Remove archived tests from CI

## Checklist

- [x] Implement `CompleteStory.to_entity_case()`
- [x] Migrate canonical_worker imports
- [x] Update builder instantiation
- [x] Archive 4 test files (42 tests → tests/archive/)
- [x] Port critical assertions to story_builder tests
- [x] Flush neo4j cases
- [x] Rebuild workers
- [x] Run real weave (89 stories, no mega-cases)
- [x] Verify case formation quality (largest: 32 incidents)
- [x] Deprecate case_builder.py (kept for EntityCase)
