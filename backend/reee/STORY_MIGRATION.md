# Story Migration Advisory

## Summary

The REEE kernel now exports a unified `Story` type for L3/L4 output. This document advises downstream engineers on required changes to API endpoints, domain models, and Neo4j labels.

## Kernel Changes (Completed)

### New Type: `Story`

Location: `backend/reee/types.py`

```python
from reee import Story, StoryScale

# Create incident-scale story
story = Story(
    id="temp",
    scale="incident",  # or "case"
    anchor_entities={"John Lee", "Hong Kong"},
    time_start=datetime(2025, 1, 15),
    title="Hong Kong Policy Announcement",
)

# Generate stable ID (deterministic across rebuilds)
story.compute_scope_signature()  # → "story_a3daa6544b10"
story.generate_stable_id()       # Sets id = scope_signature
```

### Key Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | str | Ephemeral or stable ID |
| `scale` | `"incident"` \| `"case"` | L3 vs L4 |
| `scope_signature` | str | Deterministic hash for rebuild stability |
| `title` | str | Human-readable title |
| `description` | str | Summary |
| `primary_entities` | List[str] | Top entities for display |
| `anchor_entities` | Set[str] | Identity-defining entities |
| `surface_ids` | Set[str] | L2 surfaces |
| `incident_ids` | Set[str] \| None | L3 incidents (case only) |
| `time_start` / `time_end` | datetime | Temporal bounds |
| `surface_count` / `source_count` / `claim_count` | int | Stats |
| `incident_count` | int | Number of L3 incidents (case only) |
| `justification` | EventJustification | Membrane proof |

### Conversion from Internal Event

```python
from reee import Story, Event

# Internal computation produces Event
event = compute_incident_events(surfaces)  # Returns Event

# Convert to Story for API
story = Story.from_event(event, scale="incident", title="...")
```

---

## Required Downstream Changes

### 1. Domain Model: Add `models/domain/story.py`

Create a new domain model that mirrors the kernel `Story` type:

```python
# backend/models/domain/story.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Set, Optional, Literal

@dataclass
class Story:
    """
    API-facing story object. Unifies incident/case for downstream consumption.
    """
    id: str
    scale: Literal["incident", "case"]
    scope_signature: str

    title: str
    description: str
    primary_entities: List[str]

    surface_count: int
    source_count: int
    incident_count: int = 0  # Only for case scale

    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None

    @staticmethod
    def from_kernel(kernel_story: 'reee.Story') -> 'Story':
        """Convert kernel Story to domain Story."""
        return Story(
            id=kernel_story.id,
            scale=kernel_story.scale,
            scope_signature=kernel_story.scope_signature,
            title=kernel_story.title,
            description=kernel_story.description,
            primary_entities=kernel_story.primary_entities,
            surface_count=kernel_story.surface_count,
            source_count=kernel_story.source_count,
            incident_count=kernel_story.incident_count,
            time_start=kernel_story.time_start,
            time_end=kernel_story.time_end,
        )
```

### 2. API Endpoints: Rename/Add Routes

**Option A: Add new endpoints (non-breaking)**

```python
# backend/api/stories.py

@router.get("/stories")
async def list_stories(
    scale: Optional[Literal["incident", "case"]] = None,
    offset: int = 0,
    limit: int = 20,
):
    """List stories (unified incident/case view)."""
    ...

@router.get("/stories/{story_id}")
async def get_story(story_id: str):
    """Get single story by ID or scope_signature."""
    ...
```

**Option B: Deprecate old endpoints**

```python
# backend/api/events.py

@router.get("", deprecated=True)
async def list_events(...):
    """DEPRECATED: Use /api/stories?scale=case instead."""
    ...
```

### 3. Response Models: Add `StorySummary`, `StoryDetail`

```python
# backend/api/stories.py

class StorySummary(BaseModel):
    id: str
    scale: Literal["incident", "case"]
    scope_signature: str  # For client-side caching/dedup
    title: str
    description: str
    primary_entity: str
    time_start: Optional[str] = None
    source_count: int
    surface_count: int
    incident_count: Optional[int] = None  # case only

class StoryDetail(StorySummary):
    primary_entities: List[str]
    surfaces: Optional[List[SurfaceSummary]] = None
    incidents: Optional[List[StorySummary]] = None  # case only, nested
    inquiries: Optional[List[InquirySummary]] = None
    justification: Optional[dict] = None  # Membrane proof
```

### 4. Neo4j Labels: Migrate `:Event` → `:Story`

**Phase 1: Add new label alongside old**

```cypher
// In workers, when persisting Story objects:
MERGE (s:Story {scope_signature: $scope_signature})
SET s.id = $id,
    s.scale = $scale,
    s.title = $title,
    ...
```

**Phase 2: Migrate existing data**

```cypher
// Add :Story label to existing Incidents
MATCH (i:Incident)
SET i:Story, i.scale = 'incident'

// Add :Story label to existing Cases
MATCH (c:Case)
SET c:Story, c.scale = 'case'

// Mark legacy Events
MATCH (e:Event)
WHERE NOT e:Story
SET e.kind = 'legacy'
```

**Phase 3: Query by :Story**

```cypher
// New queries use :Story
MATCH (s:Story {scale: 'incident'})
WHERE s.time_start > $since
RETURN s

// Old queries still work during migration
MATCH (e:Event)
WHERE e.kind IS NULL OR e.kind <> 'legacy'
RETURN e
```

### 5. Repository: Add `StoryRepository`

```python
# backend/repositories/story_repository.py

class StoryRepository:
    async def get_by_id(self, story_id: str) -> Optional[Story]:
        """Get by id or scope_signature."""
        ...

    async def list_by_scale(
        self,
        scale: Literal["incident", "case"],
        offset: int = 0,
        limit: int = 20,
    ) -> List[Story]:
        ...

    async def upsert(self, story: Story) -> str:
        """
        MERGE by scope_signature (stable ID).
        Returns the story ID.
        """
        await self.neo4j.execute("""
            MERGE (s:Story {scope_signature: $scope_signature})
            SET s.id = $id,
                s.scale = $scale,
                s.title = $title,
                ...
        """, story.to_dict())
        return story.id
```

---

## Stability Guarantees

### Scope Signature Computation

The `scope_signature` is deterministic based on:

1. **Scale**: `"incident"` or `"case"`
2. **Sorted anchor entities** (top 10)
3. **Time bin**:
   - Incident: Week bin (YYYY-Www)
   - Case: Month bin (YYYY-MM)
4. **Params version**

Same inputs → Same signature → Same ID across rebuilds.

### ID Format

```
story_<12char_sha256_prefix>
```

Example: `story_a3daa6544b10`

### Worker Changes Required

Workers that currently DELETE and recreate events should instead:

```python
# OLD (breaks links)
await neo4j.execute("DETACH DELETE e:Event")
await neo4j.execute("CREATE (e:Event {...})")

# NEW (stable)
story.compute_scope_signature()
await neo4j.execute("""
    MERGE (s:Story {scope_signature: $scope_signature})
    SET s.id = $id, s.title = $title, ...
""", story.to_dict())
```

---

## Worker Alignment Status

### `rebuild_topology.py` - NEEDS CHANGES

**Location:** `backend/reee/canonical/rebuild_topology.py`

**Current Issues:**

1. **DETACH DELETE pattern** (lines 139-142):
   ```python
   # PROBLEMATIC: Breaks external links
   await neo4j._execute_write('MATCH (e:Event) DETACH DELETE e')
   await neo4j._execute_write('MATCH (s:Surface) DETACH DELETE s')
   ```

2. **Ephemeral IDs** (via `generate_id('event')`):
   - Generates random IDs like `ev_a1b2c3d4`
   - Different every rebuild → breaks inquiry links

3. **Uses Event, not Story**:
   - `event_builder.py` creates internal `Event` objects
   - Never converts to `Story` with stable `scope_signature`

**Required Changes:**

```python
# In rebuild_topology.py, replace save_events_to_neo4j():

async def save_stories_to_neo4j(
    events: Dict[str, Event],
    surface_id_mapping: Dict[str, str],
    neo4j: Neo4jService
) -> Dict[str, str]:
    """
    Convert Events to Stories and persist with stable IDs.
    Uses MERGE on scope_signature for rebuild stability.
    """
    from reee import Story

    id_mapping = {}

    for reee_id, event in events.items():
        # Convert Event → Story
        story = Story.from_event(event, scale="incident")
        story.compute_scope_signature()
        story.generate_stable_id()  # Sets id = scope_signature

        id_mapping[reee_id] = story.id

        # MERGE by scope_signature (stable across rebuilds)
        await neo4j._execute_write('''
            MERGE (s:Story {scope_signature: $scope_signature})
            SET s:Incident,  // Keep :Incident for backwards compat
                s.id = $id,
                s.scale = 'incident',
                s.title = $title,
                s.surface_count = $surface_count,
                s.claim_count = $claim_count,
                s.source_count = $source_count,
                s.event_start = $event_start,
                s.event_end = $event_end,
                s.updated_at = datetime()
        ''', story.to_dict())

        # ... edges as before, using story.id ...

    return id_mapping
```

**Do NOT delete old topology first** - use MERGE to update in place.

---

### `principled_weaver.py` - MOSTLY ALIGNED

**Location:** `backend/workers/principled_weaver.py`

**What's Already Correct:**

1. **L2 routing by question_key** ✅
   - Lines 640-652: Routes claims by `question_key`
   - Creates surfaces with `question_key` property
   - Matches kernel L2 identity invariant

2. **L3 membrane logic** ✅
   - Lines 865-947: `_route_to_incident()` checks anchor overlap + companion compatibility
   - Bridge immunity implemented (lines 896-912)
   - Emits `bridge_blocked` meta-claims

3. **Uses MERGE for persistence** ✅
   - Lines 951-965: `_persist_incident()` uses MERGE
   - Lines 1162-1189: `_persist_case()` uses MERGE

**Remaining Issues:**

1. **Creates `:Incident` and `:Case`, not `:Story`**:
   - Need to add `:Story` label with `scale` property
   - OR convert to `Story` objects before persistence

2. **No `scope_signature`**:
   - Incidents/Cases use `generate_id()` (random IDs)
   - Need to compute deterministic signatures

**Required Changes (minimal):**

```python
# In _persist_incident():
await self.neo4j._execute_write("""
    MERGE (i:Incident {id: $id})
    SET i:Story,  // ADD: Story label for unified queries
        i.scale = 'incident',
        i.scope_signature = $scope_sig,  // ADD: for stability
        i.anchor_entities = $anchors,
        ...
""", {
    ...
    'scope_sig': self._compute_incident_signature(incident),
})

# In _persist_case():
await self.neo4j._execute_write("""
    MERGE (c:Case {id: $id})
    SET c:Story,  // ADD: Story label
        c.scale = 'case',
        c.scope_signature = $scope_sig,  // ADD
        ...
""", ...)
```

**Add helper method:**

```python
def _compute_incident_signature(self, incident: L3Incident) -> str:
    """Compute stable scope signature for incident."""
    import hashlib
    sorted_anchors = sorted(incident.anchor_entities)[:10]
    time_bin = "unknown"
    if incident.time_start:
        time_bin = f"{incident.time_start.year}-W{incident.time_start.isocalendar()[1]:02d}"
    sig_parts = ["incident", ",".join(sorted_anchors), time_bin]
    sig_hash = hashlib.sha256("|".join(sig_parts).encode()).hexdigest()[:12]
    return f"story_{sig_hash}"
```

---

### `weaver_worker.py` (Archive) - DEPRECATED

**Location:** `backend/workers/archive/weaver_worker.py`

**Status:** Legacy, should NOT be used.

**Issues:**
- Uses embedding similarity for surface linking (violates L2=question_key invariant)
- No question_key extraction
- DETACH DELETE pattern

**Action:** Keep in archive. Do not resurrect.

---

### `event_builder.py` - INTERNAL ONLY

**Location:** `backend/reee/canonical/event_builder.py`

**Status:** Internal computation module, produces `Event` objects.

**Required Changes:**

The caller (`rebuild_topology.py`) must convert `Event` → `Story`:

```python
# After event_builder produces events:
for event in event_result.events.values():
    story = Story.from_event(event, scale="incident")
    story.compute_scope_signature()
    # Then persist story...
```

No changes needed to `event_builder.py` itself.

---

## Migration Timeline Suggestion

1. **Week 1**: Add `Story` domain model, `StoryRepository`, `/api/stories` endpoint
2. **Week 2**: Update `principled_weaver.py` to add `:Story` label and `scope_signature`
3. **Week 3**: Update `rebuild_topology.py` to use MERGE instead of DELETE+CREATE
4. **Week 4**: Frontend switches to `/api/stories`
5. **Week 5**: Deprecate `/api/events`, clean up legacy labels

---

## Questions for API/Data Model Engineer

1. **Frontend expectations**: What fields does the frontend currently use from `/api/events`? Any breaking changes needed?

2. **Inquiry links**: Inquiries reference `source_event`. Should this become `source_story`? Or keep as-is with ID compatibility?

3. **WebSocket/SSE**: Are there real-time event streams that need updating?

4. **Caching**: Frontend can now use `scope_signature` for cache keys. Is this useful?

---

## Files Changed in Kernel

| File | Change |
|------|--------|
| `backend/reee/types.py` | Added `Story`, `StoryScale` dataclasses |
| `backend/reee/__init__.py` | Reorganized exports, added `Story` to public API |
| `backend/reee/tests/test_canonical_stability.py` | New test file (21 tests) |

## Test Coverage

Run kernel tests:
```bash
docker exec herenews-app python -m pytest backend/reee/tests/test_canonical_stability.py -v
```

All 21 tests verify:
- Scope signature determinism
- Time bin stability (week for incident, month for case)
- Scale affects signature
- Serialization roundtrip
- Edge cases (no time, empty anchors, >10 anchors)
