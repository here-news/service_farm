# Event Worker Overhaul - Preparation Audit

## What We Have (Validated & Working)

### ✅ Data Layer
```
Claims (PostgreSQL):
  ├── Text, confidence, modality, event_time
  ├── metadata JSON:
  │   ├── entity_ids: [uuid1, uuid2, ...]
  │   └── entity_names: ["Hong Kong", "John Lee", ...]
  └── embedding: vector(1536)

Entities (Neo4j):
  ├── id, canonical_name, entity_type
  ├── wikidata_qid (for matching)
  ├── mention_count
  └── profile_summary

Events (PostgreSQL - schema exists):
  ├── id, title, event_type
  ├── parent_event_id (for recursion)
  ├── confidence, coherence
  ├── event_start, event_end
  ├── status (provisional/stable)
  └── embedding: vector(1536)
```

**Validated:**
- HKFP article: 15 claims with entity_ids ✅
- All 15 entities in Neo4j ✅
- ClaimRepository.hydrate_entities() works ✅

### ✅ Domain Models
```python
# backend/models/claim.py
@dataclass
class Claim:
    id, page_id, text, event_time, confidence, modality
    metadata: dict  # includes entity_ids
    embedding: List[float]

    @property
    def entity_ids(self) -> List[uuid.UUID]  # ✅ Working

# backend/models/event.py
@dataclass
class Event:
    id, canonical_name, event_type
    parent_event_id: Optional[uuid.UUID]  # ✅ Recursive structure
    claim_ids: List[uuid.UUID]  # Direct claims
    confidence, coherence
    event_start, event_end
    status, embedding

    # Enums defined
    ClaimDecision: MERGE, ADD, DELEGATE, YIELD_SUBEVENT, REJECT
    ExaminationResult: claims_added, sub_events_created, claims_rejected
```

### ✅ Repositories
```python
# backend/repositories/claim_repository.py
ClaimRepository:
  ✅ get_by_page(page_id) -> List[Claim]
  ✅ hydrate_entities(claim) -> Claim with entities populated
  ✅ Parses metadata JSON correctly

# backend/repositories/entity_repository.py
EntityRepository:
  ✅ get_by_ids(entity_ids) -> List[Entity]
  ✅ get_by_canonical_name(name, type) -> Entity
  ✅ create(entity) -> Entity in Neo4j

# backend/repositories/event_repository.py (EXISTS - needs updates)
EventRepository:
  ✅ create(event) -> Event (dual-write PostgreSQL + Neo4j)
  ✅ update(event) -> Event
  ✅ get_by_id(event_id) -> Event
  ⬜ NEEDS: Fix field name mapping (title → canonical_name)
  ⬜ NEEDS: Add parent_event_id, claim_ids handling
  ⬜ NEEDS: Add coherence field handling
  ⬜ NEEDS: Update Neo4j calls (earliest_time → event_start)
  ⬜ NEEDS: find_candidates() method
  ⬜ NEEDS: get_sub_events(parent_id) method
```

---

## What New EventWorker Needs

### Input
```python
async def process_job(job: dict):
    page_id = job['page_id']

    # 1. Fetch claims (with entity_ids in metadata)
    claims = await claim_repo.get_by_page(page_id)

    # 2. Fetch entities for all claims
    all_entity_ids = set()
    for claim in claims:
        all_entity_ids.update(claim.entity_ids)
    entities = await entity_repo.get_by_ids(list(all_entity_ids))

    # 3. Find candidate events (NEW - needs implementation)
    candidates = await find_candidate_events(entities, claims[0].event_time)

    # 4. Process claims recursively (NEW - needs implementation)
    if candidates:
        best_event = candidates[0]
        result = await event_service.examine_claims(best_event, claims)
    else:
        event = await event_service.create_root_event(claims)
```

### Dependencies Needed

#### 1. EventRepository (Needs Implementation)
```python
class EventRepository:
    async def create(event: Event) -> Event
    async def get_by_id(event_id: uuid.UUID) -> Event
    async def get_sub_events(parent_id: uuid.UUID) -> List[Event]
    async def find_candidates(
        entity_ids: Set[uuid.UUID],
        reference_time: datetime,
        time_window_days: int = 7
    ) -> List[Tuple[Event, float]]  # (event, match_score)
    async def update(event: Event) -> Event
```

#### 2. EventService (Needs Implementation)
```python
class EventService:
    """Implements Event.examine() logic"""

    async def examine_claims(
        event: Event,
        claims: List[Claim]
    ) -> ExaminationResult:
        """
        Recursively classify and process claims into event structure

        For each claim:
        1. Check if duplicate (MERGE)
        2. Check if fits event topic (ADD)
        3. Check if sub-event handles better (DELEGATE)
        4. Check if novel aspect (YIELD_SUBEVENT)
        5. Otherwise REJECT
        """

    async def create_root_event(claims: List[Claim]) -> Event:
        """Create new root event from claims"""

    async def create_sub_event(
        parent: Event,
        claims: List[Claim]
    ) -> Event:
        """Create sub-event under parent"""
```

#### 3. Event Matching (Needs Implementation)
```python
async def find_candidate_events(
    entity_ids: Set[uuid.UUID],
    reference_time: datetime,
    page_embedding: Optional[List[float]] = None
) -> List[Tuple[Event, float]]:
    """
    Find events that might match new information

    Scoring:
    - Entity overlap: 40%
    - Time proximity: 30%
    - Semantic similarity: 30%
    """
```

#### 4. Event Embedding Generation (Needs Implementation)
```python
async def generate_event_embedding(claims: List[Claim]) -> List[float]:
    """
    Generate embedding for event from its claims

    Options:
    1. Average claim embeddings
    2. LLM-based: "Event about {summary of claims}"
    """
```

#### 5. Claim Similarity (Needs Implementation)
```python
def calculate_claim_similarity(claim1: Claim, claim2: Claim) -> float:
    """
    Semantic similarity between claims (using embeddings)

    Returns: 0.0 to 1.0
    """
```

---

## Neo4j Schema Status

### Current Schema (To Check)
```cypher
// Check constraints
SHOW CONSTRAINTS

// Check indexes
SHOW INDEXES

// Check if Event nodes support parent_event_id
MATCH (e:Event) RETURN e LIMIT 1
```

### Required Schema
```cypher
// Event nodes with recursive structure
CREATE CONSTRAINT event_id IF NOT EXISTS
FOR (e:Event) REQUIRE e.id IS UNIQUE;

// Parent-child relationships
(Event)-[:CONTAINS]->(Event)

// Claims link to events (not phases)
(Event)-[:CONTAINS_CLAIM {sequence: int}]->(claim_id: string)

// Entities link to events
(Event)-[:INVOLVES]->(Entity)
```

---

## Database Schema Alignment

### PostgreSQL Events Table - VERIFIED ✅
```sql
✅ parent_event_id: uuid (with FK constraint and index)
✅ embedding: vector(1536) (with ivfflat index)
✅ coherence: double precision
✅ coherence_score: double precision (legacy, keep for compatibility)
✅ event_start, event_end: timestamp with time zone
✅ confidence: double precision
✅ status: varchar(50)
✅ title: text (NOTE: Event model uses 'canonical_name')
✅ event_type: varchar(50)
✅ metadata: jsonb

Foreign Key: parent_event_id → core.events(id) (recursive)
Index: idx_events_parent on parent_event_id
```

**Schema is READY for recursive events!** ✅

---

## Implementation Plan

### Phase 1: Foundation (Before EventWorker)
1. ✅ Event model with recursive fields
2. ✅ Claim.entity_ids property
3. ✅ PostgreSQL schema verified (parent_event_id, coherence, embedding)
4. ⬜ EventRepository refactoring (fix field names, add find_candidates)
5. ⬜ Event embedding generation
6. ⬜ Claim similarity function
7. ⬜ Remove PhaseRepository (phases are now Events)

### Phase 2: EventService Core Logic
6. ⬜ EventService.examine_claims() - classify claim
7. ⬜ EventService.create_root_event()
8. ⬜ EventService.create_sub_event()
9. ⬜ Event matching algorithm

### Phase 3: EventWorker Refactor
10. ⬜ New EventWorker using EventService
11. ⬜ Remove old phase discovery logic
12. ⬜ Test with HKFP article (should create root event)

### Phase 4: Multi-Article Testing
13. ⬜ Process AP article (should attach to HKFP event)
14. ⬜ Process BBC article (should create sub-events)
15. ⬜ Validate recursive structure in Neo4j

---

## Testing Strategy

### Test 1: Single Article (Root Event)
```
Input: HKFP article (15 claims)
Expected:
  - Creates root event: "2025 Hong Kong Tai Po Fire"
  - All 15 claims attached to root event
  - No sub-events yet
  - Status: provisional
  - Confidence: ~0.3
```

### Test 2: Second Article (Event Attachment)
```
Input: AP article (10 claims)
Expected:
  - Finds HKFP event (entity overlap > 0.6)
  - Examines 10 claims:
    - 2 duplicates → MERGE
    - 5 fit existing topic → ADD
    - 3 novel aspect → YIELD_SUBEVENT ("Investigation")
  - Status: stable
  - Confidence: ~0.6
```

### Test 3: Third Article (Sub-Event Delegation)
```
Input: BBC article (12 claims)
Expected:
  - Finds HKFP event
  - Examines 12 claims:
    - 2 duplicates → MERGE
    - 4 fit "Investigation" sub-event → DELEGATE
    - 3 fit root → ADD
    - 3 novel aspect → YIELD_SUBEVENT ("Public Response")
  - Status: stable
  - Confidence: ~0.8
  - Sub-events: 2 (Investigation, Public Response)
```

---

## What to Build First

### Critical Path (Must Build Before EventWorker)
1. **EventRepository** - CRUD operations
2. **Event embedding generation** - For semantic matching
3. **Claim similarity function** - For duplicate detection
4. **Event matching algorithm** - Find candidates

### Can Build Incrementally
- EventService.examine_claims() - Start with simple logic, iterate
- Sub-event creation - Can defer to Phase 2
- Complex matching - Start with entity overlap only

---

## Open Questions

1. **Event embedding strategy:**
   - Option A: Average claim embeddings (fast, simple)
   - Option B: LLM-generated from event summary (better quality)
   - **Recommendation:** Start with A, upgrade to B later

2. **Claim similarity threshold:**
   - 0.9 for duplicates?
   - 0.7 for related?
   - **Recommendation:** Test with real claims

3. **Entity overlap calculation:**
   - Use all entity types equally?
   - Weight LOCATION higher for location-specific events?
   - **Recommendation:** Equal weight initially

4. **Sub-event threshold:**
   - How different must claims be to yield sub-event?
   - **Recommendation:** Entity overlap < 0.5, coherence with parent < 0.6

---

## Next Steps

1. Create EventRepository skeleton
2. Implement event embedding generation
3. Implement claim similarity (cosine of embeddings)
4. Implement simple event matching (entity overlap only)
5. Test find_candidate_events() with existing fire events
6. Then start EventService logic
