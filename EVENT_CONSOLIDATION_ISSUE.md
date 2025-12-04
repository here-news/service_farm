# Issue: Implement Event Deduplication and Cross-Article Consolidation

## Priority: 🔴 CRITICAL

## Problem Statement

When multiple news articles cover the **same event** from different sources, the system currently creates **duplicate events** instead of recognizing they should be consolidated. This fragments the knowledge graph and prevents proper event tracking across sources.

### Example (from testing 2025-12-04)

**Article 1**: HKFP - "You are not alone: Countries, officials, Grenfell victims express condolences"
- Created Event: `"2025 Hong Kong Tai Po Fire"`
- Event ID: `65d88b16-49ce-4394-9de1-45a7623220b9`
- Claims: 14

**Article 2**: AP News - "Deadly fire raises fears about safety elsewhere"
- Created Event: `"2025 Hong Kong High-Rise Apartment Fire"` ❌ **DUPLICATE**
- Event ID: `01341302-b0ef-4b90-8d63-000b1e0f0a6d`
- Claims: 10

**Expected Behavior**: System should detect these cover the **same fire event** and either:
1. Merge Article 2's claims into existing event from Article 1
2. Treat Article 2 as an amendment/update to the existing event
3. Link them as different perspectives on the same event

**Evidence they're the same event**:
- ✅ Same location: Wang Fuk Court, Tai Po, Hong Kong
- ✅ Same event type: FIRE
- ✅ Same date: 2025-11-28
- ✅ Shared entities: Hong Kong, Wang Fuk Court, John Lee
- ✅ Related claims: Both discuss casualties, investigations, safety concerns

---

## Impact

**Without event consolidation**:
- ❌ Knowledge graph fragmentation - same event split across multiple nodes
- ❌ Cannot track event evolution across different news sources
- ❌ Duplicate entities - same person/org/location appears multiple times
- ❌ Timeline reconstruction fails - claims not chronologically ordered
- ❌ Cross-article entity resolution broken
- ❌ Cannot identify consensus vs. discrepancies across sources
- ❌ Wasted storage and processing

**With event consolidation**:
- ✅ Single authoritative event node with all coverage
- ✅ Timeline shows event evolution across sources and time
- ✅ Can identify which claims are corroborated by multiple sources
- ✅ Detect contradictions or inconsistencies between sources
- ✅ Entity resolution across articles works correctly
- ✅ Richer event context from multiple perspectives

---

## Requirements

### 1. Event Similarity Detection

**Algorithm** to determine if two events are the same:

```python
def are_same_event(event1: Event, event2: Event) -> tuple[bool, float]:
    """
    Detect if two events are the same.

    Returns:
        (is_same, confidence_score)
    """
    signals = []

    # 1. Temporal proximity
    time_diff = abs(event1.event_time - event2.event_time)
    if time_diff < timedelta(hours=24):
        signals.append(('temporal', 0.3, True))
    elif time_diff < timedelta(days=7):
        signals.append(('temporal', 0.15, True))
    else:
        signals.append(('temporal', 0.0, False))

    # 2. Location/entity overlap
    shared_entities = set(event1.entities) & set(event2.entities)
    entity_similarity = len(shared_entities) / max(len(event1.entities), len(event2.entities))
    if entity_similarity > 0.5:
        signals.append(('entity_overlap', 0.25, True))

    # 3. Event type match
    if event1.event_type == event2.event_type:
        signals.append(('event_type', 0.15, True))

    # 4. Semantic title similarity (using embeddings)
    title_similarity = cosine_similarity(event1.title_embedding, event2.title_embedding)
    if title_similarity > 0.7:
        signals.append(('title_semantic', 0.2, True))

    # 5. Claim semantic overlap
    claim_overlap = calculate_claim_overlap(event1.claims, event2.claims)
    if claim_overlap > 0.3:
        signals.append(('claim_overlap', 0.1, True))

    # Calculate confidence
    confidence = sum(weight for _, weight, match in signals if match)
    is_same = confidence >= 0.6  # Threshold for consolidation

    return is_same, confidence
```

**Required signals**:
- [x] Temporal proximity (within 24-72 hours)
- [x] Entity overlap (shared entities like locations, people)
- [x] Event type match (both FIRE, both PROTEST, etc.)
- [x] Semantic similarity (title embeddings)
- [x] Claim content overlap (semantic similarity of claims)

### 2. Event Consolidation Strategy

**When second article about same event arrives**:

```
IF event_similarity_confidence > 0.6:
    IF existing_event.confidence > 0.7:
        → Strategy A: MERGE (high confidence in existing structure)
    ELSE:
        → Strategy B: REGENERATE (low confidence, benefit from more data)
ELSE:
    → Create new event (different events)
```

**Strategy A: MERGE** (existing event is high quality)
1. Add new claims to existing event
2. Recalculate phases (may create new phases or extend existing)
3. Update entity links (merge duplicate entities)
4. Update event confidence (increases with more sources)
5. Track source articles (event.source_articles list)
6. Update event timeline with new information

**Strategy B: REGENERATE** (existing event is low confidence)
1. Combine all claims from both articles
2. Re-run phase discovery on combined claim set
3. Regenerate event title based on fuller picture
4. Update event confidence
5. Archive old event structure (for audit trail)

### 3. Entity Consistency Across Articles

**Problem**: Same entity mentioned in multiple articles creates duplicate nodes.

**Solution**: Entity matching before creation
```python
async def get_or_create_entity(
    canonical_name: str,
    entity_type: str,
    context: dict
) -> Entity:
    """
    Get existing entity or create new one.
    Uses Wikidata QID as ground truth for deduplication.
    """
    # 1. Check if entity exists with Wikidata QID
    if context.get('wikidata_qid'):
        existing = await find_entity_by_qid(context['wikidata_qid'])
        if existing:
            return existing

    # 2. Check by canonical name + type
    existing = await find_entity_by_name_type(canonical_name, entity_type)
    if existing:
        # Merge/enrich with new context
        await enrich_entity(existing, context)
        return existing

    # 3. Create new entity
    return await create_entity(canonical_name, entity_type, context)
```

**Key**: Wikidata QIDs enable cross-article entity resolution

### 4. Event Relationship Detection

**Beyond duplication**: Detect related but distinct events

```python
class EventRelationship(Enum):
    DUPLICATE_OF = "DUPLICATE_OF"      # Same event, should merge
    RELATED_TO = "RELATED_TO"          # Related but distinct
    CAUSED_BY = "CAUSED_BY"            # Causal relationship
    AFTERMATH_OF = "AFTERMATH_OF"       # Temporal follow-up
    RESPONSE_TO = "RESPONSE_TO"        # Official response
```

**Examples**:
- Fire incident → Fire investigation (AFTERMATH_OF)
- Fire incident → Protest about fire safety (RESPONSE_TO)
- Building collapse → Fire at rescue scene (CAUSED_BY)

---

## Technical Implementation

### Phase 1: Detection (1 week)

**Files to modify**:
- `backend/workers/event_worker_neo4j.py` - Add event similarity check before creation
- `backend/services/neo4j_service.py` - Add event query methods
- `backend/services/event_consolidation.py` - NEW: Core consolidation logic

**Algorithm**:
```python
async def process_page_events(page_id: uuid.UUID):
    """Event worker main logic with consolidation"""

    # Extract claims and entities (existing logic)
    claims = await get_claims_for_page(page_id)

    # Discover events/phases (existing logic)
    discovered_events = await discover_events_and_phases(claims)

    # NEW: Check for existing similar events
    for discovered_event in discovered_events:
        similar_events = await find_similar_events(
            discovered_event,
            time_window=timedelta(days=7),
            min_similarity=0.6
        )

        if similar_events:
            # Consolidate into existing event
            best_match = similar_events[0]  # Highest confidence
            await consolidate_events(
                existing_event=best_match,
                new_claims=discovered_event.claims,
                source_page=page_id
            )
        else:
            # Create new event
            await create_event(discovered_event)
```

**Deliverables**:
- [ ] `find_similar_events()` function with similarity scoring
- [ ] `consolidate_events()` function (merge or regenerate)
- [ ] Event similarity unit tests
- [ ] Integration test with 2 articles on same event

### Phase 2: Consolidation Strategy (1 week)

**Implementation**:

```python
async def consolidate_events(
    existing_event: Event,
    new_claims: List[Claim],
    source_page: uuid.UUID,
    strategy: ConsolidationStrategy = ConsolidationStrategy.AUTO
) -> Event:
    """
    Consolidate new article's claims into existing event.
    """
    if strategy == ConsolidationStrategy.AUTO:
        if existing_event.confidence > 0.7:
            strategy = ConsolidationStrategy.MERGE
        else:
            strategy = ConsolidationStrategy.REGENERATE

    if strategy == ConsolidationStrategy.MERGE:
        # Add new claims to existing phases or create new phases
        updated_phases = await merge_claims_into_phases(
            existing_phases=existing_event.phases,
            new_claims=new_claims
        )

        # Update event
        await update_event(
            event_id=existing_event.id,
            phases=updated_phases,
            source_articles=existing_event.sources + [source_page],
            confidence=calculate_updated_confidence(existing_event, new_claims)
        )

    elif strategy == ConsolidationStrategy.REGENERATE:
        # Combine all claims and regenerate
        all_claims = existing_event.all_claims + new_claims

        # Re-discover phases with full context
        new_phases = await discover_phases(all_claims)

        # Update event (keeps same ID)
        await update_event(
            event_id=existing_event.id,
            phases=new_phases,
            title=await generate_event_title(all_claims),
            source_articles=existing_event.sources + [source_page],
            confidence=0.8  # Higher confidence with multiple sources
        )

    return await get_event(existing_event.id)
```

**Deliverables**:
- [ ] Merge strategy implementation
- [ ] Regenerate strategy implementation
- [ ] Phase re-discovery with combined claims
- [ ] Confidence updating logic

### Phase 3: Entity Deduplication (1 week)

**Implementation**:
```python
async def create_or_merge_entity(
    entity_data: dict,
    existing_entities: List[Entity]
) -> Entity:
    """
    Create new entity or merge with existing based on matching.
    """
    # Try matching by Wikidata QID first
    if entity_data.get('wikidata_qid'):
        for existing in existing_entities:
            if existing.wikidata_qid == entity_data['wikidata_qid']:
                # Same entity - merge information
                return await merge_entity_info(existing, entity_data)

    # Try matching by canonical name + type
    canonical_name = entity_data['canonical_name']
    entity_type = entity_data['entity_type']

    for existing in existing_entities:
        if (existing.canonical_name == canonical_name and
            existing.entity_type == entity_type):
            # Likely same entity
            similarity = calculate_entity_similarity(existing, entity_data)
            if similarity > 0.8:
                return await merge_entity_info(existing, entity_data)

    # New entity
    return await create_entity(entity_data)
```

**Deliverables**:
- [ ] Entity matching logic (by QID, by name+type)
- [ ] Entity merging (combine aliases, update mention counts)
- [ ] Handle entity conflicts (different descriptions for same name)
- [ ] Tests for entity deduplication

### Phase 4: Testing & Validation (1 week)

**Test Cases**:

1. **Same Event, Different Sources**
   - [ ] Submit Article A (HKFP about Tai Po fire)
   - [ ] Submit Article B (AP News about same fire)
   - [ ] Verify: Single event created, claims consolidated
   - [ ] Verify: Entities shared (not duplicated)

2. **Related But Different Events**
   - [ ] Submit Article A (Fire incident)
   - [ ] Submit Article B (Protest about fire safety)
   - [ ] Verify: Two events created
   - [ ] Verify: Events linked with RESPONSE_TO relationship

3. **Event Updates Over Time**
   - [ ] Submit Article A (Breaking: Fire occurs)
   - [ ] Submit Article B (Update: Casualty count rises)
   - [ ] Submit Article C (Investigation findings)
   - [ ] Verify: Single event with timeline showing evolution

4. **Cross-Article Entity Resolution**
   - [ ] Submit 3 articles mentioning "John Lee"
   - [ ] Verify: Single entity node for John Lee
   - [ ] Verify: All mentions link to same node

**Deliverables**:
- [ ] Automated test suite for event consolidation
- [ ] Regression tests for existing functionality
- [ ] Performance benchmarks (consolidation shouldn't slow pipeline)

---

## Success Criteria

### Functional Requirements
- [ ] System detects duplicate events with >90% accuracy
- [ ] Duplicate events consolidated automatically
- [ ] Entity deduplication works across articles
- [ ] Event confidence increases with multiple sources
- [ ] Timeline reconstruction spans multiple articles
- [ ] No regression in existing event formation quality

### Quality Metrics
```
Event Consolidation Rate: {consolidated}/{total_candidates} >80%
False Positive Rate: {wrong_consolidations}/{total_consolidations} <5%
Entity Deduplication: {shared_entities}/{potential_duplicates} >85%
Processing Time Impact: <10% increase per article
```

### Documentation
- [ ] Testing framework updated with consolidation checks
- [ ] Architecture documentation for consolidation logic
- [ ] Examples of consolidated events in docs

---

## Testing Protocol (After Implementation)

**For every new article submission**:

1. **Before processing**:
   ```sql
   -- Check for potential related events (manual)
   SELECT id, title, event_type, created_at
   FROM core.events
   WHERE event_time > (NOW() - INTERVAL '7 days')
   AND event_type IN ('FIRE', 'PROTEST', /* etc */)
   ORDER BY created_at DESC;
   ```

2. **After processing**:
   ```sql
   -- Check if new event was created or consolidated
   SELECT
       e.id,
       e.title,
       (SELECT COUNT(*) FROM core.page_events WHERE event_id = e.id) as source_count,
       e.confidence
   FROM core.events e
   WHERE e.updated_at > (NOW() - INTERVAL '5 minutes')
   ORDER BY e.updated_at DESC;
   ```

3. **Verify consolidation**:
   ```cypher
   // Neo4j: Check event has claims from multiple sources
   MATCH (event:Event {id: $event_id})
   MATCH (event)<-[:PART_OF]-(claim:Claim)
   RETURN DISTINCT claim.page_id as source_page
   ```

4. **Expected**:
   - ✅ Single event for same incident from multiple sources
   - ✅ Event confidence >0.7 with multiple sources
   - ✅ Phases coherently combine information from all articles
   - ✅ Entity nodes shared across articles

---

## Related Issues

- [ ] Issue #XXX: Lower Wikidata enrichment threshold (enables entity matching)
- [ ] Issue #XXX: Improve entity recall in semantic extraction
- [ ] Issue #XXX: Add event relationship types (CAUSED_BY, RESPONSE_TO, etc.)
- [ ] Issue #XXX: Implement event timeline visualization

---

## References

- Testing Framework: `TESTING_FRAMEWORK.md` - Stage 5B (Event Consolidation)
- Event Formation Design: `EVENT_FORMATION_UNIVERSAL_DESIGN.md`
- Neo4j Schema: `NEO4J_SCHEMA.md`

---

## Priority Justification

**Why CRITICAL**:
1. Core value proposition of system is **multi-source intelligence**
2. Currently **cannot track events across articles** - defeats the purpose
3. Graph fragmentation makes entity resolution impossible
4. Blocks timeline reconstruction and fact-checking features
5. Wastes storage and compute on duplicate processing

**Without this, the system is basically a single-article analyzer, not a knowledge graph.**

---

## Estimated Effort

- **Phase 1 (Detection)**: 1 week / 1 developer
- **Phase 2 (Consolidation)**: 1 week / 1 developer
- **Phase 3 (Entity Dedup)**: 1 week / 1 developer
- **Phase 4 (Testing)**: 1 week / 1 developer

**Total**: 4 weeks (can parallelize some work)

**Dependencies**:
- Wikidata enrichment working (currently blocked by high threshold)
- Neo4j entity queries optimized
- Event formation stable (already working)

---

## Acceptance Test

**Final test before closing issue**:

1. Submit 3 articles about Tai Po fire:
   - HKFP article (already tested)
   - AP News article (already tested)
   - Third source (e.g., Reuters, SCMP)

2. Verify:
   - [ ] Single event created: "2025 Hong Kong Tai Po Fire"
   - [ ] Event has 3 source articles
   - [ ] All ~30+ claims consolidated into coherent phases
   - [ ] Entities (Hong Kong, John Lee, Wang Fuk Court) appear once
   - [ ] Timeline shows event evolution
   - [ ] Event confidence >0.8

3. Performance:
   - [ ] Third article processing time <90 seconds
   - [ ] No errors in logs
   - [ ] Graph queries remain fast (<100ms)

**If all checks pass: Issue can be closed. ✅**

---

**Created**: 2025-12-04
**Priority**: CRITICAL
**Assignee**: TBD
**Labels**: `enhancement`, `critical`, `event-formation`, `graph-integrity`
