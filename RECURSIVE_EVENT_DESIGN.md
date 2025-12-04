# Recursive Event Formation Architecture

## Core Principle

**Events are recursive containers**: No distinction between "phase" and "event" - both are Events at different maturity levels. An event can contain sub-events, which can contain their own sub-events.

## Event Structure

```python
@dataclass
class Event:
    id: uuid.UUID
    canonical_name: str
    event_type: str  # FIRE, PROTEST, ELECTION, etc.

    # Recursive structure
    parent_event_id: Optional[uuid.UUID] = None

    # Direct claims supporting THIS event (high-confidence facts)
    claim_ids: List[uuid.UUID] = field(default_factory=list)

    # Quality metrics
    confidence: float = 0.0  # Evidence strength (0-1)
    coherence: float = 0.0   # How well claims fit together (0-1)

    # Temporal bounds
    earliest_time: Optional[datetime] = None
    latest_time: Optional[datetime] = None

    # Embedding for semantic matching
    embedding: Optional[List[float]] = None

    # Status
    status: str = 'provisional'  # provisional, stable, archived
```

## Information Flow

### 1. New Page Arrives

```
Input: Page with claims, entities, page embedding

Step 1: Find Candidate Events
  - Query Neo4j for events with:
    * Entity overlap > 0.3
    * Time proximity < 7 days
    * Semantic similarity > 0.6 (if embeddings available)

Step 2: Score Candidates
  - Combined score = 0.4 * entity_overlap + 0.3 * time_proximity + 0.3 * semantic_sim

Step 3: Decision
  - If best_score > 0.6 → Event.examine(claims)
  - Else → Create new root event
```

### 2. Event.examine(claims) - Recursive Core

```python
def examine(self, new_claims: List[Claim]) -> ExaminationResult:
    """
    Recursively process new claims into event structure

    Returns:
        - claims_added: Claims merged into this event
        - sub_events_created: New sub-events yielded
        - claims_rejected: Claims that don't fit
    """
    claims_added = []
    claims_rejected = []
    sub_events_created = []
    remaining_claims = new_claims.copy()

    for claim in remaining_claims:
        decision = self._classify_claim(claim)

        if decision == ClaimDecision.MERGE:
            # Duplicate of existing claim → corroborate
            self._merge_duplicate(claim)
            claims_added.append(claim)

        elif decision == ClaimDecision.ADD:
            # Novel but fits my topic
            self.claim_ids.append(claim.id)
            claims_added.append(claim)

        elif decision == ClaimDecision.DELEGATE:
            # A sub-event handles this better
            sub_event = self._find_best_sub_event(claim)
            result = sub_event.examine([claim])  # RECURSIVE
            claims_added.extend(result.claims_added)
            sub_events_created.extend(result.sub_events_created)

        elif decision == ClaimDecision.YIELD_SUBEVENT:
            # Novel aspect → create sub-event
            sub_event = self._create_sub_event([claim])
            sub_events_created.append(sub_event)
            claims_added.append(claim)

        elif decision == ClaimDecision.REJECT:
            claims_rejected.append(claim)

    # Update event quality metrics
    self._update_metrics()

    return ExaminationResult(
        claims_added=claims_added,
        sub_events_created=sub_events_created,
        claims_rejected=claims_rejected
    )
```

### 3. Claim Classification Logic

```python
def _classify_claim(self, claim: Claim) -> ClaimDecision:
    """Decide how to handle a new claim"""

    # 1. Check for duplicates (semantic similarity > 0.9)
    for existing_claim_id in self.claim_ids:
        existing_claim = get_claim(existing_claim_id)
        if semantic_similarity(claim, existing_claim) > 0.9:
            return ClaimDecision.MERGE

    # 2. Check if it fits my topic
    my_entities = self._get_entities()
    claim_entities = claim.entity_ids
    entity_overlap = len(set(my_entities) & set(claim_entities)) / max(len(my_entities), len(claim_entities))

    if entity_overlap > 0.5:
        # Check semantic coherence with my existing claims
        coherence = self._calculate_coherence_with_claim(claim)
        if coherence > 0.6:
            return ClaimDecision.ADD

    # 3. Check if sub-event handles it better
    for sub_event in self._get_sub_events():
        sub_match = sub_event._match_score(claim)
        my_match = self._match_score(claim)
        if sub_match > my_match and sub_match > 0.5:
            return ClaimDecision.DELEGATE

    # 4. Is it novel but related? (shares context but different aspect)
    if entity_overlap > 0.3 and entity_overlap < 0.5:
        # Related but distinct aspect
        return ClaimDecision.YIELD_SUBEVENT

    # 5. Unrelated
    return ClaimDecision.REJECT
```

## Relationship Types in Neo4j

```cypher
// Parent-child structure
(ParentEvent)-[:CONTAINS {created_at: datetime()}]->(ChildEvent)

// Temporal progression (sub-events that form a sequence)
(Event)-[:PHASE_OF {sequence: 1}]->(ParentEvent)
(Event1)-[:FOLLOWED_BY]->(Event2)

// Cross-event relationships
(Event1)-[:SIMILAR_TO {score: 0.85}]->(Event2)
(Event1)-[:CAUSES]->(Event2)
(Event1)-[:RESPONDS_TO]->(Event2)
```

## Storage Strategy

### PostgreSQL
```sql
-- Add embedding column to events table (if not exists)
ALTER TABLE core.events ADD COLUMN IF NOT EXISTS embedding vector(1536);

-- Claims already have embeddings
-- No changes needed to claims table
```

### Neo4j
```cypher
// Event nodes with parent_id for recursion
CREATE (e:Event {
    id: $id,
    canonical_name: $name,
    event_type: $type,
    parent_event_id: $parent_id,  // NULL for root events
    confidence: $confidence,
    coherence: $coherence,
    status: $status,
    earliest_time: $earliest,
    latest_time: $latest
})

// Claims link to events (not phases)
(Event)-[:CONTAINS_CLAIM {confidence: 0.8}]->(claim_id)

// Entities link to events
(Event)-[:INVOLVES]->(Entity)
```

## Matching & Scoring

### Event Matching Algorithm
```python
def find_candidate_events(
    entities: Set[str],
    reference_time: datetime,
    page_embedding: List[float]
) -> List[Tuple[Event, float]]:
    """
    Find events that might match new information

    Returns: List of (event, match_score) sorted by score
    """
    candidates = []

    # Query Neo4j for events with entity overlap
    events = neo4j.query_events_by_entities(entities, time_window_days=7)

    for event in events:
        # 1. Entity overlap score
        event_entities = event.get_entities()
        entity_overlap = len(entities & event_entities) / max(len(entities), len(event_entities))

        # 2. Temporal proximity score
        time_diff = abs((reference_time - event.earliest_time).days)
        time_score = max(0, 1 - time_diff / 7)  # Decay over 7 days

        # 3. Semantic similarity score (if embeddings available)
        semantic_score = 0
        if event.embedding and page_embedding:
            semantic_score = cosine_similarity(event.embedding, page_embedding)

        # Combined score
        score = (
            0.4 * entity_overlap +
            0.3 * time_score +
            0.3 * semantic_score
        )

        candidates.append((event, score))

    return sorted(candidates, key=lambda x: x[1], reverse=True)
```

## Sub-Event Creation Logic

```python
def _create_sub_event(self, claims: List[Claim]) -> Event:
    """
    Create a sub-event from novel claims

    Strategy:
    - If claims form temporal sequence → create PHASE_OF sub-event
    - If claims share distinct aspect → create CONTAINS sub-event
    """
    # Determine sub-event type
    if self._is_temporal_sequence(claims):
        relationship_type = "PHASE_OF"
        canonical_name = self._generate_phase_name(claims)
    else:
        relationship_type = "CONTAINS"
        canonical_name = self._generate_aspect_name(claims)

    # Create sub-event
    sub_event = Event(
        id=uuid.uuid4(),
        canonical_name=canonical_name,
        event_type=self.event_type,
        parent_event_id=self.id,
        claim_ids=[c.id for c in claims],
        confidence=0.5,  # Initial confidence
        earliest_time=min(c.event_time for c in claims if c.event_time),
        latest_time=max(c.event_time for c in claims if c.event_time)
    )

    # Store in Neo4j
    neo4j.create_event(sub_event)
    neo4j.create_relationship(
        self.id,
        sub_event.id,
        relationship_type
    )

    return sub_event
```

## Example: Hong Kong Fire

### First Article (HKFP - Condolences)
```
Input: 14 claims about international condolences
Process:
  - No candidate events found
  - Create root event: "2025 Hong Kong Tai Po Fire"
  - Add all 14 claims to root event
Result: Event(status=provisional, confidence=0.3)
```

### Second Article (AP News - Safety Concerns)
```
Input: 10 claims about safety issues, arrests, investigation
Process:
  1. Find candidates: "2025 Hong Kong Tai Po Fire" (entity_overlap=0.8, time_proximity=1.0)
  2. Root event examines claims:
     - 2 claims overlap with existing → MERGE
     - 3 claims about investigation → Novel aspect → YIELD_SUBEVENT
     - 5 claims about safety → Novel aspect → YIELD_SUBEVENT
  3. Create sub-events:
     - "Investigation and Arrests" (3 claims)
     - "Safety Concerns" (5 claims)
Result: Event(status=stable, confidence=0.6, sub_events=2)
```

### Third Article (BBC - Anger & Questions)
```
Input: 12 claims about public reaction, alarm failures, inspections
Process:
  1. Find candidates: "2025 Hong Kong Tai Po Fire" (entity_overlap=0.7)
  2. Root event examines claims:
     - 2 claims duplicate → MERGE
     - 4 claims fit "Investigation" sub-event → DELEGATE
     - 3 claims fit "Safety Concerns" sub-event → DELEGATE
     - 3 claims about public reaction → YIELD_SUBEVENT
  3. Sub-events recursively process delegated claims
  4. Create new sub-event: "Public Response" (3 claims)
Result: Event(status=stable, confidence=0.8, sub_events=3)
```

## Promotion & Demotion

### Sub-Event Promotion
When sub-event becomes substantial enough to be independent:
```python
if sub_event.claim_count > 10 and sub_event.confidence > 0.7:
    # Promote to sibling event
    sub_event.parent_event_id = parent.parent_event_id
    relationship_type = "RELATED_TO"  # No longer CONTAINS
```

### Event Merging
When two sibling events are too similar:
```python
if similarity(event1, event2) > 0.9 and same_parent:
    # Merge into stronger event
    merge_events(event1, event2)
```

## Configuration

```python
# Thresholds
MATCH_THRESHOLD = 0.6           # Minimum score to trigger examine()
DUPLICATE_THRESHOLD = 0.9       # Semantic similarity for duplicates
COHERENCE_THRESHOLD = 0.6       # Min coherence to add claim
ENTITY_OVERLAP_THRESHOLD = 0.5  # Entity overlap for topic match
SUB_EVENT_THRESHOLD = 0.3       # Min overlap for yielding sub-event

# Limits
MAX_SUB_EVENT_DEPTH = 3         # Max recursion depth
MIN_CLAIMS_FOR_PROMOTION = 10   # Promote sub-event to independent
MAX_CLAIMS_PER_EVENT = 50       # Split if too many claims

# Scoring weights
ENTITY_WEIGHT = 0.4
TIME_WEIGHT = 0.3
SEMANTIC_WEIGHT = 0.3
```

## Implementation Steps

1. **Update Event model** - Add parent_id, embedding, recursive methods
2. **Generate event embeddings** - Average claim embeddings or LLM-based
3. **Implement Event.examine()** - Core recursive logic
4. **Implement matching/scoring** - Find candidate events
5. **Update Neo4j schema** - Support recursive CONTAINS relationships
6. **Refactor EventWorker** - Use new recursive model
7. **Test with fire articles** - Process HKFP → AP → BBC step by step

## Testing Protocol

```python
# Test 1: First article creates root event
process_page(HKFP_page_id)
assert event_count == 1
assert event.status == 'provisional'
assert event.parent_event_id is None

# Test 2: Second article creates sub-events
process_page(AP_page_id)
assert event_count == 1  # Still same root event
assert len(event.sub_events) == 2
assert event.status == 'stable'

# Test 3: Third article delegates to sub-events
process_page(BBC_page_id)
assert event_count == 1
assert len(event.sub_events) == 3
assert event.confidence > 0.7
```
