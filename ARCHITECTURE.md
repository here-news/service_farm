# Architecture: Data Lake + Knowledge Graph

## Overview

This system uses a **hybrid storage strategy** with clear separation of concerns:

- **PostgreSQL**: Data lake for raw artifacts (pages, claims with full metadata)
- **Neo4j**: Knowledge graph for curated, stable relationships (events, entities, relationships)

##Storage Strategy

### PostgreSQL (Data Lake)
**Purpose**: Store ALL raw extraction data from external sources

**Tables**:
- `core.pages` - All scraped pages (title, content, metadata, embeddings)
- `core.claims` - ALL extracted claims (text, embeddings, entity_ids in JSON)
- `core.entities` - Entity metadata cache
- `core.event_embeddings` - Event embeddings for vector search

**Characteristics**:
- Immutable artifacts from external sources
- Rich metadata and embeddings for search
- Every page and claim stays here permanently
- No deletion - this is the audit trail

### Neo4j (Knowledge Graph)
**Purpose**: Curated knowledge - only stable, verified relationships

**Nodes**:
- `Event` - Curated event hierarchy (NO embeddings - stored in PostgreSQL)
- `Entity` - Stable entities (people, places, organizations)
- `Claim` - **ONLY** claims that passed event examination (NO embeddings - stored in PostgreSQL)

**Relationships**:
- `Event-[CONTAINS]->Event` - Parent-child event hierarchy
- `Event-[SUPPORTS]->Claim` - Claims supporting events
- `Event-[INVOLVES]->Entity` - Entities involved in events
- `Claim-[MENTIONS]->Entity` - Entities mentioned in claims

**Note**: Source tracking (claim→page) uses `claim.page_id` in PostgreSQL, not graph relationships

**Characteristics**:
- Only curated, verified knowledge
- Claims in Neo4j are a SUBSET of claims in PostgreSQL
- Graph enables fast traversal (event→claims→entities)
- Clean, queryable structure

## Domain Model Layer

### Models (storage-agnostic)
Located in `models/`:
- `Event` - Domain model for events
- `Claim` - Domain model for claims
- `Entity` - Domain model for entities
- `Page` - Domain model for pages

**Key principle**: Models have NO knowledge of storage (no SQL, no Cypher)

### Repositories (data access)
Located in `repositories/`:
- `EventRepository` - ALL Event data access (PostgreSQL + Neo4j)
- `ClaimRepository` - ALL Claim data access (PostgreSQL + Neo4j)
- `EntityRepository` - ALL Entity data access (PostgreSQL + Neo4j)
- `PageRepository` - ALL Page data access (PostgreSQL)

**Key principle**: Repositories encapsulate storage decisions

**Methods return domain models**:
```python
event = await event_repo.get_by_id(event_id)  # Returns Event model
await event_repo.link_claim(event, claim)      # Links in Neo4j graph
claims = await event_repo.get_event_claims(event_id)  # Returns List[Claim]
```

### Services (business logic)
Located in `services/`:
- `EventService` - Event formation, claim examination, narrative generation
- `Neo4jService` - Low-level Neo4j operations (used BY repositories only)

**Key principle**: Services use ONLY domain models and repositories
- ✅ `await self.event_repo.link_claim(event, claim)`
- ❌ `await self.event_repo.neo4j.create_claim(...)`  # VIOLATION

### Workers (task processing)
Located in `workers/`:
- `extraction_worker` - Extract content from URLs
- `semantic_worker` - Extract claims and entities
- `event_worker` - Form events from claims
- `wikidata_worker` - Enrich entities

**Key principle**: Workers use repositories for most data access
- Complex transactional operations MAY use direct DB access for performance
- Lifecycle operations (connect/close) are acceptable
- Services MUST use repositories

## Data Flow

### 1. Page Extraction
```
URL → extraction_worker
  → PostgreSQL core.pages (title, content, metadata)
  → Queue: semantic processing
```

### 2. Semantic Analysis
```
Page → semantic_worker
  → Extract claims → PostgreSQL core.claims (ALL claims)
  → Extract entities → Neo4j Entity nodes
  → Queue: event processing
```

### 3. Event Formation
```
Claims → event_worker
  → event_service.examine_claims()
  → Classify each claim (ADD/MERGE/REJECT/YIELD)
  → For ACCEPTED claims:
    1. Create Claim node in Neo4j (subset)
    2. Create Event-[SUPPORTS]->Claim relationship
    3. Update event narrative
```

### 4. Querying Events
```
GET /event/{id}/claims
  → event_repo.get_event_claims(event_id)
  → Cypher: MATCH (e:Event)-[SUPPORTS]->(c:Claim)
  → Returns List[Claim] with minimal data
  → Optional: Hydrate from PostgreSQL for full metadata
```

## Key Architectural Decisions

### Why PostgreSQL + Neo4j?

**PostgreSQL strengths**:
- Excellent for document storage (pages, full claims)
- Vector search with pgvector (embeddings)
- JSONB for flexible metadata
- Audit trail - nothing is deleted

**Neo4j strengths**:
- Fast graph traversal (event hierarchy, entity relationships)
- Flexible schema for evolving relationships
- Cypher query language for complex graph patterns
- Visual exploration and debugging

### Why not store ALL claims in Neo4j?

**Space efficiency**: Typical page has 10-50 claims, but only 2-5 are event-relevant
- PostgreSQL stores 100% of claims (audit trail)
- Neo4j stores ~10-20% (curated knowledge)

**Query performance**: Neo4j optimized for relationships, not bulk storage
- Event→Claims is fast graph traversal
- Searching ALL claims uses PostgreSQL vector search

### Why not use claim_ids in Event metadata?

**Graph native**: Relationships are first-class in Neo4j
- `MATCH (e:Event)-[SUPPORTS]->(c:Claim)` is natural
- Adding/removing claims updates graph, not JSON

**Query flexibility**:
```cypher
// Get all claims for an event and its sub-events (recursive)
MATCH (parent:Event {id: $id})-[:CONTAINS*0..]->(event:Event)-[:SUPPORTS]->(claim:Claim)
RETURN claim
```

**Relationship metadata**: Can add properties to relationships
```cypher
Event-[SUPPORTS {confidence: 0.9, timestamp: ...}]->Claim
Event-[CONTRADICTS {evidence: "..."}]->Claim
```

## Testing the Architecture

### Verify separation of concerns:
```bash
# Services should NOT have direct DB access
grep -r "self.neo4j\." services/ --include="*.py"
# Should return ZERO results

# Repositories should be the ONLY place with storage logic
grep -r "CREATE (.*:Claim" --include="*.py"
# Should only appear in repositories/
```

### Verify data flow:
```bash
# 1. Check page in PostgreSQL
psql -c "SELECT id, title, status FROM core.pages WHERE url LIKE '%cnn.io%';"

# 2. Check claims in PostgreSQL (data lake)
psql -c "SELECT COUNT(*) FROM core.claims WHERE page_id = '...';"

# 3. Check claims in Neo4j (knowledge graph - subset)
cypher-shell "MATCH (c:Claim) RETURN count(c);"

# 4. Check Event-Claim relationships
cypher-shell "MATCH (e:Event)-[r:SUPPORTS]->(c:Claim) RETURN e.canonical_name, count(c);"
```

## Migration Notes

### Existing Data
Old events may have `claim_ids` in metadata JSON. These will be ignored.
New events use graph relationships exclusively.

To migrate:
```python
# For each old event with claim_ids in metadata:
for claim_id in event.metadata.get('claim_ids', []):
    await event_repo.link_claim(event, claim)
```

### Backward Compatibility
- Event model still has `claims_count` field (legacy)
- Set to 0 on load, use `get_event_claims()` for actual count
- Will be removed in future version

## Future Enhancements

### Page Nodes in Neo4j
Currently pages are PostgreSQL-only. Consider adding minimal Page nodes:
```cypher
Claim-[FROM_PAGE]->Page-[FROM_DOMAIN]->Domain
```

Benefits:
- Track source provenance in graph
- Query: "Which pages contributed to this event?"
- Detect source diversity/bias

### Claim Evolution
Track how claims evolve over time:
```cypher
Claim-[UPDATES {reason: "death toll increased"}]->Claim
```

### Event Causality
```cypher
Event-[CAUSED {confidence: 0.8}]->Event
Event-[TRIGGERED]->Event
```

## Maintenance

### Health Checks
1. PostgreSQL: Check for orphaned claims (page deleted but claims remain)
2. Neo4j: Check for orphaned Claim nodes (not linked to any Event)
3. Consistency: Verify Claim IDs in Neo4j exist in PostgreSQL

### Performance Monitoring
1. PostgreSQL: Vector search performance on `core.claims.embedding`
2. Neo4j: Graph traversal depth (event hierarchy depth)
3. Worker throughput: Pages/minute, Claims/minute, Events/hour

### Scaling
- PostgreSQL: Partition `core.claims` by created_at
- Neo4j: Shard by event_type or geographic region
- Workers: Horizontal scaling (add more containers)
