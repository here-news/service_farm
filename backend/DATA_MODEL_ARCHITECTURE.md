# Data Model Architecture

## Overview

This system uses **storage-agnostic domain models** with a **repository pattern** to abstract PostgreSQL and Neo4j storage details from business logic.

## Architecture Principles

### 1. **Domain Models** (`backend/models/`)

Pure Python dataclasses representing business entities:

- **Page**: News article/page
- **Claim**: Factual assertion extracted from page
- **Entity**: Named entity (person, org, location)
- **Event**: Real-world event formed from multiple pages
- **Phase**: Semantic phase within an event
- **Relationships**: Links between entities (ClaimEntityLink, PhaseClaimLink, etc.)

**Key features:**
- Storage-agnostic (no database-specific code)
- Type-safe with dataclasses
- Business logic methods (e.g., `event.update_status_from_page_count()`)
- Easy to test and reason about

### 2. **Repository Pattern** (`backend/repositories/`)

Repositories abstract storage operations:

- **EntityRepository**: CRUD for entities (PostgreSQL + Neo4j)
- **ClaimRepository**: CRUD for claims (PostgreSQL + Neo4j)
- **EventRepository**: CRUD for events (PostgreSQL + Neo4j)
- **PhaseRepository**: CRUD for phases (PostgreSQL + Neo4j)

**Benefits:**
- Business logic doesn't know about databases
- Easy to switch storage backends
- Centralized storage logic
- Handles dual-write (PostgreSQL + Neo4j) transparently

### 3. **Storage Strategy**

#### PostgreSQL (Content + Embeddings)
- **Purpose**: Store content, metadata, embeddings (pgvector)
- **Tables**:
  - `core.pages`: Page content, metadata
  - `core.claims`: Claim text, embeddings
  - `core.entities`: Entity metadata, aliases
  - `core.events`: Event metadata, embedding (mean of pages)
  - `core.event_phases`: Phase embeddings (mean of claims)
  - `core.claim_entities`: Join table for claim-entity relationships

#### Neo4j (Graph Structure)
- **Purpose**: Store relationships and graph structure
- **Nodes**:
  - `Event`: Event nodes
  - `Phase`: Phase nodes
  - `Claim`: Claim nodes
  - `Entity`: Entity nodes
  - `Page`: Page nodes (for tracking)
- **Relationships**:
  - `(Event)-[:HAS_PHASE]->(Phase)`
  - `(Phase)-[:SUPPORTED_BY]->(Claim)`
  - `(Claim)-[:FROM_PAGE]->(Page)`
  - `(Claim)-[:MENTIONS|ACTOR|SUBJECT|LOCATION]->(Entity)`

## Dual-Write Strategy

Some entities are stored in **both** PostgreSQL and Neo4j:

| Entity | PostgreSQL | Neo4j | Why Both? |
|--------|-----------|-------|-----------|
| Event | ✅ (metadata + embedding) | ✅ (graph node) | Need embedding for similarity, graph for relationships |
| Phase | ✅ (embedding) | ✅ (graph node) | Need embedding for matching, graph for claim links |
| Claim | ✅ (content + embedding) | ✅ (graph node) | Need embedding for phase computation, graph for entity links |
| Entity | ✅ (metadata + aliases) | ✅ (graph node) | Need metadata for resolution, graph for claim relationships |
| Page | ✅ (content + embedding) | ✅ (tracking node) | Need content/embedding for event formation, graph for tracking |

**Repositories handle dual-write transparently** - calling `entity_repo.create(entity)` writes to both databases.

## Usage Example

### Before (Direct Database Access)
```python
# Old way: Direct database queries, dict manipulation
async with db_pool.acquire() as conn:
    row = await conn.fetchrow("SELECT * FROM core.entities WHERE id = $1", entity_id)
    entity_dict = dict(row)
    # ... lots of dict manipulation
    await neo4j.execute("MERGE (e:Entity {id: $id, ...})", entity_dict)
```

### After (Domain Models + Repositories)
```python
# New way: Clean domain models
entity = Entity(
    id=uuid.uuid4(),
    canonical_name="Hong Kong Fire Services",
    entity_type="ORG",
    mention_count=0
)

# Repository handles dual-write
entity_repo = EntityRepository(db_pool, neo4j_service)
created_entity = await entity_repo.create(entity)

# Business logic operates on models
if created_entity.is_organization:
    created_entity.add_alias("HKFSD")
    await entity_repo.update(created_entity)
```

## Benefits of This Architecture

1. **Separation of Concerns**
   - Business logic: Works with domain models
   - Storage logic: Isolated in repositories
   - Database-specific code: Hidden from workers

2. **Type Safety**
   - Dataclasses provide IDE autocomplete
   - Catch errors at development time
   - Self-documenting code

3. **Testability**
   - Easy to mock repositories
   - Test business logic without databases
   - Integration tests use real repositories

4. **Flexibility**
   - Change PostgreSQL schema without touching workers
   - Switch from Neo4j to another graph DB
   - Add caching layer in repositories

5. **Maintainability**
   - Clear contracts (model interfaces)
   - Single source of truth (repository)
   - Easy to onboard new developers

## Migration Path

We're gradually migrating to this architecture:

1. ✅ **Created domain models** (Page, Claim, Entity, Event, Phase)
2. ✅ **Created repositories** (EntityRepository, ClaimRepository, etc.)
3. ⏳ **Update semantic_worker** to use repositories
4. ⏳ **Update event_worker_neo4j** to use repositories
5. ⏳ **Update APIs** to return domain models

## File Structure

```
backend/
├── models/                    # Domain models (storage-agnostic)
│   ├── __init__.py
│   ├── page.py               # Page model
│   ├── claim.py              # Claim model
│   ├── entity.py             # Entity model
│   ├── event.py              # Event model
│   ├── phase.py              # Phase model
│   └── relationships.py      # Relationship models
│
├── repositories/              # Storage abstraction
│   ├── __init__.py
│   ├── entity_repository.py  # Entity CRUD (PostgreSQL + Neo4j)
│   ├── claim_repository.py   # Claim CRUD (PostgreSQL + Neo4j)
│   ├── event_repository.py   # Event CRUD (PostgreSQL + Neo4j)
│   └── phase_repository.py   # Phase CRUD (PostgreSQL + Neo4j)
│
├── workers/                   # Business logic (uses models + repos)
│   ├── semantic_worker.py    # Extract claims/entities
│   └── event_worker_neo4j.py # Form events from pages
│
└── services/                  # Low-level services
    └── neo4j_service.py      # Neo4j Cypher operations
```

## Next Steps

- Update `semantic_worker.py` to use `EntityRepository` and `ClaimRepository`
- Refactor `event_worker_neo4j.py` to use `EventRepository` and `PhaseRepository`
- Create API endpoints that return domain models directly
- Add caching layer in repositories for frequently accessed entities
