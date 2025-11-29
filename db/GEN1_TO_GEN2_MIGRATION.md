# Gen1 to Gen2 Migration Mapping

**Purpose:** Define how to migrate data and patterns from Gen1 (Neo4j + PostgreSQL hybrid) to Gen2 (PostgreSQL-only)

**Date:** 2025-11-29
**Status:** Draft

---

## Migration Overview

### What We're Migrating

**Gen1 Architecture:**
- PostgreSQL: `extraction_tasks` table + user tables
- Neo4j: `Page`, `Claim`, `Person`, `Organization`, `Location`, `MediaSource`, `Story` nodes + relationships

**Gen2 Architecture:**
- PostgreSQL only: `core.*` schema for knowledge graph
- No Neo4j

---

## Table-by-Table Mapping

### 1. PostgreSQL → PostgreSQL (Direct Migration)

#### `extraction_tasks` → `core.pages`

**Gen1:**
```sql
CREATE TABLE extraction_tasks (
    id UUID PRIMARY KEY,
    url TEXT,
    canonical_url TEXT UNIQUE,
    status VARCHAR(50),
    result JSONB,
    semantic_data JSONB,
    metadata JSONB,
    word_count INT,
    error_message TEXT,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);
```

**Gen2:**
```sql
CREATE TABLE core.pages (
    id UUID PRIMARY KEY,
    url TEXT NOT NULL,
    canonical_url TEXT UNIQUE NOT NULL,
    title TEXT,
    content_text TEXT,
    status VARCHAR(50) DEFAULT 'stub',
    word_count INT DEFAULT 0,
    error_message TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    embedding vector(1536)
);
```

**Migration Strategy:**
```sql
-- 1. Direct column mapping
INSERT INTO core.pages (
    id,
    url,
    canonical_url,
    status,
    word_count,
    error_message,
    metadata,
    created_at,
    updated_at,
    completed_at
)
SELECT
    id,
    url,
    canonical_url,
    status,
    word_count,
    error_message,
    metadata,
    created_at,
    updated_at,
    completed_at
FROM gen1.extraction_tasks;

-- 2. Extract title and content_text from result JSONB
UPDATE core.pages
SET
    title = (
        SELECT result->'meta'->>'title'
        FROM gen1.extraction_tasks
        WHERE gen1.extraction_tasks.id = core.pages.id
    ),
    content_text = (
        SELECT result->>'content_text'
        FROM gen1.extraction_tasks
        WHERE gen1.extraction_tasks.id = core.pages.id
    );
```

**Fields to Drop:**
- ❌ `result` JSONB → Extract specific fields (title, content_text)
- ❌ `semantic_data` JSONB → Migrate to claims/entities tables

**New Fields (populate later):**
- `embedding` → Generate via worker after migration
- `language` → Re-detect or extract from metadata
- `byline`, `site_name`, `domain` → Extract from existing data

---

#### User Tables → `bridge.*` (webapp responsibility)

**Gen1 User Tables:**
- `users`
- `user_urls`
- `user_sessions`
- `credits_transactions`
- `page_annotations`

**Gen2 Strategy:**
- ✅ Keep Gen1 user tables as-is (webapp owns them)
- ✅ Create bridge table for artifact ownership

**Migration:**
```sql
-- Create bridge metadata from user_urls
INSERT INTO bridge.artifact_metadata (
    artifact_id,
    submitted_by_id,
    submission_source,
    submitted_at
)
SELECT
    task_id AS artifact_id,  -- Assumes task_id = page_id
    user_id AS submitted_by_id,
    'browser_extension' AS submission_source,
    submitted_at
FROM gen1.user_urls;
```

---

### 2. Neo4j → PostgreSQL (Data Extraction Required)

#### Neo4j `Page` Nodes → `core.pages` (enrich existing)

**Gen1 Neo4j Page:**
```cypher
(:Page {
    url: "...",
    canonical_url: "...",
    title: "...",
    byline: "...",
    pub_time: "...",
    site: "...",
    language: "en",
    gist: "...",
    task_id: "...",
    is_healthy: true
})
```

**Migration:**
```sql
-- Enrich core.pages with Neo4j Page data
UPDATE core.pages
SET
    title = neo4j_page.title,
    byline = neo4j_page.byline,
    pub_time = neo4j_page.pub_time,
    site_name = neo4j_page.site,
    language = neo4j_page.language,
    is_healthy = neo4j_page.is_healthy
FROM (
    -- Export from Neo4j as CSV/JSON, then load
    SELECT * FROM neo4j_export.pages
) AS neo4j_page
WHERE core.pages.canonical_url = neo4j_page.canonical_url;
```

**Neo4j Export Query:**
```cypher
MATCH (p:Page)
RETURN
    p.url AS url,
    p.canonical_url AS canonical_url,
    p.title AS title,
    p.byline AS byline,
    p.pub_time AS pub_time,
    p.site AS site,
    p.language AS language,
    p.gist AS gist,
    p.is_healthy AS is_healthy
ORDER BY p.fetch_time DESC
```

---

#### Neo4j Entity Nodes → `core.entities`

**Gen1 Neo4j Entities:**
```cypher
(:Person {
    canonical_id: "...",
    canonical_name: "...",
    wikidata_qid: "Q123",
    ...
})

(:Organization {
    canonical_id: "...",
    canonical_name: "...",
    wikidata_qid: "Q456",
    ...
})

(:Location {
    canonical_id: "...",
    canonical_name: "...",
    wikidata_qid: "Q789",
    ...
})
```

**Migration:**
```sql
-- Import from Neo4j export
INSERT INTO core.entities (
    canonical_name,
    entity_type,
    wikidata_qid,
    semantic_confidence,
    created_at
)
SELECT
    canonical_name,
    CASE node_label
        WHEN 'Person' THEN 'PERSON'::core.entity_type
        WHEN 'Organization' THEN 'ORGANIZATION'::core.entity_type
        WHEN 'Location' THEN 'LOCATION'::core.entity_type
        ELSE 'OTHER'::core.entity_type
    END,
    wikidata_qid,
    0.75,  -- Default semantic confidence
    created_at
FROM neo4j_export.entities;
```

**Neo4j Export Query:**
```cypher
MATCH (n)
WHERE n:Person OR n:Organization OR n:Location
RETURN
    labels(n)[0] AS node_label,
    n.canonical_id AS canonical_id,
    n.canonical_name AS canonical_name,
    n.wikidata_qid AS wikidata_qid
```

---

#### Neo4j `Claim` Nodes → `core.claims`

**Gen1 Neo4j Claims:**
```cypher
(:Claim {
    claim_id: "...",
    text: "...",
    event_time: "...",
    confidence: 0.8
})
```

**Migration:**
```sql
INSERT INTO core.claims (
    id,
    page_id,
    text,
    event_time,
    confidence,
    created_at
)
SELECT
    claim_id::uuid AS id,
    page_id::uuid,
    text,
    event_time::timestamptz,
    confidence,
    NOW() AS created_at
FROM neo4j_export.claims;
```

**Neo4j Export Query:**
```cypher
MATCH (p:Page)-[:HAS_CLAIM]->(c:Claim)
RETURN
    c.claim_id AS claim_id,
    p.url AS page_url,
    c.text AS text,
    c.event_time AS event_time,
    c.confidence AS confidence
```

---

#### Neo4j Relationships → `core.edges`

**Gen1 Neo4j Relationships:**
```cypher
(p:Person)-[:EMPLOYED_BY]->(o:Organization)
(l:Location)-[:PART_OF]->(parent:Location)
(c:Claim)-[:MENTIONS]->(e:Person|Organization|Location)
```

**Migration:**
```sql
-- Example: EMPLOYED_BY relationships
INSERT INTO core.edges (
    from_entity_id,
    to_entity_id,
    edge_type,
    confidence,
    source,
    first_seen
)
SELECT
    from_entity.id AS from_entity_id,
    to_entity.id AS to_entity_id,
    'EMPLOYED_BY'::core.edge_type,
    0.8 AS confidence,
    'neo4j_migration' AS source,
    NOW() AS first_seen
FROM neo4j_export.relationships r
JOIN core.entities from_entity ON from_entity.canonical_name = r.from_canonical_name
JOIN core.entities to_entity ON to_entity.canonical_name = r.to_canonical_name
WHERE r.relationship_type = 'EMPLOYED_BY';
```

**Neo4j Export Query:**
```cypher
MATCH (from)-[r]->(to)
WHERE (from:Person OR from:Organization OR from:Location)
  AND (to:Person OR to:Organization OR to:Location)
RETURN
    type(r) AS relationship_type,
    from.canonical_name AS from_canonical_name,
    to.canonical_name AS to_canonical_name,
    r.confidence AS confidence
```

---

#### Neo4j Page-Entity Relationships → `core.page_entities`

**Gen1 Neo4j:**
```cypher
(p:Page)-[:MENTIONS {count: 5}]->(e:Person)
```

**Migration:**
```sql
INSERT INTO core.page_entities (
    page_id,
    entity_id,
    mention_count
)
SELECT
    p.id AS page_id,
    e.id AS entity_id,
    r.mention_count
FROM neo4j_export.page_entity_mentions r
JOIN core.pages p ON p.canonical_url = r.page_url
JOIN core.entities e ON e.canonical_name = r.entity_canonical_name;
```

**Neo4j Export Query:**
```cypher
MATCH (p:Page)-[m:MENTIONS]->(e)
WHERE e:Person OR e:Organization OR e:Location
RETURN
    p.canonical_url AS page_url,
    e.canonical_name AS entity_canonical_name,
    m.count AS mention_count
```

---

### 3. What NOT to Migrate

#### ❌ Neo4j `Story` Nodes

**Rationale:** Stories are community-curated narratives, not system facts (ARCHITECTURE.md lines 138-142). Gen2 stops at events; stories are webapp's responsibility.

**Alternative:** Export Neo4j stories as JSON for webapp to import into `app.stories` table.

---

#### ❌ Neo4j `MediaSource` Nodes

**Rationale:** Gen2 doesn't have first-class media source entities. Domain/site info stored in `core.pages.domain`.

**Alternative:** Extract domain from page URL, no separate entity needed.

---

## Migration Phases

### Phase 1: Core Data (1 week)

**Goal:** Migrate pages, entities, claims without relationships

1. Export Neo4j data to CSV/JSON
2. Migrate `extraction_tasks` → `core.pages`
3. Migrate Neo4j `Person/Org/Location` → `core.entities`
4. Migrate Neo4j `Claim` → `core.claims`
5. Validate row counts

### Phase 2: Relationships (1 week)

**Goal:** Migrate Neo4j relationships to `core.edges`

1. Export Neo4j relationships
2. Map relationship types to `core.edge_type` enum
3. Insert into `core.edges`
4. Insert into `core.page_entities`, `core.claim_entities`
5. Validate graph connectivity

### Phase 3: Enrichment (1 week)

**Goal:** Generate embeddings, detect languages, compute confidence

1. Generate embeddings for entities/claims/events
2. Re-detect languages for pages
3. Compute structural confidence for entities
4. Run event clustering worker on migrated data
5. Validate event formation quality

### Phase 4: User Linkage (3 days)

**Goal:** Connect user tables to new core schema

1. Create `bridge.artifact_metadata` from `user_urls`
2. Update webapp to query `bridge.artifacts_for_users` view
3. Test bookmarks, annotations integration
4. Validate user→artifact relationships

---

## Migration Scripts

### Export from Neo4j

```bash
# Export all nodes and relationships
cypher-shell -u neo4j -p $NEO4J_PASSWORD < export_entities.cypher > entities.csv
cypher-shell -u neo4j -p $NEO4J_PASSWORD < export_claims.cypher > claims.csv
cypher-shell -u neo4j -p $NEO4J_PASSWORD < export_relationships.cypher > relationships.csv
```

### Import to PostgreSQL

```bash
# Load CSVs into temp tables
psql -U postgres -d herenews < load_temp_tables.sql

# Run migration SQL
psql -U postgres -d herenews < migrate_entities.sql
psql -U postgres -d herenews < migrate_claims.sql
psql -U postgres -d herenews < migrate_edges.sql
```

### Validation Queries

```sql
-- Count comparison
SELECT 'Gen1 extraction_tasks' AS source, COUNT(*) FROM gen1.extraction_tasks
UNION ALL
SELECT 'Gen2 core.pages', COUNT(*) FROM core.pages;

-- Neo4j entity count (from export)
SELECT 'Neo4j entities (export)', COUNT(*) FROM neo4j_export.entities
UNION ALL
SELECT 'Gen2 core.entities', COUNT(*) FROM core.entities;

-- Orphan check (entities with no mentions)
SELECT COUNT(*)
FROM core.entities e
WHERE NOT EXISTS (
    SELECT 1 FROM core.page_entities pe WHERE pe.entity_id = e.id
);
```

---

## Rollback Plan

**If migration fails:**

1. **Backup:** Full PostgreSQL dump before migration
   ```bash
   pg_dump -U postgres herenews > backup_pre_gen2.sql
   ```

2. **Restore:** Drop Gen2 schemas, restore backup
   ```sql
   DROP SCHEMA core CASCADE;
   DROP SCHEMA bridge CASCADE;
   DROP SCHEMA system CASCADE;
   -- Then restore from backup
   ```

3. **Validation:** Verify Gen1 data integrity
   ```sql
   SELECT COUNT(*) FROM gen1.extraction_tasks;
   ```

---

## Success Criteria

- ✅ All `extraction_tasks` migrated to `core.pages` (100%)
- ✅ All Neo4j entities migrated to `core.entities` (>95%)
- ✅ All Neo4j claims migrated to `core.claims` (>95%)
- ✅ Page-entity relationships preserved (>90%)
- ✅ Event clustering produces meaningful hierarchies
- ✅ Webapp can query via `bridge.*` views
- ✅ No data loss (validated via checksums)

---

## Next Steps

1. Write Neo4j export Cypher queries
2. Create temp staging tables in PostgreSQL
3. Write migration SQL scripts
4. Test migration on dev database (1000 rows)
5. Run full migration on staging
6. Validate with Gen1 vs Gen2 comparison queries
