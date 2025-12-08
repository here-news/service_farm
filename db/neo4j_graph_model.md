# Neo4j Graph Model (Single Source of Truth)

## Node Types

### Page
Stores all page metadata. Content stored in PostgreSQL.
```cypher
(:Page {
    id: UUID,              -- Primary key, matches PostgreSQL content.pages.id
    url: String,           -- Original URL
    canonical_url: String, -- Normalized URL for dedup
    title: String,         -- Page title
    description: String,   -- Meta description
    author: String,        -- Author name
    thumbnail_url: String, -- Featured image
    domain: String,        -- Source domain
    language: String,      -- Detected language
    word_count: Int,       -- Content word count
    pub_time: DateTime,    -- Publication timestamp
    metadata_confidence: Float,
    status: String,        -- 'stub' | 'extracted' | 'knowledge_complete' | 'failed'
    created_at: DateTime,
    updated_at: DateTime
})
```

### Claim
Atomic factual statements extracted from pages.
```cypher
(:Claim {
    id: UUID,
    text: String,          -- The claim text (max ~500 chars for graph)
    event_time: DateTime,  -- When the claim event occurred
    confidence: Float,     -- Extraction confidence
    modality: String,      -- 'factual' | 'alleged' | 'opinion'
    created_at: DateTime
})
```

### Entity
Named entities with Wikidata enrichment.
```cypher
(:Entity {
    id: UUID,
    canonical_name: String,     -- Primary name
    entity_type: String,        -- 'PERSON' | 'ORGANIZATION' | 'LOCATION'
    wikidata_qid: String,       -- Wikidata QID (e.g., 'Q8646' for Hong Kong)
    wikidata_label: String,     -- Official Wikidata label
    wikidata_description: String,
    wikidata_image: String,     -- Thumbnail URL
    profile_summary: String,    -- AI-generated profile
    aliases: [String],          -- Alternative names
    mention_count: Int,         -- How many claims mention this entity
    confidence: Float,          -- Wikidata match confidence
    status: String,             -- 'pending' | 'enriched' | 'checked'
    created_at: DateTime,
    updated_at: DateTime
})
```

### Event
Clustered news events.
```cypher
(:Event {
    id: UUID,
    title: String,           -- Event headline
    summary: String,         -- AI-generated summary
    event_time: DateTime,    -- When the event occurred
    status: String,          -- 'forming' | 'active' | 'resolved'
    claim_count: Int,
    source_count: Int,
    created_at: DateTime,
    updated_at: DateTime
})
```

### Source
News source/domain with credibility.
```cypher
(:Source {
    id: UUID,
    domain: String,          -- e.g., 'bbc.com'
    name: String,            -- Display name
    credibility: Float,      -- 0.0-1.0 credibility score
    created_at: DateTime
})
```

## Relationships

### Page Relationships
```cypher
(Page)-[:CONTAINS]->(Claim)      -- Page contains these claims
(Page)-[:FROM_SOURCE]->(Source)  -- Page is from this source
```

### Claim Relationships
```cypher
(Claim)-[:MENTIONS]->(Entity)    -- Claim mentions this entity
(Claim)-[:PART_OF]->(Event)      -- Claim is part of this event
```

### Event Relationships
```cypher
(Event)-[:INVOLVES]->(Entity)    -- Event involves this entity
```

### Entity Relationships
```cypher
(Entity)-[:LOCATED_IN]->(Entity)     -- Location hierarchy
(Entity)-[:AFFILIATED_WITH]->(Entity) -- Person/Org affiliation
```

## Constraints
```cypher
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT entity_qid IF NOT EXISTS FOR (e:Entity) REQUIRE e.wikidata_qid IS UNIQUE;
CREATE CONSTRAINT page_id IF NOT EXISTS FOR (p:Page) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT event_id IF NOT EXISTS FOR (ev:Event) REQUIRE ev.id IS UNIQUE;
CREATE CONSTRAINT source_domain IF NOT EXISTS FOR (s:Source) REQUIRE s.domain IS UNIQUE;
```

## Indexes
```cypher
CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.canonical_name);
CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type);
CREATE INDEX page_url IF NOT EXISTS FOR (p:Page) ON (p.url);
CREATE INDEX page_status IF NOT EXISTS FOR (p:Page) ON (p.status);
CREATE INDEX claim_time IF NOT EXISTS FOR (c:Claim) ON (c.event_time);
CREATE INDEX event_time IF NOT EXISTS FOR (ev:Event) ON (ev.event_time);
```

## Data Flow

1. **Extraction Worker**:
   - Creates Page node in Neo4j with metadata
   - Stores content_text in PostgreSQL content.pages

2. **Knowledge Worker**:
   - Creates Claim nodes in Neo4j
   - Creates Entity nodes (MERGE on QID for dedup)
   - Creates relationships (CONTAINS, MENTIONS)
   - Enriches entities with Wikidata

3. **Event Worker**:
   - Clusters claims into Event nodes
   - Creates INVOLVES relationships
