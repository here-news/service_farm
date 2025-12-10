// Neo4j Schema for Knowledge Graph
// Architecture: Neo4j holds graph structure, Postgres holds content/embeddings
//
// Node Types: Page, Claim, Entity, Event
// Relationships: EXTRACTED, MENTIONS, CORROBORATES, SUPPORTS, INVOLVES

// =============================================================================
// CONSTRAINTS - Ensure uniqueness
// =============================================================================

// Pages
CREATE CONSTRAINT page_id_unique IF NOT EXISTS
FOR (p:Page) REQUIRE p.id IS UNIQUE;

// Claims
CREATE CONSTRAINT claim_id_unique IF NOT EXISTS
FOR (c:Claim) REQUIRE c.id IS UNIQUE;

// Entities
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;

// Events
CREATE CONSTRAINT event_id_unique IF NOT EXISTS
FOR (e:Event) REQUIRE e.id IS UNIQUE;

// =============================================================================
// INDEXES - Performance for common queries
// =============================================================================

// Page indexes
CREATE INDEX page_url IF NOT EXISTS
FOR (p:Page) ON (p.url);

CREATE INDEX page_status IF NOT EXISTS
FOR (p:Page) ON (p.status);

// Entity indexes
CREATE INDEX entity_canonical_name IF NOT EXISTS
FOR (e:Entity) ON (e.canonical_name);

CREATE INDEX entity_type IF NOT EXISTS
FOR (e:Entity) ON (e.entity_type);

CREATE INDEX entity_wikidata_qid IF NOT EXISTS
FOR (e:Entity) ON (e.wikidata_qid);

// Event indexes
CREATE INDEX event_canonical_name IF NOT EXISTS
FOR (e:Event) ON (e.canonical_name);

CREATE INDEX event_coherence IF NOT EXISTS
FOR (e:Event) ON (e.coherence);

// =============================================================================
// NODE SCHEMAS (Property patterns)
// =============================================================================

// -----------------------------------------------------------------------------
// Page Node
// -----------------------------------------------------------------------------
// Properties:
//   id: VARCHAR(20) - pg_xxxxxxxx format
//   url: TEXT - Original URL
//   status: VARCHAR(50) - pending, extracted, knowledge_complete, event_complete
//   created_at: DATETIME
//   updated_at: DATETIME
//
// Relationships:
//   (Page)-[:EXTRACTED]->(Claim)
//
// Note: Content and embedding stored in Postgres core.pages
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Claim Node
// -----------------------------------------------------------------------------
// Properties:
//   id: VARCHAR(20) - cl_xxxxxxxx format
//   text: TEXT - Claim text (truncated to 500 chars)
//   deterministic_id: TEXT - Hash for deduplication
//   event_time: DATETIME - When the claim describes
//   confidence: FLOAT - Extraction confidence (0.0-1.0)
//   modality: VARCHAR(50) - factual, hypothetical, opinion
//   created_at: DATETIME
//
// Relationships:
//   (Page)-[:EXTRACTED]->(Claim)
//   (Claim)-[:MENTIONS]->(Entity)
//   (Claim)-[:CORROBORATES {similarity: FLOAT}]->(Claim)
//   (Event)-[:SUPPORTS]->(Claim)
//
// Note: Embedding stored in Postgres content.claim_embeddings
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Entity Node
// -----------------------------------------------------------------------------
// Properties:
//   id: VARCHAR(20) - en_xxxxxxxx format
//   canonical_id: VARCHAR(20) - Resolved canonical entity ID
//   canonical_name: TEXT - Resolved name
//   entity_type: VARCHAR(50) - PERSON, ORGANIZATION, LOCATION, etc.
//   confidence: FLOAT - Entity resolution confidence
//   mention_count: INT - Number of times mentioned
//   wikidata_qid: VARCHAR(20) - Wikidata Q-ID (e.g., Q8646)
//   wikidata_label: TEXT - Wikidata label
//   wikidata_description: TEXT - Wikidata description
//   wikidata_image: TEXT - Wikidata image URL
//   profile_summary: TEXT - Entity profile (optional)
//   status: VARCHAR(50) - stub, enriched, canonical
//   created_at: DATETIME
//   updated_at: DATETIME
//
// Relationships:
//   (Claim)-[:MENTIONS]->(Entity)
//   (Entity)-[:CANONICAL_OF]->(Entity) - For entity resolution
//   (Event)-[:INVOLVES]->(Entity)
//
// Note: Entity embeddings stored in Postgres core.entities
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Event Node (Event Organism)
// -----------------------------------------------------------------------------
// Properties:
//   id: VARCHAR(20) - ev_xxxxxxxx format
//   canonical_name: TEXT - Event name
//   event_type: VARCHAR(50) - INCIDENT, DEVELOPMENT, STATEMENT, etc.
//   event_scale: VARCHAR(20) - micro, local, regional, global
//   status: VARCHAR(50) - provisional, confirmed, established
//   confidence: FLOAT - Event confidence
//   coherence: FLOAT - Structural coherence score (0.0-1.0)
//   summary: TEXT - Generated narrative
//   location: TEXT - Geographic location (optional)
//   event_start: DATETIME - Start time (optional)
//   event_end: DATETIME - End time (optional)
//   parent_event_id: VARCHAR(20) - Parent event for recursive structure
//   claims_count: INT - Number of supporting claims
//   created_at: DATETIME
//   updated_at: DATETIME
//
// Relationships:
//   (Event)-[:SUPPORTS]->(Claim)
//   (Event)-[:INVOLVES]->(Entity)
//   (Event)-[:CONTAINS]->(Event) - Recursive sub-events
//
// Coherence Calculation:
//   coherence = 0.6 * hub_coverage + 0.4 * graph_connectivity
//   - hub_coverage: % of claims touching hub entities (3+ mentions)
//   - graph_connectivity: 1.0 / connected_components in claim-entity graph
// -----------------------------------------------------------------------------

// =============================================================================
// RELATIONSHIP SCHEMAS
// =============================================================================

// (Page)-[:EXTRACTED]->(Claim)
// Properties:
//   created_at: DATETIME

// (Claim)-[:MENTIONS]->(Entity)
// Properties:
//   created_at: DATETIME
// Side effect: Increments Entity.mention_count

// (Claim)-[:CORROBORATES]->(Claim)
// Properties:
//   similarity: FLOAT - Embedding cosine similarity (0.85-1.0)
//   created_at: DATETIME
// Detection: Claims with embedding similarity > 0.85

// (Event)-[:SUPPORTS]->(Claim)
// Properties:
//   relatedness_score: FLOAT - Multi-signal relatedness
//   created_at: DATETIME
// Signals: semantic similarity, entity overlap, temporal proximity

// (Event)-[:INVOLVES]->(Entity)
// Properties:
//   created_at: DATETIME
// Derived from: Claims supported by event that mention this entity

// (Event)-[:CONTAINS]->(Event)
// Properties:
//   created_at: DATETIME
// For recursive event structure (future feature)

// (Entity)-[:CANONICAL_OF]->(Entity)
// Properties:
//   confidence: FLOAT - Resolution confidence
//   created_at: DATETIME
// For entity resolution and deduplication

// =============================================================================
// NOTES
// =============================================================================
//
// Architecture Philosophy:
// - Neo4j: Graph structure, relationships, graph-native queries
// - Postgres: Content storage, vector embeddings, text search
//
// This separation allows:
// - Fast graph traversals in Neo4j (Cypher)
// - Efficient vector search in Postgres (pgvector)
// - Content storage without graph database bloat
//
// See also: backend/migrations/002_add_core_schema.sql
