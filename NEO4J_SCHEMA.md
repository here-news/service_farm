# Neo4j Event Graph Schema

## Design Principles

1. **Events are graphs, not tables**: Phases, causality, temporal evolution are natural graph structures
2. **Scaffold pattern**: Events start from earliest observation, grow as information arrives
3. **Temporal edges**: Time is explicit in relationships (BEFORE, AFTER, EVOLVED_TO)
4. **Evidence-based**: Claims are nodes with confidence, linked to sources
5. **Narrative generation**: Graph traversal → structured narrative

## Node Types

### Event
Core event entity (e.g., "2025 Hong Kong Tai Po Fire")

```cypher
(:Event {
  id: "uuid",
  canonical_name: "2025 Hong Kong Tai Po High-Rise Fire",
  event_type: "FIRE|EARTHQUAKE|SHOOTING|...",
  status: "provisional|emerging|stable",
  confidence: 0.95,
  event_scale: "micro|meso|macro",
  earliest_time: datetime,
  latest_time: datetime,
  created_at: datetime,
  updated_at: datetime
})
```

### Phase
Temporal phase within an event (e.g., "Fire Breakout", "Emergency Response", "Investigation")

```cypher
(:Phase {
  id: "uuid",
  name: "Fire Breakout|Emergency Response|Casualty Assessment|Investigation|...",
  phase_type: "INCIDENT|RESPONSE|CONSEQUENCE|INVESTIGATION|POLITICAL",
  start_time: datetime,
  end_time: datetime,
  confidence: 0.9
})
```

### Claim
Factual assertion from a source

```cypher
(:Claim {
  id: "uuid",
  text: "Fire started at approximately 17:59 in Tai Po district",
  modality: "observation|reported_speech|allegation|opinion",
  confidence: 0.95,
  event_time: datetime,
  extracted_at: datetime
})
```

### Entity
People, organizations, locations

```cypher
(:Person|:Organization|:Location {
  id: "uuid",
  canonical_name: "John Lee",
  entity_type: "PERSON|ORG|GPE|LOC",
  wikidata_qid: "Q12345",
  mention_count: 15
})

// Entity hierarchies
(:Person {name: "John Lee"})-[:HAS_ROLE]->(:Role {title: "Hong Kong Chief Executive"})
(:Location {name: "Tai Po"})-[:PART_OF]->(:Location {name: "Hong Kong"})
```

### Page
Source document (stored in PostgreSQL, referenced in Neo4j)

```cypher
(:Page {
  id: "uuid",
  url: "https://...",
  pub_time: datetime,
  source: "BBC|CNN|..."
})
```

## Relationship Types

### Event Structure

```cypher
// Event phases
(:Event)-[:HAS_PHASE {sequence: 1}]->(:Phase)

// Sub-events (causality)
(:Event)-[:CAUSED]->(:Event)  // Fire caused Investigation
(:Event)-[:TRIGGERED]->(:Event)  // Fire triggered Political Response
(:Event)-[:RELATED_TO {type: "SIMILAR"}]->(:Event)  // Similar to Grenfell fire

// Temporal relationships between phases
(:Phase)-[:BEFORE]->(:Phase)
(:Phase)-[:CONCURRENT_WITH]->(:Phase)
```

### Evidence Relationships

```cypher
// Claims attached to phases
(:Phase)-[:SUPPORTED_BY {confidence: 0.9}]->(:Claim)

// Claims from pages
(:Claim)-[:FROM_PAGE]->(:Page)

// Claims reference entities
(:Claim)-[:MENTIONS]->(:Entity)
(:Claim)-[:ACTOR]->(:Entity)  // Who performed action
(:Claim)-[:SUBJECT]->(:Entity)  // Who/what was affected
(:Claim)-[:LOCATION]->(:Location)  // Where it happened

// Claim relationships (contradiction, corroboration)
(:Claim)-[:CORROBORATES {confidence: 0.8}]->(:Claim)
(:Claim)-[:CONTRADICTS]->(:Claim)
(:Claim)-[:EVOLVED_TO]->(:Claim)  // "4 dead" evolved to "128 dead"
```

### Entity Relationships

```cypher
(:Person)-[:HAS_ROLE]->(:Role)
(:Person)-[:AFFILIATED_WITH]->(:Organization)
(:Location)-[:PART_OF]->(:Location)
(:Organization)-[:BASED_IN]->(:Location)
```

## Example Graph: Hong Kong Fire 2025

```cypher
// Main event
(fire:Event {
  canonical_name: "2025 Hong Kong Tai Po High-Rise Fire",
  event_type: "FIRE",
  earliest_time: datetime("2025-11-26T17:59:00Z")
})

// Phases
(breakout:Phase {name: "Fire Breakout", start_time: "2025-11-26T17:59:00Z"})
(response:Phase {name: "Emergency Response", start_time: "2025-11-26T18:00:00Z"})
(casualties:Phase {name: "Casualty Assessment", start_time: "2025-11-26T18:30:00Z"})
(investigation:Phase {name: "Investigation", start_time: "2025-11-27T00:00:00Z"})
(arrests:Phase {name: "Arrests", start_time: "2025-11-28T00:00:00Z"})

// Phase structure
(fire)-[:HAS_PHASE {sequence: 1}]->(breakout)
(fire)-[:HAS_PHASE {sequence: 2}]->(response)
(fire)-[:HAS_PHASE {sequence: 3}]->(casualties)
(breakout)-[:BEFORE]->(response)
(response)-[:BEFORE]->(casualties)

// Sub-events
(investEvent:Event {canonical_name: "Hong Kong Fire Investigation"})
(fire)-[:CAUSED]->(investEvent)
(investEvent)-[:HAS_PHASE]->(investigation)
(investEvent)-[:HAS_PHASE]->(arrests)

// Claims
(claim1:Claim {text: "Fire started at Wang Fuk Court", modality: "observation"})
(claim2:Claim {text: "4 people confirmed dead", modality: "reported_speech"})
(claim3:Claim {text: "128 people confirmed dead", modality: "reported_speech"})

(breakout)-[:SUPPORTED_BY]->(claim1)
(casualties)-[:SUPPORTED_BY]->(claim2)
(casualties)-[:SUPPORTED_BY]->(claim3)
(claim2)-[:EVOLVED_TO]->(claim3)

// Entities
(taiPo:Location {canonical_name: "Tai Po"})
(hongKong:Location {canonical_name: "Hong Kong"})
(fireDept:Organization {canonical_name: "Hong Kong Fire Services Department"})

(claim1)-[:LOCATION]->(taiPo)
(taiPo)-[:PART_OF]->(hongKong)
(response)-[:SUPPORTED_BY]->(:Claim)-[:ACTOR]->(fireDept)
```

## Query Patterns for Narrative Generation

### 1. Get Event Timeline
```cypher
MATCH (e:Event {id: $event_id})-[:HAS_PHASE]->(p:Phase)
OPTIONAL MATCH (p)-[:SUPPORTED_BY]->(c:Claim)
RETURN p, collect(c) as claims
ORDER BY p.start_time
```

### 2. Get Casualty Evolution
```cypher
MATCH (e:Event {id: $event_id})-[:HAS_PHASE]->(p:Phase {name: "Casualty Assessment"})
      -[:SUPPORTED_BY]->(c1:Claim)
MATCH (c1)-[:EVOLVED_TO*]->(c2:Claim)
RETURN c1, c2
ORDER BY c1.event_time
```

### 3. Get Causality Chain
```cypher
MATCH path = (e:Event {id: $event_id})-[:CAUSED|TRIGGERED*]->(sub:Event)
RETURN path
```

### 4. Get Entity Context
```cypher
MATCH (e:Entity {canonical_name: "John Lee"})-[:HAS_ROLE]->(r:Role)
MATCH (e)<-[:ACTOR]-(c:Claim)
RETURN e, r, collect(c) as statements
```

## API Endpoints to Support

### `/api/events/{id}/graph`
Returns Neo4j subgraph for visualization

### `/api/events/{id}/narrative`
Generates narrative from graph traversal:
```json
{
  "summary": "One-sentence overview",
  "timeline": [
    {"phase": "Fire Breakout", "start": "...", "description": "...", "claims": [...]}
  ],
  "casualties": {
    "evolution": ["4 dead", "36 dead", "128 dead"],
    "final": 128
  },
  "investigation": {...},
  "key_entities": [...]
}
```

### `/api/events/{id}/timeline`
Timeline view for frontend

### `/api/events/{id}/map`
Geospatial view (locations from graph)

## Migration Strategy

1. **Keep PostgreSQL for:**
   - Pages (raw content, HTML)
   - Generated narratives (cached)
   - Embeddings (vectors)

2. **Move to Neo4j:**
   - Events, phases, sub-events
   - Claims (as graph nodes)
   - Entities + relationships
   - All temporal/causal edges

3. **Sync approach:**
   - Event worker writes to Neo4j (source of truth)
   - API reads from Neo4j for structure
   - Generated narratives cached in PostgreSQL
