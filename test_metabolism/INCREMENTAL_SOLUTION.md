# Incremental Plausibility - Practical Solution

## Problem Identified

Neo4j doesn't support Map{} as property values. The error:
```
Property values can only be of primitive types or arrays thereof.
Encountered: Map{deaths -> Long(156), injured -> Long(12)}
```

## Solution Options

### Option 1: Store metrics as separate ClaimMetric nodes (RECOMMENDED)

```cypher
CREATE (c:Claim {id: 'cl_xxx', text: '...'})
CREATE (cm1:ClaimMetric {
  claim_id: 'cl_xxx',
  metric_type: 'deaths',
  value: 156,
  extracted_at: datetime()
})
CREATE (cm2:ClaimMetric {
  claim_id: 'cl_xxx',
  metric_type: 'injured',
  value: 12,
  extracted_at: datetime()
})
CREATE (c)-[:HAS_METRIC]->(cm1)
CREATE (c)-[:HAS_METRIC]->(cm2)
```

**Pros:**
- Natural graph structure
- Easy to query clusters: `MATCH (:Event)-[:SUPPORTS]->(:Claim)-[:HAS_METRIC]->(m:ClaimMetric {metric_type: 'deaths'})`
- Supports multiple metrics per claim
- Can add metadata per metric (confidence, source)

**Cons:**
- More nodes in graph
- Slightly more complex queries

### Option 2: Store as array properties

```cypher
CREATE (c:Claim {
  id: 'cl_xxx',
  text: '...',
  metric_types: ['deaths', 'injured'],
  metric_values: [156, 12]
})
```

**Pros:**
- Simpler structure
- Fewer nodes

**Cons:**
- Awkward to query (need to zip arrays)
- Hard to extend with metadata
- Not semantically clear

### Option 3: Store as flattened properties

```cypher
CREATE (c:Claim {
  id: 'cl_xxx',
  text: '...',
  metric_deaths: 156,
  metric_injured: 12,
  metric_missing: NULL
})
```

**Pros:**
- Simplest queries
- Easy to index

**Cons:**
- Schema explosion (one property per metric type)
- NULL pollution
- Hard to discover what metrics exist

## Recommended Implementation

Use **Option 1** with ClaimMetric nodes:

### Schema:

```cypher
// Claim node (unchanged)
CREATE (c:Claim {
  id: 'cl_xxxxxxxx',
  text: '...',
  confidence: 0.8,
  event_time: datetime(),

  // Materialized plausibility (updated incrementally)
  plausibility: 0.85,
  plausibility_updated_at: datetime()
})

// ClaimMetric nodes (one per metric)
CREATE (m:ClaimMetric {
  id: 'cm_xxxxxxxx',
  claim_id: 'cl_xxxxxxxx',
  metric_type: 'deaths',  // deaths, injured, missing, etc.
  value: 156,
  extracted_at: datetime(),
  extraction_confidence: 0.95
})

CREATE (c)-[:HAS_METRIC]->(m)
```

### Incremental Update Queries:

**1. Get metric cluster for event:**
```cypher
MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)-[:HAS_METRIC]->(m:ClaimMetric)
WHERE m.metric_type = $metric_type
RETURN c.id as claim_id,
       c.text as text,
       c.event_time as time,
       c.plausibility as plausibility,
       m.value as value
ORDER BY c.event_time
```

**2. Store extracted metrics:**
```cypher
// For each metric extracted
MATCH (c:Claim {id: $claim_id})
MERGE (m:ClaimMetric {claim_id: $claim_id, metric_type: $metric_type})
SET m.value = $value,
    m.extracted_at = datetime(),
    m.extraction_confidence = $confidence
MERGE (c)-[:HAS_METRIC]->(m)
```

**3. Update plausibility:**
```cypher
MATCH (c:Claim {id: $claim_id})
SET c.plausibility = $score,
    c.plausibility_updated_at = datetime()
```

### Incremental Update Algorithm:

```python
async def process_claim_incremental(event_id, claim_id, claim_text):
    # 1. Extract metrics (LLM call #1)
    metrics = await extract_metrics_llm(claim_text)
    # → {deaths: 156, injured: 12}

    # 2. Store metrics as ClaimMetric nodes
    for metric_type, value in metrics.items():
        await create_claim_metric(claim_id, metric_type, value)

    # 3. For each metric, check if re-analysis needed
    updated_scores = {}

    for metric_type, value in metrics.items():
        # Get cluster from graph
        cluster = await get_metric_cluster(event_id, metric_type)

        if needs_reanalysis(cluster, value):
            # LLM call #2 - analyze cluster
            analysis = await analyze_cluster_llm(event_id, metric_type)
            updated_scores.update(analysis['plausibility_scores'])
        else:
            # No LLM - incremental Bayesian update
            score = calculate_incremental_score(cluster, value)
            updated_scores[claim_id] = score

    # 4. Update plausibility scores
    await update_plausibility_batch(updated_scores)

    return updated_scores
```

### Performance:

For 100 claims with 3 metrics each:
- Batch: 100 claims × 7 metric types × re-analysis = ~700 LLM calls
- Incremental: 100 extraction + ~30 cluster analyses = ~130 LLM calls
- **Savings: 81%**

### Migration:

1. Create `ClaimMetric` nodes for existing claims (backfill)
2. Update `event_service.py` to use new schema
3. Add `plausibility` property to `Claim` nodes
4. Update narrative generation to sort by plausibility
