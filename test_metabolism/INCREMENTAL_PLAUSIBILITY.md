# Incremental Plausibility Update Strategy

## Problem

Current temporal clustering approach recomputes plausibility for ALL claims whenever a new claim arrives:
- Load all claims from graph
- Re-cluster by metric type
- Re-analyze each cluster with LLM
- Recompute all plausibility scores

**This is O(N²) and won't scale for streaming claims.**

## Core Issue

When claim #100 arrives, we shouldn't need to re-analyze claims #1-99. We need:
1. **Incremental updates** - Only update affected claims
2. **Cached metric extraction** - Store metrics on claim nodes
3. **Materialized plausibility** - Store scores in graph, not recompute
4. **Targeted re-scoring** - Only re-analyze clusters that changed

## Proposed Architecture

### 1. Store Metrics on Claim Nodes

```cypher
CREATE (c:Claim {
  id: 'cl_xxxxxxxx',
  text: '...',
  confidence: 0.8,
  event_time: datetime(),

  // STORED METRICS (extracted once at claim creation)
  metrics: {
    deaths: 156,
    injured: 12,
    missing: 30
  },

  // MATERIALIZED PLAUSIBILITY
  plausibility: 0.85,
  plausibility_updated_at: datetime(),
  plausibility_factors: {
    prior: 0.8,
    progression_boost: 0.05,
    consensus_boost: 0.10,
    contradiction_penalty: 0.0
  }
})
```

### 2. Incremental Update Workflow

**When new claim arrives:**

```
1. Extract metrics from new claim (LLM call #1)
   → deaths: 44

2. Find claims in same event with overlapping metrics
   MATCH (e:Event)-[:SUPPORTS]->(c:Claim)
   WHERE c.metrics.deaths IS NOT NULL
   → Found 3 claims: deaths=[36, 44, 156]

3. Check if cluster changed significantly
   - New value within existing range → Minor update
   - New value outside range → Major update
   - New metric type not seen before → New cluster

4. IF cluster changed significantly:
   → Re-analyze ONLY the deaths cluster (LLM call #2)
   → Update plausibility for affected claims (4 claims)

5. ELSE:
   → Use incremental Bayesian update (no LLM call)
   → new_claim.plausibility = 0.7 (moderate, middle of range)
```

**Complexity:**
- Best case: O(1) - no re-analysis needed
- Average case: O(k) where k = cluster size (typically 3-10 claims)
- Worst case: O(N) - event splits or major update

## 3. Incremental Bayesian Update Rules

### Rule 1: New claim corroborates existing consensus
```
Cluster: [36, 44, 44] (consensus around 44)
New: 44

Action: Boost all claims that said 44
  claim1 (44): 0.75 → 0.85
  claim2 (44): 0.80 → 0.90
  new_claim (44): 0.85

No LLM call needed - pure graph update
```

### Rule 2: New claim extends progression
```
Cluster: [36 @ t1, 44 @ t2]
New: 156 @ t3

Action: Recognize as progression
  claim1 (36): keep 0.70 (early report)
  claim2 (44): keep 0.75 (update)
  new_claim (156): 0.90 (latest)

LLM call: Check if progression or contradiction
```

### Rule 3: New claim contradicts consensus
```
Cluster: [44, 44, 44] (strong consensus)
New: 156

Action: Flag for review
  existing (44): keep 0.90 (majority)
  new_claim (156): 0.30 (outlier needs verification)

LLM call: Analyze contradiction
```

## 4. Graph Queries for Incremental Updates

### Find affected claims (same metric type)
```cypher
MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
WHERE c.metrics[$metric_type] IS NOT NULL
RETURN c.id, c.metrics[$metric_type] as value,
       c.event_time, c.plausibility
ORDER BY c.event_time
```

### Update plausibility scores
```cypher
MATCH (c:Claim {id: $claim_id})
SET c.plausibility = $new_score,
    c.plausibility_updated_at = datetime(),
    c.plausibility_factors = $factors
```

### Get plausibility-ranked claims for narrative
```cypher
MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
RETURN c
ORDER BY c.plausibility DESC, c.corroboration_count DESC
LIMIT 50
```

## 5. Trigger Conditions for Re-analysis

**Full cluster re-analysis needed when:**
1. New claim value is outlier (>2σ from mean)
2. Cluster size doubles (3 → 6 claims)
3. Time gap suggests new phase (6+ hours later)
4. First claim for new metric type
5. Manual flag for contradiction

**Incremental update sufficient when:**
1. New value within existing range
2. Extends known progression pattern
3. Corroborates existing consensus
4. Cluster already large (>10 claims)

## 6. Implementation Strategy

### Phase 1: Store metrics on claims
```python
async def _extract_and_store_metrics(self, claim: Claim) -> dict:
    """Extract metrics using LLM and store on claim node"""
    metrics = await self._extract_metrics_llm(claim.text)

    await self.neo4j._execute_write("""
        MATCH (c:Claim {id: $claim_id})
        SET c.metrics = $metrics
    """, {'claim_id': claim.id, 'metrics': metrics})

    return metrics
```

### Phase 2: Incremental update logic
```python
async def _update_plausibility_incremental(
    self,
    event: Event,
    new_claim: Claim,
    metrics: dict
):
    """Update plausibility scores incrementally"""

    for metric_type, value in metrics.items():
        # Get existing cluster
        cluster = await self._get_metric_cluster(event.id, metric_type)

        # Check if re-analysis needed
        if self._needs_reanalysis(cluster, value):
            # Full LLM analysis
            await self._reanalyze_cluster(event.id, metric_type)
        else:
            # Incremental Bayesian update
            await self._apply_incremental_update(new_claim, cluster, value)
```

### Phase 3: Cached results
```python
# Store cluster analysis results
CREATE (ca:ClusterAnalysis {
  event_id: 'ev_xxxxxxxx',
  metric_type: 'deaths',
  pattern: 'progression',
  analyzed_at: datetime(),
  claim_count: 7,
  value_range: [1, 156],
  consensus_value: 156,
  reasoning: '...'
})
```

## 7. Performance Comparison

### Current approach (batch re-analysis)
```
Claim 1:   1 LLM call  (extract metrics)
Claim 2:   3 LLM calls (extract + re-analyze 2 clusters)
Claim 3:   4 LLM calls (extract + re-analyze 3 clusters)
...
Claim 100: 102 LLM calls

Total: ~5,000 LLM calls for 100 claims
```

### Incremental approach
```
Claim 1:   1 LLM call  (extract, no cluster yet)
Claim 2:   2 LLM calls (extract + analyze new cluster)
Claim 3:   1 LLM call  (extract, incremental update)
Claim 4:   1 LLM call  (extract, incremental update)
...
Claim 20:  2 LLM calls (extract + re-analyze threshold)
...
Claim 100: 1 LLM call  (extract, incremental update)

Total: ~150 LLM calls for 100 claims (97% reduction)
```

## 8. Accuracy Considerations

**Incremental updates may miss:**
1. Cluster-wide pattern changes (progression → contradiction)
2. Multi-metric correlations (deaths + injured patterns)
3. Source credibility evolution

**Mitigation:**
1. Periodic full re-analysis (daily/weekly)
2. Triggered re-analysis on major events
3. Manual review flags for contested claims
4. Confidence decay over time (incentivizes refresh)

## 9. Next Steps

1. **Test metric storage** - Modify claim creation to store metrics
2. **Implement incremental logic** - Add `_needs_reanalysis()` heuristic
3. **Benchmark performance** - Compare batch vs incremental on 100 claims
4. **Validate accuracy** - Ensure incremental scores match batch scores
5. **Integration** - Wire into event_service metabolism

## 10. Open Questions

1. **How often to force full re-analysis?** (Every 10th claim? Daily? Never?)
2. **Should we store cluster analysis results?** (Caching pattern/reasoning)
3. **How to handle claim edits/retractions?** (Invalidate cached scores)
4. **Multi-event claims?** (Claim supports 2+ events, different plausibility per event?)
5. **Source credibility?** (NYT vs random blog affects plausibility?)
