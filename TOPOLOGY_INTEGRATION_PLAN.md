# Claim Topology Integration Plan

## Current State Analysis

### Current Architecture
```
LiveEventPool
  └─> LiveEvent.examine(claims)
        └─> EventService.examine_claims()
              ├─> _classify_claim() - multi-signal scoring
              ├─> _merge_duplicate() - corroboration only
              └─> _generate_event_narrative() - corroboration-guided
```

### Current Metabolism (Problems)
1. **Claim scoring**: Multi-signal (entity, temporal, semantic) but **no plausibility resolution**
2. **Corroboration**: Creates `CORROBORATES` relationships, boosts confidence
3. **Narrative**: Groups by topic keywords, but **doesn't weight by plausibility**
4. **No contradiction detection**: Doesn't identify conflicting claims
5. **No temporal progression**: Doesn't recognize that "36→44→156 deaths" is progression, not contradiction

### Test Results (Topology Approach)
- **87 claims** analyzed with semantic network topology
- **Plausibility range**: 0.30-0.85 (good differentiation)
- **Outliers detected**: "156 deaths" at 0.30 (minority view correctly penalized)
- **Progressive pattern**: Recognized temporal updates
- **Network density**: 1,296 edges (rich connectivity)
- **LLM efficiency**: 5 calls for 87 claims

---

## Integration Design

### New Components

#### 1. ClaimTopologyService (New)
```python
class ClaimTopologyService:
    """
    Manages claim network topology for an event.
    Computes plausibility scores via semantic network analysis.
    """

    def __init__(self, openai_client: AsyncOpenAI):
        self.openai = openai_client

        # In-memory topology state (per event)
        self.embeddings: Dict[str, List[float]] = {}
        self.network: Dict[str, List[Tuple[str, float]]] = {}
        self.plausibility: Dict[str, float] = {}
        self.analysis: Optional[dict] = None  # LLM analysis result

    async def add_claims(
        self,
        event_id: str,
        new_claims: List[Claim],
        existing_claims: List[Claim]
    ) -> TopologyUpdate:
        """
        Add new claims to topology and recompute plausibility.

        Returns:
            TopologyUpdate with:
            - plausibility_scores: {claim_id: score}
            - score_changes: [(claim_id, old, new)]
            - pattern: 'consensus' | 'progressive' | 'contradictory' | 'mixed'
            - contradictions: ['36 vs 156 deaths']
            - consensus_points: ['Fire at Wang Fuk Court']
        """

    async def get_weighted_claims(
        self,
        event_id: str,
        threshold: float = 0.50
    ) -> List[WeightedClaim]:
        """Get claims with plausibility >= threshold for narrative."""

    def should_reanalyze(
        self,
        new_claim_count: int,
        time_since_last: float
    ) -> bool:
        """
        Determine if topology needs full re-analysis.

        Triggers reanalysis when:
        - new_claim_count >= 5
        - time_since_last > 1 hour AND new_claim_count > 0
        - First claims for event
        """
```

#### 2. TopologyUpdate (Dataclass)
```python
@dataclass
class TopologyUpdate:
    plausibility_scores: Dict[str, float]
    score_changes: List[Tuple[str, float, float]]  # (claim_id, old, new)
    pattern: str  # 'consensus' | 'progressive' | 'contradictory' | 'mixed'
    contradictions: List[str]
    consensus_points: List[str]
    network_stats: dict  # edges, density, clusters
```

#### 3. WeightedClaim (Dataclass)
```python
@dataclass
class WeightedClaim:
    claim: Claim
    plausibility: float
    agreements: List[str]  # claim_ids that agree
    contradictions: List[str]  # claim_ids that contradict
    network_degree: int
```

---

### Integration Points

#### 1. LiveEvent.examine() - Add Topology Update
```python
async def examine(self, new_claims: List[Claim]):
    """Layer 3: Event's metabolism with topology analysis."""

    # Existing: Capture coherence before
    old_coherence = self.event.coherence

    # Existing: Process claims via EventService
    result = await self.service.examine_claims(self.event, new_claims)

    # NEW: Update topology with accepted claims
    if result.claims_added:
        self.claims.extend(result.claims_added)

        # NEW: Topology analysis
        topology_update = await self.topology_service.add_claims(
            event_id=self.event.id,
            new_claims=result.claims_added,
            existing_claims=self.claims
        )

        # Store plausibility scores on claims (in graph)
        for claim_id, score in topology_update.plausibility_scores.items():
            await self._store_plausibility(claim_id, score)

        # Check if narrative needs regeneration
        if self._should_regenerate_narrative(topology_update):
            await self.regenerate_narrative_weighted(topology_update)

    return result
```

#### 2. LiveEvent.regenerate_narrative_weighted() - New Method
```python
async def regenerate_narrative_weighted(self, topology: TopologyUpdate):
    """Generate narrative using topology weights."""

    # Get claims sorted by plausibility
    weighted_claims = await self.topology_service.get_weighted_claims(
        self.event.id,
        threshold=0.50
    )

    # Generate with explicit weights
    narrative = await self.service.generate_weighted_narrative(
        event=self.event,
        weighted_claims=weighted_claims,
        topology=topology
    )

    # Update
    await self.service.event_repo.update_narrative(self.event.id, narrative)
    self.event.summary = narrative
    self.last_narrative_update = datetime.utcnow()
```

#### 3. EventService.generate_weighted_narrative() - New Method
```python
async def generate_weighted_narrative(
    self,
    event: Event,
    weighted_claims: List[WeightedClaim],
    topology: TopologyUpdate
) -> str:
    """Generate narrative with explicit plausibility weighting."""

    # Separate claims by plausibility tier
    high = [c for c in weighted_claims if c.plausibility >= 0.75]
    medium = [c for c in weighted_claims if 0.55 <= c.plausibility < 0.75]
    low = [c for c in weighted_claims if c.plausibility < 0.55]

    # Build prompt with explicit scoring
    prompt = f"""Generate factual narrative for: {event.canonical_name}

PATTERN: {topology.pattern}

CONSENSUS (established facts):
{self._format_list(topology.consensus_points)}

CONTRADICTIONS (resolved by plausibility):
{self._format_contradictions(topology.contradictions, weighted_claims)}

HIGH CONFIDENCE CLAIMS (0.75-1.0) - USE THESE AS FACTS:
{self._format_weighted_claims(high[:15])}

MEDIUM CONFIDENCE (0.55-0.74) - Supporting details:
{self._format_weighted_claims(medium[:10])}

LOW CONFIDENCE (<0.55) - Mention as 'some reports claim' or omit:
{self._format_weighted_claims(low[:5])}

RULES:
1. State HIGH confidence claims as FACTS (e.g., "36 people died" not "between 36-156")
2. For resolved contradictions: Use the HIGH plausibility value
3. Only show uncertainty when HIGH claims themselves are uncertain
4. LOW claims: Either omit or say "unverified reports mention..."
5. Structure: Incident → Casualties → Response → Timeline → Impact
"""

    response = await self.openai_client.chat.completions.create(...)
    return response.choices[0].message.content
```

---

### Graph Schema Changes

#### Store Plausibility on SUPPORTS Relationship
```cypher
-- Current
(e:Event)-[:SUPPORTS]->(c:Claim)

-- New: Add plausibility score to relationship
(e:Event)-[:SUPPORTS {plausibility: 0.85, last_updated: datetime()}]->(c:Claim)

-- Query claims by plausibility
MATCH (e:Event {id: $event_id})-[r:SUPPORTS]->(c:Claim)
WHERE r.plausibility >= 0.75
RETURN c, r.plausibility
ORDER BY r.plausibility DESC
```

#### Store Topology Metadata on Event
```cypher
-- Add topology state to Event node
(:Event {
    id: 'ev_xxx',
    topology_pattern: 'progressive',
    topology_consensus: ['Fire at Wang Fuk Court', 'Fire alarms not working'],
    topology_contradictions: ['Death toll: 36 (0.80) vs 156 (0.30)'],
    topology_last_updated: datetime(),
    topology_claim_count: 87,
    topology_edge_count: 1296
})
```

---

### Incremental Update Strategy

#### When to Re-analyze Topology

```python
def should_reanalyze(self, new_claim_count: int, time_since_last: float) -> bool:
    """
    Balance accuracy vs LLM cost.

    Full re-analysis when:
    - 5+ new claims (significant network change)
    - >1 hour since last AND any new claims (staleness)
    - First analysis (cold start)

    Quick update otherwise:
    - Just add edges to network
    - Inherit scores from similar existing claims
    - Flag for re-analysis on next metabolism cycle
    """
    if self.analysis is None:
        return True  # Cold start

    if new_claim_count >= 5:
        return True  # Significant change

    if time_since_last > 3600 and new_claim_count > 0:
        return True  # Stale + new data

    return False
```

#### Quick Incremental Update (No LLM)
```python
async def quick_update(self, new_claims: List[Claim]) -> Dict[str, float]:
    """
    Fast incremental update without LLM.

    1. Generate embeddings for new claims
    2. Add edges to network
    3. Estimate plausibility from network position:
       - High connectivity to high-plausibility claims → high score
       - High connectivity to low-plausibility claims → lower score
       - Isolated claims → neutral (0.55)
    """
    new_scores = {}

    for claim in new_claims:
        # Generate embedding
        embedding = await self._generate_embedding(claim.text)
        self.embeddings[claim.id] = embedding

        # Find neighbors and their scores
        neighbors = self._find_neighbors(claim.id, embedding)

        if not neighbors:
            new_scores[claim.id] = 0.55  # Isolated → neutral
        else:
            # Weighted average of neighbor scores
            total_weight = 0
            score_sum = 0
            for neighbor_id, similarity in neighbors:
                neighbor_score = self.plausibility.get(neighbor_id, 0.55)
                score_sum += similarity * neighbor_score
                total_weight += similarity

            new_scores[claim.id] = score_sum / total_weight

        # Add to network
        for neighbor_id, similarity in neighbors:
            self.network[claim.id].append((neighbor_id, similarity))
            self.network[neighbor_id].append((claim.id, similarity))

    return new_scores
```

---

### Implementation Phases

#### Phase 1: ClaimTopologyService (Foundation)
1. Create `services/claim_topology.py`
2. Implement `add_claims()` with full LLM analysis
3. Implement `get_weighted_claims()`
4. Add unit tests with mock claims

#### Phase 2: Graph Schema
1. Add `plausibility` property to SUPPORTS relationship
2. Add topology metadata to Event node
3. Update EventRepository with new queries
4. Migration script for existing data (set plausibility=0.50 as default)

#### Phase 3: LiveEvent Integration
1. Add `topology_service` to LiveEvent
2. Modify `examine()` to call topology analysis
3. Add `regenerate_narrative_weighted()`
4. Update `needs_narrative_update()` to check topology staleness

#### Phase 4: Weighted Narrative
1. Create `EventService.generate_weighted_narrative()`
2. Better prompt engineering for decisive narratives
3. Test with existing events

#### Phase 5: Incremental Optimization
1. Implement `quick_update()` for low-latency updates
2. Add `should_reanalyze()` logic
3. Background re-analysis in metabolism cycle
4. Performance benchmarking

---

### Expected Outcomes

1. **Plausibility differentiation**: Claims get 0.30-0.90 scores (not all 0.50)
2. **Outlier detection**: Minority claims flagged automatically
3. **Progressive recognition**: "36→44→156" recognized as updates, not contradictions
4. **Decisive narratives**: "36 people died" (not "36-156 reported")
5. **Efficient updates**: 5 LLM calls for 87 claims (~17 claims/call)
6. **Incremental scaling**: Quick updates for streaming claims

---

### Testing Strategy

1. **Unit tests**: ClaimTopologyService with mock claims
2. **Integration tests**: LiveEvent → TopologyService → Graph
3. **Regression tests**: Existing events should maintain quality
4. **A/B test**: Compare narratives with/without topology weighting
5. **Load test**: 100+ claims incremental ingestion
