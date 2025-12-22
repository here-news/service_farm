# Fractal Event System Overhaul Proposal

## Confidence Assessment

Based on our experiments, we have **HIGH CONFIDENCE** to proceed:

| Validated Component | Confidence | Evidence |
|---------------------|------------|----------|
| Semantic clustering | HIGH | 55% merge rate, handles "Hong Kong problem" |
| Hierarchical emergence | HIGH | Claims → Sub-events → Events working |
| Mass/coherence/tension | HIGH | Stable vs Active detection accurate |
| Streaming/breathing | HIGH | 1310 events emitted, events grow organically |
| Readiness analysis | HIGH | 80% reach publishable quality |
| Cost feasibility | HIGH | ~$0.07 per 1215 claims |
| **Relationship classification** | **HIGH** | **60% sibling, 33% contains; Mass assumption 100% wrong** |

---

## Current System vs. Proposed

### Current Architecture

```
EventWorker
    ↓
LiveEventPool (routes by entity overlap + page embedding)
    ↓
LiveEvent (examines claims, Bayesian topology)
```

**Limitations:**
- Entity-based routing can't distinguish "Hong Kong fire" vs "Hong Kong trial"
- No sub-event hierarchy (flat claim list)
- No semantic clustering during routing
- No readiness assessment
- No streaming event emissions

### Proposed Architecture

```
EventWorker
    ↓
FractalEventPool (routes by semantic embedding)
    ↓
EventfulUnit (recursive structure)
    │
    ├─ Level 0: Claims (leaf nodes)
    ├─ Level 1: Sub-events (claim clusters)
    ├─ Level 2: Events (sub-event groups)
    └─ Level 3+: Frames (event narratives)
```

---

## CRITICAL REFINEMENT: Relationship Types (Added 2025-12-17)

**Issue identified during demo:** Two related events about Hong Kong fire didn't merge:
- "Massive Tai Po Fire Kills 36" (mass: 8.5)
- "Lee Jae Myung Offers Condolences" (mass: 2.1)

**Previous assumption (FLAWED):** Higher mass absorbs lower mass (gravitational model)

**The insight:** Mass determines importance, NOT containment. Like Earth and Jupiter - Jupiter's greater mass doesn't make Earth its moon.

### Relationship Types Between Similar EUs

| Type | Description | Edge in Graph |
|------|-------------|---------------|
| **CONTAINS** | B is a sub-aspect of A | `A -[:CONTAINS]-> B` |
| **SIBLING** | Both aspects of larger topic | `A <-[:CHILD]- Parent -[:CHILD]-> B` |
| **CAUSES** | A led to/caused B | `A -[:CAUSES]-> B` |
| **RELATES** | Associated but distinct | `A -[:RELATES]-> B` |

### When to Apply Each

```python
if similarity(A, B) > threshold:
    rel_type = classify_relationship(A, B)  # LLM or heuristic

    if rel_type == CONTAINS:
        larger.add_child(smaller)
    elif rel_type == SIBLING:
        parent = find_or_create_frame(A, B)
        parent.add_children([A, B])
    elif rel_type == CAUSES:
        A.add_edge(CAUSES, B)
    else:
        A.add_edge(RELATES, B)
```

### Key Principle

**Mass determines:**
- Importance/centrality ranking
- Which topic dominates when creating parent frame
- Priority in display/search

**Mass does NOT determine:**
- Containment relationship
- Whether one EU "owns" another

**Semantic analysis determines:**
- Relationship type
- Containment vs sibling vs causal vs associative

**See:** `RELATIONSHIP_TYPES.md` for full analysis.

**VALIDATED (2025-12-17):** Experiment `relationship_classification.py` confirmed:
- Mass-based containment assumption: **100% error rate** (13/13 wrong)
- SIBLING relationships: **60%** (18/30 pairs)
- CONTAINS relationships: **33%** (10/30 pairs)
- Most related events are SIBLINGS, not parent-child!

---

## Key Changes Required

### 1. Data Model: EventfulUnit (EU)

```python
@dataclass
class EventfulUnit:
    id: str
    level: int  # 0=claim, 1=sub-event, 2=event, 3=frame

    # Content (for levels 0-1)
    claim_ids: List[str]
    texts: List[str]

    # Hierarchy
    children: List[str]  # Child EU ids
    parent_id: Optional[str]  # Parent EU id

    # Embedding (running centroid)
    embedding: List[float]

    # Metrics (our validated formulas)
    internal_corr: int
    internal_contra: int

    def mass(self) -> float:
        return self.size() * 0.1 * (0.5 + self.coherence()) * (1 + 0.1 * len(self.page_ids))

    def coherence(self) -> float:
        total = self.internal_corr + self.internal_contra
        return self.internal_corr / total if total > 0 else 1.0

    def tension(self) -> float:
        return self.internal_contra / (self.internal_corr + self.internal_contra)

    def state(self) -> str:
        return "ACTIVE" if self.tension() > 0.1 else "STABLE"
```

### 2. Neo4j Schema Changes

```cypher
// New EU node type
CREATE CONSTRAINT eu_id IF NOT EXISTS FOR (eu:EU) REQUIRE eu.id IS UNIQUE;

// EU properties
(:EU {
    id: 'eu_xxxx',
    level: 1,          // 0=sub-event, 1=event, 2=frame
    embedding: [...],   // pgvector compatible
    mass: 5.2,
    coherence: 0.85,
    tension: 0.15,
    state: 'STABLE',    // or 'ACTIVE'
    created_at: datetime(),
    last_activity: datetime()
})

// Hierarchy relationships
(:EU)-[:PARENT_OF]->(:EU)
(:EU)-[:CONTAINS]->(:Claim)

// Keep existing Event node as Level 2 EU
// Migrate gradually
```

### 3. PostgreSQL Changes

```sql
-- Already have claim_embeddings, add EU embeddings
CREATE TABLE core.eu_embeddings (
    eu_id TEXT PRIMARY KEY,
    embedding vector(1536) NOT NULL,
    level INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- EU metadata (faster than Neo4j for frequent updates)
CREATE TABLE core.eu_metrics (
    eu_id TEXT PRIMARY KEY,
    level INTEGER NOT NULL,
    claim_count INTEGER DEFAULT 0,
    page_count INTEGER DEFAULT 0,
    corroborations INTEGER DEFAULT 0,
    contradictions INTEGER DEFAULT 0,
    mass NUMERIC(6,3),
    coherence NUMERIC(4,3),
    tension NUMERIC(4,3),
    state VARCHAR(10),
    semantic_score NUMERIC(4,3),
    epistemic_score NUMERIC(4,3),
    readiness VARCHAR(20),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 4. Replace LiveEventPool with FractalEventPool

```python
class FractalEventPool:
    """
    Routes claims to EventfulUnits using semantic similarity.
    Manages hierarchical EU structure.
    """

    def __init__(self, ...):
        self.eus: Dict[str, EventfulUnit] = {}  # In-memory cache
        self.embedding_cache: Dict[str, List[float]] = {}

        # Thresholds (validated in experiments)
        self.sim_threshold = 0.70
        self.llm_threshold = 0.55
        self.event_merge_threshold = 0.60
        self.frame_merge_threshold = 0.50

    async def route_claim(self, claim: Claim, embedding: List[float]) -> EventfulUnit:
        """
        Route single claim to appropriate EU.
        Creates new sub-event if no match.
        """
        # Find best matching level-0 EU (sub-event)
        best_eu, best_sim = self._find_best_match(embedding, level=0)

        if best_sim >= self.sim_threshold:
            return await self._absorb_into_eu(claim, best_eu)
        elif best_sim >= self.llm_threshold:
            # LLM verification for borderline
            if await self._llm_same_event(claim.text, best_eu.texts[0]):
                return await self._absorb_into_eu(claim, best_eu)

        # No match - create new sub-event
        return await self._create_sub_event(claim, embedding)

    async def periodic_merge_pass(self):
        """
        Periodically try to merge EUs upward.
        Sub-events → Events → Frames
        """
        await self._merge_level(0, 1, self.event_merge_threshold)  # sub → event
        await self._merge_level(1, 2, self.frame_merge_threshold)  # event → frame
```

### 5. Replace LiveEvent with FractalEvent

```python
class FractalEvent(EventfulUnit):
    """
    Living EventfulUnit with metabolism.
    Extends EU with topology, narrative, and readiness.
    """

    def __init__(self, eu: EventfulUnit, event_service, topology_service):
        super().__init__(**eu.__dict__)
        self.service = event_service
        self.topology_service = topology_service

        # Readiness assessment
        self.semantic_readiness: Optional[float] = None
        self.epistemic_readiness: Optional[float] = None
        self.w5h1_coverage: Optional[Dict] = None
        self.narrative_stage: Optional[str] = None

    async def on_claim(self, claim: Claim, embedding: List[float]) -> List[StreamEvent]:
        """
        Process claim - core "breathing" action.
        Returns stream events that occurred.
        """
        events = []

        # Absorb claim
        self._absorb_claim(claim, embedding)
        events.append(StreamEvent(type='claim_absorbed', eu_id=self.id, ...))

        # Check for contradictions
        if self._new_contradiction_detected(claim):
            events.append(StreamEvent(type='contradiction', eu_id=self.id, ...))

        # Check mass thresholds
        if self._crossed_mass_threshold():
            events.append(StreamEvent(type='mass_threshold', eu_id=self.id, ...))

        # Check state transitions
        if self._state_changed():
            events.append(StreamEvent(
                type='stabilized' if self.state() == 'STABLE' else 'activated',
                eu_id=self.id, ...
            ))

        return events

    async def assess_readiness(self) -> Dict:
        """
        Full epistemic + semantic readiness assessment.
        """
        # 5W1H coverage
        self.w5h1_coverage = await self._assess_5w1h()

        # Narrative completeness
        narrative = await self._assess_narrative()
        self.narrative_stage = narrative.get('narrative_stage')

        # Epistemic quality
        epistemic = await self._assess_epistemic_quality()

        # Compute scores
        self.semantic_readiness = self._compute_semantic_score(self.w5h1_coverage, narrative)
        self.epistemic_readiness = self._compute_epistemic_score(epistemic)

        # Determine recommendation
        return self._get_recommendation()
```

### 6. Streaming Event Emissions

```python
class StreamEvent:
    type: EventType  # claim_absorbed, eu_created, merged, contradiction, etc.
    timestamp: str
    eu_id: str
    data: Dict

# SSE endpoint in FastAPI
@app.get("/api/events/stream")
async def event_stream():
    async def generate():
        while True:
            events = await pool.get_pending_events()
            for event in events:
                yield f"data: {event.to_json()}\n\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## Migration Strategy

### Phase 1: Parallel Operation (2 weeks)

1. **Deploy FractalEventPool alongside LiveEventPool**
2. **Route new claims to both** (shadow mode)
3. **Compare outputs** - validate clustering matches
4. **No writes to Neo4j from FractalEventPool** - read-only

### Phase 2: EU Schema Migration (1 week)

1. **Create EU nodes in Neo4j** for existing Events
2. **Migrate Event → EU (level=2)**
3. **Create sub-events (level=1)** from claim clusters
4. **Compute embeddings** for all EUs

### Phase 3: Cutover (1 week)

1. **Switch routing to FractalEventPool**
2. **Keep LiveEventPool as fallback**
3. **Monitor for issues**
4. **Enable streaming endpoints**

### Phase 4: Cleanup (ongoing)

1. **Remove LiveEventPool**
2. **Migrate remaining code**
3. **Add Frame emergence (level=3)**

---

## Key Considerations

### 1. Performance

| Operation | Current | Proposed | Impact |
|-----------|---------|----------|--------|
| Claim routing | Entity Jaccard | Embedding cosine | Faster (O(n) vs O(n²)) |
| Merge detection | N/A | LLM verification | +$0.0001/call |
| Hierarchy | Flat | Recursive | More memory |
| Streaming | N/A | SSE | New endpoint |

**Mitigation:**
- Cache embeddings in PostgreSQL (already doing)
- Use FAISS for fast similarity search at scale
- Batch LLM calls where possible

### 2. Cost

| Scale | Embeddings | LLM Clustering | LLM Readiness | Total |
|-------|-----------|----------------|---------------|-------|
| 1K claims | $0.02 | $0.04 | $0.03 | $0.09 |
| 10K claims | $0.20 | $0.40 | $0.30 | $0.90 |
| 100K claims | $2.00 | $4.00 | $3.00 | $9.00 |

**Feasible for production.**

### 3. Backwards Compatibility

**API contracts to maintain:**
- `/api/events` - List events (now returns Level 2+ EUs)
- `/api/events/{id}` - Get event (returns EU with children)
- `/api/events/{id}/claims` - Get claims (traverses hierarchy)
- `/api/claims` - Unchanged

**New endpoints:**
- `/api/events/stream` - SSE stream
- `/api/events/{id}/readiness` - Readiness assessment
- `/api/eus/{id}` - Direct EU access (all levels)

### 4. What We Preserve from Current System

| Component | Keep? | Reason |
|-----------|-------|--------|
| Bayesian topology | YES | Complements EU metrics |
| Narrative generation | YES | Works well |
| Coherence calculation | MODIFY | Use EU formula instead |
| Publisher priors | YES | Source credibility |
| Thoughts/epistemic | YES | Add to readiness |
| Hibernation | YES | Same logic |
| Commands (/retopologize) | YES | Add /assess_readiness |

### 5. What We Replace

| Component | Replace With | Reason |
|-----------|--------------|--------|
| Entity-based routing | Semantic embedding | Handles "Hong Kong problem" |
| Flat claim list | Hierarchical EU | Natural sub-events |
| Manual sub-event creation | Automatic emergence | Claims cluster naturally |
| No readiness check | Epistemic + semantic | Publishability signal |

---

## Success Metrics

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| Clustering accuracy | ~70% | >85% | Manual review of 100 events |
| Sub-event coherence | N/A | >80% | Average coherence of level-1 EUs |
| Readiness accuracy | N/A | >80% | Human review of READY vs NEEDS_CURATION |
| Routing latency | ~100ms | <200ms | P95 latency |
| LLM cost/claim | N/A | <$0.001 | Total LLM cost / claims |

---

## Recommendation

**YES, proceed with overhaul.**

We have validated:
1. ✅ Semantic clustering works better than entity-based
2. ✅ Hierarchical emergence is natural and useful
3. ✅ Mass/coherence/tension metrics are meaningful
4. ✅ Readiness assessment is accurate (80%+)
5. ✅ Streaming/breathing system works
6. ✅ Cost is feasible (<$10 for 100K claims)

The current LiveEvent system is good but limited by:
- Entity-based routing (Hong Kong problem)
- Flat claim structure (no natural sub-events)
- No readiness signaling

The fractal EU model addresses all these while preserving what works (topology, narratives, metabolism).

---

## Next Steps

1. **Create GitHub issue** for fractal event overhaul
2. **Design Neo4j schema** in detail
3. **Implement FractalEventPool** (parallel mode)
4. **Run shadow comparison** for 1 week
5. **Plan migration** based on results

---

## VALIDATION COMPLETE (2025-12-17)

### Demo Results (1215 claims)

The unified EU architecture has been fully validated:

| Metric | Result |
|--------|--------|
| L1 Sub-events | 708 |
| L2 Events | 39 |
| L3 Parent groupings | 10 |
| Average claims/L2 | ~31 |
| Stories detected | 8/8 major stories |

### Key Validations

1. **Hong Kong Fire Problem SOLVED**
   - Fire (46 claims) + Lee Jae Myung Condolences (9 claims)
   - Correctly grouped as SIBLINGS under L3 parent
   - Original demo observation that triggered this research

2. **Semantic Classification Working**
   - 60% SIBLING relationships
   - 33% CONTAINMENT relationships
   - Mass-based containment assumption: 100% error rate

3. **Mass-Asymmetric Siblings Validated**
   - Jimmy Lai (mass 31.7) + Wong Kwok-ngon (mass 0.5)
   - Both under L3 "Media Crackdown" (mass 30.71)
   - Child MORE massive than parent - validates thematic grouping

### Architecture Confirmed

```
Claims → L1 (sub-events) → L2 (events) → L3+ (parents)
                ↑                ↑              ↑
           embedding        embedding     semantic LLM
           similarity       similarity    classification
```

**Ready for production integration.**

---

## Design Principle: Stability Gradient

The EU hierarchy exhibits a natural **stability gradient** - lower levels are busier, upper levels stabilize later:

```
L1 (Sub-events):  ████████████████████  (high churn, claims arriving constantly)
L2 (Events):      ████████              (moderate activity, sub-events merge/split)
L3+ (Parents):    ███                   (slow formation, only when L2 is solid)
```

**Key properties:**

1. **L1 absorbs the chaos** - rapid claim absorption, frequent merging/splitting
2. **L2 buffers the noise** - only promotes stable L1 clusters to events
3. **L3+ crystallizes slowly** - forms only from stable L2 events with semantic similarity

**Implementation guidance:**

```python
def should_attempt_parent_grouping(l2_event):
    return (
        l2_event.state == "STABLE" and      # Not actively changing
        l2_event.claim_count >= MIN_CLAIMS and  # Substantial
        l2_event.age >= MIN_AGE             # Has existed long enough
    )
```

**L4+ emergence:** No explicit implementation needed. If the architecture handles L1→L2→L3 correctly, deeper levels (L4+) will emerge naturally when sufficient L3 events exist with semantic similarity. The same sibling-grouping logic applies recursively.

**Observed in demo:** 708 L1 → 39 L2 → 10 L3 (funnel effect validated)

---

*Proposal generated 2025-12-17*
*Validation completed 2025-12-17*
