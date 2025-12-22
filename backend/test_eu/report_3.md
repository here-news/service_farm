# EU Experiment Report 3

## What We Tested

Semantic emergence using embeddings + LLM verification, comparing against entity-based approaches.

## The Core Problem

Previous approaches had limitations:

| Approach | Claims Assigned | Events Found | Problem |
|----------|-----------------|--------------|---------|
| Explicit links only | 12% | - | Most claims have no corr/contra links |
| Single entity gravity | 76% | 7/16 | Too broad ("Hong Kong" mixes stories) |
| Entity pairs | 18% | 2/16 | Too narrow (pairs too specific) |
| Anchor + Context | 73% | 7/16 | Some anchors are both specific AND broad |

**Root issue:** Entity-based clustering can't distinguish between:
- "Hong Kong fire" (Wang Fuk Court)
- "Hong Kong trial" (Jimmy Lai)
- "Hong Kong politics" (general)

They all mention "Hong Kong" but are different events.

## Semantic Approach

### Method

1. **Embeddings** (text-embedding-3-small): Convert claims to vectors
2. **Similarity clustering**: Group claims with cosine similarity > threshold
3. **LLM verification** (gpt-4o-mini): For borderline cases, ask "same event?"
4. **Hierarchical merge**: Cluster sub-events into events using LLM "same story?"

### Results on 70-claim sample (Wang Fuk Court + Jimmy Lai + Charlie Kirk)

```
Level 0: 70 claims
Level 1: 45 sub-events (embedding clustering at 0.75)
Level 2: 38 events (LLM-verified merging)

Significant events emerged:
- Wang Fuk Court Fire: 18 claims, 5 sub-events
- Amanda Seyfried/Kirk: 9 claims, 2 sub-events
- Jimmy Lai detention: 7 claims, 3 sub-events
```

### Sub-event structure (Wang Fuk Court)

The system naturally found meaningful sub-narratives:

| Sub-event | Claims | Summary |
|-----------|--------|---------|
| Fire outbreak | 8 | Fire at Wang Fuk Court, Tai Po |
| Death toll | 4 | At least 156 fatalities |
| Vigil | 2 | Memorial held Nov 28 |
| Fire alarms | 2 | Alarms confirmed ineffective |
| Building spread | 2 | Fire engulfed 7 buildings |

This is **exactly what we want** - semantic sub-events that compose into a larger event.

## Threshold Analysis

| Config | Clusters | Largest | Notes |
|--------|----------|---------|-------|
| Embedding only (0.80) | 37 | 5 | Too strict |
| Embedding only (0.75) | 27 | 8 | Good balance |
| Embedding + LLM (0.80/0.65) | 30 | 10 | Best - LLM catches edge cases |

LLM verification:
- Caught true positives at sim=0.69 (would be missed by threshold alone)
- Rejected false positives at sim=0.71-0.77 (unrelated claims)

## Integrating with Earlier Metrics

### Mass

From Report 1, mass = f(corroboration, entity_weight, source_diversity).

In semantic hierarchy:
```
Mass(EU) = Σ Mass(children) × coherence_bonus

Where coherence_bonus = 1 + (internal_corr / possible_pairs)
```

For Wang Fuk Court (18 claims):
- Base mass: 18 × 0.1 = 1.8
- Internal corroboration: high (death toll updates corroborate each other)
- Source diversity: 10+ pages
- **Estimated mass: ~3.5** (high - browsable event)

### Metabolism & Coherence

From Report 2, clusters can be **stable** or **active**:

| Sub-event | Coherence | Tension | State |
|-----------|-----------|---------|-------|
| Fire outbreak | 100% | 0% | Stable |
| Death toll | 75% | 25% | Active (conflicting numbers) |
| Vigil | 100% | 0% | Stable |
| Fire alarms | 100% | 0% | Stable |

The "Death toll" sub-event has tension because:
- "At least 36 killed" (early report)
- "At least 156 killed" (later report)
- "160 confirmed" (final)

This is **healthy metabolism** - the sub-event is digesting temporal updates.

### Entropy

From docs, entropy formula: Hₙ = α·Cd - β·κ - γ·Ds - δ·Sd

For semantic clusters:
- **Cd (claim diversity)**: Higher diversity → higher entropy
- **κ (coherence)**: More corroboration → lower entropy
- **Ds (source diversity)**: More sources → lower entropy (confirmation)
- **Sd (semantic density)**: Tighter embedding cluster → lower entropy

Semantic clustering naturally optimizes for low Sd (semantic density).

## Cost Analysis

For 70 claims:
- Embedding calls: 1 batch call (70 texts)
- LLM calls: ~20-25 (summaries + verification)
- Total API cost: ~$0.01-0.02

Projected for full dataset (1215 claims):
- Embedding: 1-2 batch calls
- LLM: ~300-400 calls (summaries + verification)
- Estimated cost: ~$0.30-0.50

**Conclusion:** Feasible for current scale. May need optimization for 10K+ claims.

## Progressive Emergence Model

Combining all learnings, the ideal model:

### Phase 1: Claim Arrival

```
Page arrives → emits claims
Each claim gets:
- Embedding (semantic position)
- Entity links (graph position)
- Initial mass = 0.1
```

### Phase 2: Sub-event Formation

```
For each new claim:
1. Find semantically similar existing EUs (embedding)
2. If similarity > 0.75: merge directly
3. If 0.65 < similarity < 0.75: LLM verify
4. Else: create new leaf EU

Update merged EU:
- Recalculate embedding (centroid)
- Update mass = Σ claim_mass × coherence
- Track internal corr/contra (metabolism state)
```

### Phase 3: Event Emergence

```
Periodically (or on threshold):
1. Find sub-events with shared entities OR high embedding similarity
2. LLM verify: "same broader story?"
3. If yes: create parent Event EU
4. Event inherits mass from children
```

### Phase 4: Metabolism

```
For active EUs (has contradictions):
- Track coherence over time
- As new claims resolve contradictions → coherence increases
- When coherence stabilizes → mark as "settled"

For stale EUs (no new claims):
- Apply decay to mass
- Eventually archive if mass < threshold
```

## Key Insights

### 1. Semantic > Entity for clustering

Entities create false connections ("Hong Kong" links unrelated events).
Embeddings capture actual semantic similarity.

### 2. LLM verification is cheap and effective

For borderline cases (sim 0.65-0.80), LLM catches both:
- False negatives (similar events missed by threshold)
- False positives (different events above threshold)

### 3. Hierarchy emerges naturally

Claims → Sub-events → Events is not forced structure.
It emerges from semantic similarity at different granularities.

### 4. Sub-events are real

"Death toll updates" is a coherent sub-narrative within "Wang Fuk Court Fire".
This matches how humans understand news.

### 5. Metabolism applies at all levels

- Claim level: new info arrives
- Sub-event level: conflicting reports resolved
- Event level: narrative consolidates

## Comparison with Original Events

| Original Event | Claims | Semantic Match | Overlap |
|----------------|--------|----------------|---------|
| Wang Fuk Court Fire | 130 | Wang Fuk Court Fire | ~60%* |
| Jimmy Lai Imprisonment | 117 | Jimmy Lai detention | ~50%* |
| Charlie Kirk Assassination | 35 | Amanda/Kirk controversy | ~70%* |

*Estimated from 70-claim sample. Full dataset would improve.

The semantic approach finds **tighter** clusters than original events, which may be more accurate (original events may be over-inclusive).

## Open Questions

1. **Embedding model choice**: Would a news-specific model perform better?

2. **LLM prompt tuning**: Current prompts are simple. Could improve with few-shot examples.

3. **Real-time vs batch**: Current approach is batch. How to handle streaming claims?

4. **Cross-event links**: Some claims relate to multiple events. How to handle?

5. **Temporal awareness**: Embeddings don't capture time. Should we weight recent claims higher?

## Recommendations

1. **Use semantic + entity hybrid**
   - Semantic for primary clustering
   - Entity for validation/enrichment
   - Explicit links for corroboration tracking

2. **Implement hierarchical EU structure**
   - Claim → Sub-event → Event → Narrative (optional)
   - Each level has mass, coherence, metabolism state

3. **LLM verification for borderline merges**
   - Cost-effective (~$0.001 per call)
   - Catches edge cases embedding misses

4. **Track metabolism state**
   - Stable vs Active distinction useful for UI
   - "This story is evolving" indicator

5. **Cache embeddings**
   - Store in Neo4j or separate vector DB
   - Avoid recomputing on each run

## Files Created

```
test_eu/
├── semantic_emergence.py      # Initial embedding experiment
├── semantic_emergence_v2.py   # With LLM verification
├── semantic_hierarchy.py      # Full hierarchical emergence
├── page_dissolution.py        # Page as initial EU concept
├── entity_gravity.py          # Single entity clustering
├── entity_constellation.py    # Entity pair clustering
├── anchor_context.py          # Anchor + context hybrid
└── results/
    ├── semantic_emergence.json
    ├── semantic_hierarchy.json
    └── ...
```

## Conclusion

**Semantic emergence works.**

The combination of embeddings + LLM verification produces coherent, hierarchical event structures that:
- Match human intuition about news events
- Find meaningful sub-narratives
- Handle the "Hong Kong problem" (disambiguation)
- Scale reasonably in cost

The next step is integrating this into the production system with:
- Cached embeddings
- Incremental updates (not batch)
- UI for stable vs active events
- Mass-based browsing threshold

---

*Report generated 2025-12-17*
