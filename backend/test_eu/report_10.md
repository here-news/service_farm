# EU Experiment Report 10

## Breathing Event System - Live Streaming Prototype

### What We Built

A living event system that:
1. **Claims stream in** continuously
2. **Claims find position** (cluster into EUs)
3. **EUs "breathe"** - grow, absorb, increase mass
4. **LLM curates** internally (causal relationships, coherence)
5. **System emits decisions** (stabilized, contradiction, ready to publish)

### Results

```
Total claims processed: 1215
Sub-events: 600
Events: 41
Active EUs: 7 (with contradictions)
LLM calls: 540
Stream events emitted: 1310
```

### Event Types Emitted

| Event | Icon | Count | Description |
|-------|------|-------|-------------|
| `claim_absorbed` | 🔵 | ~650 | Claim joined existing EU |
| `eu_created` | 🟢 | ~600 | New EU spawned |
| `eu_merged` | 🟡 | 41 | Sub-events merged into events |
| `mass_threshold` | 🎯 | ~20 | EU crossed mass threshold (1.0, 5.0, etc.) |
| `causal_found` | 🟣 | ~10 | Causal language detected |
| `contradiction` | 🔴 | rare | Conflicting claims detected |
| `stabilized` | ✅ | rare | Event settled |
| `activated` | ⚡ | rare | New tension introduced |

### Sample Stream Output

```
🟢 NEW [sub_1]: Two Brown University students were killed...
🟢 NEW [sub_2]: Xi Jinping extended condolences...
🔵 [sub_7] +claim (size=2, mass=0.36)
🔵 [sub_16] +claim (size=5, mass=1.05)
🎯 MASS THRESHOLD [sub_16]: crossed 1.0 (now 1.05)
🟣 CAUSAL in [sub_417]: "Since the Oct. 7, terror attacks in Israel in 2023"
🟡 MERGED → [event_31]: 2 sub-events, 16 claims
```

### Architecture for Production

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PRODUCTION ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │  Claim   │    │ Breathing │    │   Neo4j  │    │   SSE    │      │
│  │  Intake  │───→│  Engine   │───→│  Store   │───→│ Endpoint │      │
│  │  (API)   │    │           │    │          │    │          │      │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
│       ↑               │               ↑               │             │
│       │               ▼               │               ▼             │
│  ┌──────────┐    ┌──────────┐         │          ┌──────────┐      │
│  │  Pages   │    │ PostgreSQL│        │          │  Frontend│      │
│  │ (source) │    │ Embeddings│        │          │   (WS)   │      │
│  └──────────┘    └──────────┘         │          └──────────┘      │
│                                        │                            │
│  ┌─────────────────────────────────────┴──────────────────────┐    │
│  │                    EVENT STREAM                             │    │
│  │  🔵 claim_absorbed  🟢 eu_created  🟡 merged  🎯 threshold  │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Implementation Path

**Phase 1: Core Engine (Done in prototype)**
- ✅ Claim → EU clustering
- ✅ Running centroid embeddings
- ✅ Mass/coherence/tension metrics
- ✅ Event emission stream
- ✅ Periodic merge passes

**Phase 2: Persistence**
- ⬜ Store EUs in Neo4j (new node type)
- ⬜ EU → Claim relationships
- ⬜ EU → EU parent relationships
- ⬜ Save embeddings to PostgreSQL (already working)

**Phase 3: Real-time Streaming**
- ⬜ SSE endpoint: `/api/events/stream`
- ⬜ WebSocket alternative
- ⬜ Frontend visualization

**Phase 4: Curation Loop**
- ⬜ LLM curates events during "quiet" periods
- ⬜ Find internal causal links
- ⬜ Generate event summaries
- ⬜ Suggest frame merges

### Key Decisions Needed

**1. When to emit "ready to publish"?**
```python
def is_publishable(eu):
    return (
        eu.mass() > 5.0 and          # Significant size
        eu.coherence() > 0.8 and     # Internally consistent
        eu.state() == "STABLE" and   # Not actively contradicting
        eu.size() >= 10              # Enough claims
    )
```

**2. When to "freeze" an event?**
```python
def should_freeze(eu):
    # No new claims in 24 hours AND stable
    return eu.hours_since_activity() > 24 and eu.state() == "STABLE"
```

**3. When to create frames?**
```python
def should_frame(events):
    # When 3+ events share narrative or entity hub
    return len(events) >= 3 and (
        entity_overlap(events) > 0.3 or
        narrative_similarity(events) > 0.5
    )
```

### Cost Analysis (Production)

For 1000 claims/day:
```
Embeddings: $0.02 (cached)
Clustering LLM: ~$0.10 (borderline checks)
Curation LLM: ~$0.05 (causal detection)
Total: ~$0.17/day
```

Very affordable for production.

### What Makes Events "Breathe"

1. **Intake**: New claims stream in constantly
2. **Absorption**: Claims find their EU and increase its mass
3. **Tension**: Contradictions create "metabolism" needs
4. **Resolution**: Corroborating claims resolve tension
5. **Growth**: Events merge upward into frames
6. **Stabilization**: Eventually, events "freeze" when no new info

This is the **pulse of the world** - events that grow, contradict, resolve, and stabilize.

### Next Steps

1. **Add SSE endpoint** to FastAPI for live streaming
2. **Create Neo4j schema** for EU nodes
3. **Build frontend visualization** (event cards that pulse)
4. **Implement curation loop** (background job)
5. **Add "decision" emissions** (ready to publish, needs review)

---

*Report generated 2025-12-17*
