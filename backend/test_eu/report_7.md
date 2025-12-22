# EU Experiment Report 7

## Cross-Frame Entity Linking: Graph Structure at Higher Levels

### The Key Finding

**Donald Trump appears in 5 separate events**, creating a natural hub for a "Trump Presidency 2.0" meta-frame:

```
'donald trump' appears in 5 events:
  └─ BBC defamation lawsuit (22 claims)
  └─ Brown University shooting response (35 claims)
  └─ Venezuela/Machado policy (7 claims)
  └─ Jimmy Lai advocacy (100 claims)
  └─ Rob Reiner interview (12 claims)
```

These are **semantically distinct events** that don't merge based on content, but share a key actor.

### Cross-Event Entities

| Entity | Events | Potential Frame |
|--------|--------|-----------------|
| Donald Trump | 5 | Trump Presidency 2.0 |
| Hong Kong | 2 | Hong Kong 2025 |
| Venezuela | 2 | Venezuela Crisis |
| Do Kwon | 2 | Crypto Fraud Cases |
| Maria Corina Machado | 2 | Venezuelan Opposition |
| Rob Reiner | 2 | Reiner Family Saga |
| Charlie Kirk | 2 | Kirk/TPUSA Controversy |

### Entity Overlap Analysis

Event pairs with highest entity overlap (Jaccard similarity):

| Pair | Jaccard | Shared |
|------|---------|--------|
| Charlie Kirk assassination ↔ Amanda Seyfried response | 0.25 | Charlie Kirk |
| BBC lawsuit ↔ Brown shooting | 0.25 | Donald Trump |
| Do Kwon sentencing ↔ Do Kwon currencies | 0.20 | Do Kwon |
| Venezuela oil ↔ Machado opposition | 0.20 | Venezuela |
| Jimmy Lai ↔ Wang Fuk Court Fire | 0.12 | Hong Kong |

### Graph vs Tree Structure

**Key insight**: At higher levels, the hierarchy becomes a **graph**, not a tree.

```
TREE (Level 0-2):
  claim → sub-event → event
  (each claim belongs to exactly one sub-event)

GRAPH (Level 3+):
  event → multiple frames
  (Jimmy Lai belongs to both "Hong Kong 2025" AND "Press Freedom")
```

### The "Hong Kong" Problem Solved

Previously we wondered why Jimmy Lai Trial and Wang Fuk Court Fire didn't merge into a "Hong Kong" frame.

**Answer**: They're linked by ENTITY (Hong Kong), not NARRATIVE.

```
Jimmy Lai Trial (100 claims)
  ├─ Primary narrative: Press Freedom / Human Rights
  └─ Entity link: Hong Kong

Wang Fuk Court Fire (68 claims)
  ├─ Primary narrative: Building Safety / Disaster
  └─ Entity link: Hong Kong
```

Both belong to "Hong Kong 2025" via entity linking, but NOT via semantic narrative merge.

### Proposed Multi-Level Architecture

```
Level 0-2: TREE structure
  - Claims merge into sub-events via semantic similarity
  - Sub-events merge into events via "same story?" check
  - Each node has exactly one parent

Level 3+: GRAPH structure
  - Events link to multiple frames via:
    a) Semantic narrative merge (strong)
    b) Entity co-occurrence (weak)
  - Nodes can have multiple parents
```

### Implementation Implications

```python
class EU:
    # For levels 0-2
    parent_id: Optional[str]  # Single parent (tree)

    # For levels 3+
    frame_links: List[FrameLink]  # Multiple links (graph)

@dataclass
class FrameLink:
    frame_id: str
    link_type: str  # 'narrative' | 'entity' | 'causal'
    strength: float  # 0.0-1.0
```

### Entity Hub Detection

Entities appearing in 2+ events are "hubs" that could anchor frames:

```
Hub Detection Algorithm:
1. Extract entities from all events
2. Count cross-event appearances
3. If entity appears in 3+ events → candidate hub
4. Generate frame name from hub context
```

For Trump (5 events):
```
Candidate frame: "Trump Presidency 2.0"
  - BBC defamation (media relations)
  - Brown shooting (policy response)
  - Venezuela (foreign policy)
  - Jimmy Lai (diplomacy)
  - Reiner interview (public discourse)
```

### Weak vs Strong Links

| Link Type | Example | Merge? |
|-----------|---------|--------|
| **Strong (narrative)** | Do Kwon sentencing ↔ Do Kwon currencies | YES - same story |
| **Weak (entity only)** | Jimmy Lai ↔ Wang Fuk Court Fire | NO - different stories |
| **Medium (thematic)** | Venezuela oil ↔ Machado opposition | MAYBE - related theme |

### Statistics

```
Entities appearing in 2+ events: 7
Event pairs with entity overlap: 16
Average overlap (where exists): 0.161
```

Low average overlap (16%) confirms events are semantically distinct but entity-linked.

### Conclusion

**The EU model naturally produces graph structure at higher levels:**

1. **Semantic merging** (embeddings + LLM) creates clean tree hierarchy for levels 0-2
2. **Entity co-occurrence** creates graph links at level 3+
3. **Donald Trump** is the dominant hub entity in our dataset (5 events)
4. **Hong Kong** correctly links Jimmy Lai and Wang Fuk Court Fire without forcing a narrative merge

This validates the architecture proposed in Report 5:
```
Level 0-2: Strict tree (claim belongs to one sub-event)
Level 3+:  Graph allowed (event can belong to multiple frames)
```

### Next Steps

1. Implement `FrameLink` data structure for multi-parent relationships
2. Create "hub-based frames" that group events by shared entities
3. Visualize the graph structure at level 3+
4. Explore taxonomy emergence for hub-based frames

---

*Report generated 2025-12-17*
