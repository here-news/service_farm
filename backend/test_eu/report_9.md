# EU Experiment Report 9

## Causal & Consequence Relationship Detection

### Key Questions

1. Can we detect causal language in claims?
2. Can we find relationships between events as they emerge?

### Results Summary

```
Causal language detection: 18% of claims have causal patterns
Inter-event relationships: 1 causal relationship found
```

### Phase 1: Causal Language in Claims

**18 out of 100 sampled claims** contain causal language:

| Type | Count | Example |
|------|-------|---------|
| Cause | 8 | "Do Kwon's actions contributed to the collapse..." |
| Attribution | 6 | "police allege negligence" |
| Effect | 4 | "King Charles issued a heartfelt message after a deadly blaze" |

**Key patterns found:**

```
Temporal causation:
  "The massacre at Bondi Beach followed a wave of antisemitic attacks"
  "King Charles issued a heartfelt message after a deadly blaze"

Attribution (blame/credit):
  "Hong Kong apartment fire death toll rises as police allege negligence"
  "Donald Trump suggested Rob Reiner died of 'Trump derangement syndrome'"

Consequential action:
  "Amanda Seyfried addressed the backlash in a subsequent statement"
  "Washington is ratcheting up political and economic pressure"
```

**References to other events:**
- 7 out of 18 causal claims (39%) explicitly reference another event
- This creates natural "links" between events

### Phase 2: Inter-Event Relationships

Only **1 causal relationship** detected between events:

```
Starlink satellite near-collision
  → CAS Space response

Relationship: causal (A_to_B)
Confidence: 85%
Explanation: The close approach of Starlink satellite caused response
```

### Why So Few Inter-Event Relationships?

1. **Events are semantically isolated** - Most events in our dataset are independent stories
2. **Similarity filtering** - We only checked pairs with 0.3-0.7 similarity (avoiding same-event and unrelated)
3. **Data sparsity** - 1215 claims spread across many topics

### What This Reveals

**Causal relationships exist at two levels:**

```
INTRA-EVENT (within same event):
  "Fire caused deaths" ← both in Wang Fuk Court Fire event
  "Investigation found negligence" ← both in same event

INTER-EVENT (between events):
  "Antisemitic attacks → Bondi Beach shooting" ← different events
  "Starlink collision → CAS Space response" ← different events
```

Most causal language is **intra-event** (connecting sub-events), not **inter-event**.

### Implications for EU Model

**Current architecture supports:**
- ✅ Corroboration (claims support each other)
- ✅ Contradiction (claims conflict)
- ✅ Entity co-occurrence (shared actors)

**Should add:**
- ⬜ Causal links between sub-events within same event
- ⬜ Causal links between events (rare but important)
- ⬜ Temporal sequence relationships

### Proposed Relationship Types

```python
class EURelationship:
    source_id: str
    target_id: str
    relationship_type: Literal[
        'corroborates',   # Existing
        'contradicts',    # Existing
        'causes',         # NEW: A led to B
        'responds_to',    # NEW: B is reaction to A
        'follows',        # NEW: B happened after A (temporal)
        'parallels',      # NEW: A and B concurrent
    ]
    confidence: float
    evidence: str  # The claim text that establishes link
```

### Causal Language as Signal

Claims with causal language (18%) are **high-value signals**:

```
If claim contains "caused", "led to", "following", "as a result":
  → Extract referenced event
  → Check if referenced event exists in system
  → If yes, create causal link
  → If no, flag as potential missing event
```

### Example: Wang Fuk Court Fire

Within the Wang Fuk Court Fire event, we could detect:

```
[SUB-EVENT] Fire outbreak
    │
    ├─ causes → [SUB-EVENT] Casualties
    ├─ causes → [SUB-EVENT] Evacuation
    │
[SUB-EVENT] Investigation
    │
    └─ responds_to → [SUB-EVENT] Fire outbreak

[SUB-EVENT] Death toll updates
    │
    └─ follows → [SUB-EVENT] Casualties (temporal)
```

This creates a **directed graph within events**, not just tree structure.

### Cost Analysis

```
Causal language detection: 100 LLM calls
Inter-event relationships: ~20 LLM calls
Total: ~$0.01
```

Very cheap compared to clustering.

### Answers to Your Questions

**Q1: Do we need evolving taxonomy?**
- At levels 0-2: No (content is self-describing)
- At levels 3+: Yes, let taxonomy emerge from data
- Don't pre-define categories; discover them

**Q2: Can we find relationships between events?**
- YES, via causal language detection in claims
- 18% of claims have causal patterns
- Most are intra-event (sub-event to sub-event)
- Some are inter-event (event A → event B)

**Q3: Did you generate sub-event embeddings on the fly?**
- YES, using running centroid:
```python
new_embedding = (old_embedding * (n-1) + claim_embedding) / n
```
- No separate API call for sub-event embedding
- Event embeddings = average of sub-event embeddings

### Next Steps

1. ⬜ Extract causal links during claim ingestion
2. ⬜ Build sub-event DAG within events
3. ⬜ Detect inter-event causal chains
4. ⬜ Visualize event causality graph

---

*Report generated 2025-12-17*
