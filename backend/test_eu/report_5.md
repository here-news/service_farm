# EU Experiment Report 5

## Unbounded Hierarchy & Emergent Taxonomy

### The Key Insight

The hierarchy doesn't stop at Level 3. It continues until **semantic utility collapses**.

```
Level 0: Claims          → "156 people died in fire"
Level 1: Sub-events      → "Wang Fuk Court death toll"
Level 2: Events          → "Wang Fuk Court Fire"
Level 3: Frames          → "Hong Kong 2025"
Level 4: Meta-frames     → "China-HK Relations 2020s"
Level 5: Narratives      → "Authoritarianism (2020-2025)"
Level N: ...             → continues until useless
```

### When to Stop Merging

Not by level number, but by **semantic utility**:

| Signal | Example | Action |
|--------|---------|--------|
| Label becomes stop-word | "Politics", "Society" | Stop |
| Coherence collapses (<0.2) | Everything fits | Stop |
| No temporal/thematic boundary | Unbounded container | Stop |

**Useful abstractions have boundaries:**
```
✓ "Authoritarianism (2020-2025)" - bounded time, traceable
✓ "AI Race 2024-2025" - specific actors, measurable
✓ "Hong Kong Crisis" - geographic + thematic anchor

✗ "Politics" - everything fits, nothing coheres
✗ "World Events" - meaningless container
✗ "Things" - no filtering value
```

### Stopping Criteria (Proposed)

```python
def should_stop_merging(eu):
    # Junk drawer categories
    if eu.label in ["Politics", "Society", "World", "Events"]:
        return True

    # Coherence collapse (everything fits)
    if eu.coherence < 0.2 and eu.size > 100:
        return True

    # Unbounded temporal spread without anchor
    if eu.time_span > "10 years" and not eu.has_entity_anchor:
        return True

    return False  # Keep emerging
```

### Emergent Taxonomy

**Key question:** When does an EU need a taxonomy tag?

**Approach A: Taxonomy from Day One**
```
Claim → immediately categorize
Problem: Premature - is Jimmy Lai "Politics" or "Human Rights"?
```

**Approach B: Taxonomy Emerges at Higher Levels**
```
Claims → Sub-events → Events → Frames
                              ↓
                       Taxonomy crystallizes here
```

**Observation from experiments:**

| Level | Example | Taxonomy Needed? |
|-------|---------|------------------|
| 0 (Claim) | "156 died in fire" | No - just content |
| 1 (Sub-event) | "death toll updates" | No - just description |
| 2 (Event) | "Wang Fuk Court Fire" | Minimal - place+incident |
| 3 (Frame) | "Hong Kong 2025" | Starting to emerge |
| 4+ (Meta) | "Authoritarianism 2020-2025" | Yes - needs classification |

**Hypothesis:**

```
Small EU:  Just content (what happened)
Large EU:  Content + Pattern (what it means)
Mega EU:   Pattern + Taxonomy (where it fits in knowledge)
```

Taxonomy is the **name we give to the pattern** once it's large enough to see. The system **discovers categories** rather than imposing them.

### Interesting High-Level Frames (Potential)

From our 1215 claims, with more data these could emerge:

```
[META-FRAME] Authoritarianism 2020-2025
  └─ [FRAME] Hong Kong Crackdown
      └─ [EVENT] Jimmy Lai Trial (100 claims)
      └─ [EVENT] National Security Law cases
  └─ [FRAME] Venezuela Crisis
      └─ [EVENT] Oil Tanker Seizure (29 claims)
      └─ [EVENT] Machado Opposition
  └─ [FRAME] Press Freedom Decline
      └─ [EVENT] Jimmy Lai (overlap)
      └─ [EVENT] Russia journalist cases

[META-FRAME] AI Transformation 2024-2025
  └─ [FRAME] AI Leadership
      └─ [EVENT] TIME Person of Year (16 claims)
      └─ [EVENT] Jensen Huang/Nvidia
  └─ [FRAME] AI Regulation Debates
  └─ [FRAME] AI in Warfare
```

### Cross-Frame Linking

Some entities appear in multiple frames:

```
Jimmy Lai:
  ├─ Hong Kong Crackdown (primary)
  ├─ US-China Relations (Trump raised case)
  └─ Press Freedom Global (symbol)

Donald Trump:
  ├─ BBC Lawsuit (specific)
  ├─ Brown University Response (commentary)
  ├─ Venezuela Sanctions (policy)
  └─ Jimmy Lai Advocacy (diplomacy)
```

This creates a **graph, not just a tree**. EUs can have multiple parents at higher levels.

### Architecture Implication

```
Level 0-2: Strict tree (claim belongs to one sub-event)
Level 3+:  Graph allowed (event can belong to multiple frames)
```

### Open Questions

1. **When exactly does taxonomy crystallize?**
   - Size threshold? (>50 claims?)
   - Coherence threshold? (<0.5 means pattern visible?)
   - Entity diversity threshold?

2. **Who names the frames?**
   - LLM generates candidate labels
   - Human curator approves/edits
   - Hybrid: LLM proposes, user feedback refines

3. **How to handle multi-parent at high levels?**
   - Jimmy Lai in both "Hong Kong" and "Press Freedom"
   - Need graph structure, not just tree

4. **Temporal boundaries**
   - "Authoritarianism 2020-2025" - when does it end?
   - Rolling window? Fixed periods? Event-driven boundaries?

### Conclusion

The EU model is **unbounded in depth** but **bounded by utility**.

- Hierarchy emerges naturally through recursive "same narrative?" checks
- Taxonomy is discovered, not imposed
- Stopping happens when abstraction loses coherence
- High-level frames like "Authoritarianism (2020-2025)" are valid and useful

This is the fractal event model working as designed.

---

*Report generated 2025-12-17*
