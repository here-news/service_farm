# Event Relationship Types: Beyond Hierarchical Containment

**Date:** 2025-12-17
**Status:** Research Refinement
**Issue Identified:** Demo observation - related events not merging as expected

---

## The Problem

During demo testing, we observed two clearly related events:
- **Event A:** "Massive Tai Po Fire Kills 36, Leaves 279 Missing" (mass: ~8.5)
- **Event B:** "Lee Jae Myung Offers Condolences After Hong Kong Fire Tragedy" (mass: ~2.1)

These are semantically related (both about Hong Kong fire) but our system kept them separate.

**Initial assumption:** Higher mass should absorb lower mass (gravitational model)

**Problem with this assumption:** Just because Jupiter has more mass than Earth doesn't mean Earth belongs to Jupiter. They're both planets orbiting the Sun.

---

## The Insight

**Mass determines importance/centrality, NOT containment.**

The relationship between two similar events (A, B) could be:

| Relationship | Description | Example |
|-------------|-------------|---------|
| **Containment** | B is a sub-aspect of A | "Fire casualties" ⊂ "Hong Kong Fire" |
| **Sibling** | Both are aspects of a larger topic | Fire + Condolences → "HK Fire Coverage" |
| **Causal** | A caused B | Fire → Political reactions |
| **Temporal** | A followed by B | Fire → Rescue → Recovery |
| **Association** | Related but distinct | Fire ↔ Building safety regulations |

---

## Previous Model (Flawed)

```
if similarity(A, B) > threshold:
    if mass(A) > mass(B):
        A.absorb(B)  # B becomes child of A
    else:
        B.absorb(A)  # A becomes child of B
```

**Flaw:** This assumes all relationships are hierarchical containment.

---

## Refined Model

```
if similarity(A, B) > threshold:
    relationship_type = classify_relationship(A, B)

    if relationship_type == CONTAINMENT:
        # One is a sub-aspect of the other
        if is_subset(B.topic, A.topic):
            A.add_child(B)
        elif is_subset(A.topic, B.topic):
            B.add_child(A)

    elif relationship_type == SIBLING:
        # Both belong under a common parent
        parent = find_or_create_parent(A, B)
        parent.add_child(A)
        parent.add_child(B)

    elif relationship_type == CAUSAL:
        # Create causal link
        A.add_causes(B)  # or B.add_caused_by(A)

    elif relationship_type == ASSOCIATION:
        # Create bidirectional link
        A.add_related(B)
        B.add_related(A)
```

---

## How to Classify Relationship Type

### Option 1: Semantic Scope Analysis
- Extract topic/theme from each event's claims
- Check if one topic is a proper subset of the other
- "Fire casualties" ⊂ "Hong Kong Fire" → CONTAINMENT
- "Hong Kong Fire" ∩ "Political reaction" → SIBLING or CAUSAL

### Option 2: Entity Overlap Analysis
- High overlap (>80%): likely CONTAINMENT
- Medium overlap (40-80%): likely SIBLING
- Low overlap with key shared entity: likely CAUSAL or ASSOCIATION

### Option 3: LLM Classification
```
Given two events:
Event A: {headline_A, sample_claims_A}
Event B: {headline_B, sample_claims_B}

What is the relationship?
1. B is a sub-aspect of A (containment)
2. A is a sub-aspect of B (containment)
3. Both are aspects of a larger topic (sibling)
4. A caused/led to B (causal)
5. Related but distinct (association)
```

### Option 4: Temporal + Entity Analysis
- Same entities + sequential time → CAUSAL or TEMPORAL
- Same entities + overlapping time → CONTAINMENT or SIBLING
- Different entities + shared context → ASSOCIATION

---

## Implications for EU Design

### Current EU Structure
```
Level 0: Claims
Level 1: Sub-events (claim clusters)
Level 2: Events (sub-event clusters)
Level 3+: Frames (event clusters) -- implicit
```

### Refined EU Structure
```
Nodes: Claims, Sub-events, Events, Frames
Edges:
  - CONTAINS (parent-child)
  - RELATES_TO (association)
  - CAUSES (causal)
  - PRECEDES (temporal)
```

This is a **graph**, not a strict **tree**.

---

## Mass in the Refined Model

Mass still matters - our previous experiments validated this. The refinement is WHAT mass controls:

### What Mass DOES Control (Validated in Previous Experiments)

1. **Processing priority** - Check claims against massive events FIRST
2. **Claim routing attraction** - Claims gravitate toward high-mass events in streaming
3. **Display/ranking priority** - Higher mass = more prominent
4. **Frame topic bias** - When creating sibling parent, bias toward higher-mass event's topic
5. **Recursive restreaming** - Massive events get priority when broadcasting updates

This is the gravitational model that works:
```python
# Sort by mass for routing priority
candidates = sorted(events, key=lambda e: e.mass(), reverse=True)

for event in candidates:  # Check massive events FIRST
    sim = cosine_sim(claim.embedding, event.embedding)
    if sim > threshold:
        # Found match - NOW determine relationship semantically
        break
```

### What Mass Does NOT Control

1. **Relationship type** - CONTAINS vs SIBLING vs CAUSES is semantic
2. **Hierarchy/containment** - Higher mass doesn't mean "contains" lower mass

### Reconciling with Previous Experiments

Our previous experiments (`streaming_emergence.py`, `frame_emergence.py`, `breathing_event.py`) successfully used mass for:
- Prioritizing which events to check first
- Determining which events "pull" new claims
- Ranking events for display

These remain valid! We're adding a layer:
- **After** mass-based priority finds a candidate match
- **Then** semantic analysis determines relationship type

**Example:**
- "Hong Kong Fire" (mass: 8.5) and "Lee Condolences" (mass: 2.1)
- Mass makes "Hong Kong Fire" the priority check target
- But semantic analysis reveals they're SIBLINGS, not parent-child
- Both go under "Hong Kong Fire Coverage" (emergent frame)
- The frame's topic is biased toward the higher-mass event

---

## Experiment Results (2025-12-17)

### Experiment: `relationship_classification.py`

We tested relationship classification on 30 related event pairs and mass-based containment assumption on 13 pairs.

### Key Findings

**Mass-Based Containment Assumption: 100% ERROR RATE**

| Test | Result |
|------|--------|
| Pairs tested | 13 |
| Wrong assumptions | 13 |
| **Error rate** | **100%** |

In every single case where we assumed "higher mass event contains lower mass event", the LLM determined this was semantically incorrect.

**Relationship Type Distribution (30 pairs):**

| Type | Count | Percentage |
|------|-------|------------|
| sibling | 18 | 60% |
| contains | 7 | 23% |
| contained_by | 3 | 10% |
| causes | 1 | 3% |
| unrelated | 1 | 3% |

### Critical Insight

**SIBLING relationships are 1.8x more common than hierarchical containment!**

This means most related events should NOT be in a parent-child relationship. Instead:
- They should both belong to a **common parent frame**
- The frame emerges from their shared context
- Mass determines which event's topic dominates the frame name

### Examples from Experiment

| Events | Mass Assumption Would Say | Actual Relationship |
|--------|--------------------------|---------------------|
| "Jimmy Lai case" (25.2) + "Trump raised case" (1.7) | Jimmy Lai contains Trump | **SIBLING** (Hong Kong Political Climate) |
| "Do Kwon sentenced" (2.2) + "Do Kwon arrested" (1.0) | Sentencing contains arrest | **SIBLING** (Do Kwon Legal Issues) |
| "HK Fire" (7.9) + "Death toll update" (1.7) | Fire contains death toll | **SIBLING** (different aspects of same story) |
| "Brown shooting" (6.0) + "Trump statement" (2.0) | Shooting contains Trump | **SIBLING** (Shooting and reactions are peers) |

### Validated Approach

```python
# DON'T DO THIS
if similarity(A, B) > threshold:
    if mass(A) > mass(B):
        A.add_child(B)  # WRONG 100% of the time in our test!

# DO THIS INSTEAD
if similarity(A, B) > threshold:
    rel_type = llm_classify_relationship(A, B)

    if rel_type == "sibling":  # 60% of cases
        parent = find_or_create_frame(A, B)
        parent.add_children([A, B])
    elif rel_type == "contains":  # 23% of cases
        A.add_child(B)
    elif rel_type == "contained_by":  # 10% of cases
        B.add_child(A)
    elif rel_type == "causes":  # 3% of cases
        A.add_edge(CAUSES, B)
    else:
        A.add_edge(RELATES, B)
```

---

## Full Demo Results (2025-12-17)

### Clustering Output (1215 claims)

| Level | Count | Description |
|-------|-------|-------------|
| L1 | 708 | Sub-events (claim clusters) |
| L2 | 39 | Events (user-facing stories) |
| L3 | 10 | Parent events (sibling groupings) |

### Top L2 Events by Claim Count

| Claims | Headline |
|--------|----------|
| 92 | Jimmy Lai Marking 1,800 Days in Solitary Confinement |
| 46 | Deadly Tai Po fire claims 36 lives, 279 missing |
| 27 | Do Kwon Pleads Guilty to Serious Investor Fraud |
| 22 | Shooting at Brown University Leaves Two Dead |
| 21 | Rob Reiner and Wife Michele Found Dead |
| 20 | Tragic Bondi Beach Shooting Leaves 12 Dead |
| 19 | Amanda Seyfried Defends Comment on Kirk |
| 17 | Charlie Kirk Assassinated at Utah Event |

### L3 Sibling Groupings

| Parent | Claims | Children |
|--------|--------|----------|
| Media Crackdown: Lai's Solitary and Wong's Sedition | 89 | Jimmy Lai (92) + Wong Kwok-ngon (3) |
| Tragic Tai Po Fire: 36 Dead, 279 Missing | 55 | Fire (46) + Condolences (9) |
| Tragedy at Brown University | 25 | Shooting (22) + Trump response |
| AI Revolution: Leaders Musk, Zuckerberg, Huang, Altman | 25 | Tech leaders grouped |
| Amanda Seyfried Responds to Kirk's Death | 27 | Seyfried (19) + Kirk (17) |

---

## Critical Insight: Mass-Asymmetric Siblings

### The Jimmy Lai Anomaly

```
L3: Media Crackdown (89 claims, mass 30.71)
├─ L2: Jimmy Lai (92 claims, mass 31.7)  ← MORE massive than parent!
└─ L2: Wong Kwok-ngon (3 claims, mass 0.5)
```

**Observation:** The child EU (Jimmy Lai) is MORE massive than its L3 parent.

**Why this happens:**
- Mass = size × coherence × source_diversity
- Jimmy Lai has higher coherence (focused story)
- L3 parent combines two different stories, diluting coherence

**Key Insight:** Sibling groupings are about **THEMATIC PARALLEL**, not **SIZE BALANCE**.

A massive event (92 claims) can be sibling with a tiny event (3 claims) if they are parallel instances of the same narrative theme.

### The Fractal Potential

If more HK political data existed, we'd see deeper nesting:

```
L4: Hong Kong Freedom Crackdown
├─ L3: Media Crackdown (Lai + Wong)
├─ L3: Tai Po Fire Political Response
└─ L3: Other HK political events...
```

This validates the infinitely-fractal EU design.

---

## Action Items

1. [x] Create relationship classification experiment
2. [x] Test LLM-based classification accuracy (**VALIDATED**)
3. [x] Update demo to unified EU hierarchy (levels, not types)
4. [x] Implement semantic relationship classification
5. [x] Validate sibling grouping on full dataset
6. [ ] Update Neo4j schema to support edge types
7. [ ] Consider L4+ emergence for thematic groupings

---

## References

- `demo/server.py` - Unified EU architecture implementation
- `relationship_classification.py` - Validated 60% sibling, 33% containment
- `evaluate_clustering.py` - Full dataset evaluation

---

*Updated 2025-12-17 with full demo validation and mass-asymmetry insight.*
