# Event Formation Design - Graph Scaffold Framework

## Problem Statement

Events are temporal graphs with phases, causality, and evolution - not flat tables. The current approach fails because:

1. **First-page bias**: Creates flat "Initial Reports" even when page describes mid-event phase
2. **No temporal inference**: Can't detect that "death toll rises" implies earlier unreported phases
3. **No structure prediction**: Doesn't build scaffold with slots for missing phases
4. **Poor matching**: Only uses entity overlap, ignores embeddings and temporal patterns
5. **Ad-hoc growth**: No principled way to attach new information to right phase

## Design Principles

### 1. Event Scale Hierarchy

```
MICRO (single moment/incident)
  - Example: "Fire breaks out at Wang Fuk Court at 17:59"
  - Structure: Single phase, specific time, localized
  - Growth: Can be absorbed into MESO event

MESO (multi-phase bounded event)
  - Example: "2025 Hong Kong Tai Po Fire" (breakout → response → casualties → investigation)
  - Structure: Multiple phases with temporal sequence
  - Growth: Can spawn sub-events, evolve to MACRO

MACRO (extended complex event)
  - Example: "Hong Kong Fire → Building Code Reform → Political Crisis"
  - Structure: Multiple meso events with causality
  - Growth: Long-term consequences, policy changes
```

### 2. Phase Taxonomy

**Core Event Phases** (ordered by typical sequence):

| Phase Type | Keywords | Temporal Markers | Example Claims |
|------------|----------|------------------|----------------|
| **INCIDENT** | "broke out", "started", "occurred" | Earliest time | "Fire started at 17:59" |
| **RESPONSE** | "firefighters", "evacuated", "rescue" | During event | "200 firefighters deployed" |
| **CONSEQUENCE** | "death toll", "casualties", "damage" | Post-event | "128 confirmed dead" |
| **INVESTIGATION** | "probe", "inquiry", "arrested" | Days/weeks after | "Police arrest 2 suspects" |
| **POLITICAL** | "regulations", "reform", "inquiry" | Weeks/months | "Government announces inquiry" |

**Phase Detection Strategy**:
- Analyze claim keywords and temporal markers
- Infer phase from semantic content
- Multiple phases can coexist (concurrent rescue + casualties)

### 3. Scaffold Building Strategy

#### First Page Analysis

When processing the **first page** for an event:

```python
# 1. Detect which phase(s) this page describes
observed_phases = detect_phases_from_claims(claims)
# e.g., ["CONSEQUENCE", "RESPONSE"]

# 2. Infer complete event structure
if "CONSEQUENCE" in observed_phases:
    # This is mid-event reporting
    # Infer earlier phases exist but not yet observed
    scaffold = {
        "INCIDENT": {"status": "inferred", "observed": False},
        "RESPONSE": {"status": "partial", "observed": True},
        "CONSEQUENCE": {"status": "active", "observed": True},
        "INVESTIGATION": {"status": "pending", "observed": False}
    }
    scale = "meso"  # Multi-phase event

elif "INCIDENT" in observed_phases:
    # This is early reporting
    scaffold = {
        "INCIDENT": {"status": "active", "observed": True},
        "RESPONSE": {"status": "pending", "observed": False}
    }
    scale = "micro"  # May grow to meso

# 3. Create phase nodes with slots
for phase_type, state in scaffold.items():
    if state["observed"]:
        # Create actual phase with claims
        create_phase(phase_type, claims=filtered_claims)
    else:
        # Create placeholder phase (no claims yet)
        create_phase_slot(phase_type, expected=True)
```

#### Subsequent Pages

For **pages 2+**:

```python
# 1. Find candidate events (embedding + entity + temporal)
candidates = find_candidate_events(
    embedding=page_embedding,
    entities=page_entities,
    time_window=7_days,
    event_type=inferred_type
)

# 2. Score multi-signal match
for event in candidates:
    score = compute_match_score(
        embedding_similarity=0.4,   # Semantic closeness
        entity_overlap=0.3,          # Shared participants
        temporal_proximity=0.2,      # Time window
        location_match=0.1           # Geographic
    )

# 3. If match found: determine phase attachment
if best_score > threshold:
    target_phase = find_best_phase(
        event=best_event,
        page_claims=claims,
        semantic_match=True
    )
    attach_to_phase(target_phase, claims)

# 4. If no match: create new event with scaffold
else:
    create_new_event_with_scaffold(page)
```

### 4. Multi-Signal Event Matching

**Matching Algorithm** (replaces pure entity overlap):

```python
def compute_event_match_score(page, candidate_event):
    """
    Multi-signal scoring for event attachment

    Weights empirically tuned:
    - Embedding: 40% (semantic similarity)
    - Entities: 30% (participant overlap)
    - Temporal: 20% (time proximity)
    - Location: 10% (geographic match)
    """
    signals = {}

    # 1. Semantic similarity (embedding cosine)
    signals['embedding'] = cosine_similarity(
        page.embedding,
        candidate_event.centroid_embedding
    )

    # 2. Entity overlap (Jaccard)
    signals['entities'] = jaccard_similarity(
        page.entity_names,
        candidate_event.entity_names
    )

    # 3. Temporal proximity
    days_diff = abs((page.pub_time - candidate_event.latest_time).days)
    signals['temporal'] = max(0, 1 - days_diff / 7)  # 7-day window

    # 4. Location match (binary or Jaccard)
    signals['location'] = location_overlap(
        page.locations,
        candidate_event.locations
    )

    # Weighted combination
    score = (
        0.4 * signals['embedding'] +
        0.3 * signals['entities'] +
        0.2 * signals['temporal'] +
        0.1 * signals['location']
    )

    # Decision thresholds
    if score >= 0.50:
        return "ATTACH"
    elif score >= 0.30:
        return "RELATE"  # Create relationship, not merge
    else:
        return "SPAWN"  # New event
```

### 5. Phase Assignment Logic

**When attaching to existing event**:

```python
def find_best_phase(event, new_claims):
    """
    Determine which phase these claims belong to

    Strategy:
    1. Match by temporal markers + overlap
    2. Match by semantic similarity (claim embeddings)
    3. Match by phase keywords
    4. Create new phase if no good match
    5. Allow concurrent phases; don't force a single sequence
    """

    # Detect phase type of new claims
    detected_phases = detect_phases_from_claims(new_claims)

    for phase_type in detected_phases:
        # Check if event has this phase
        existing_phase = event.get_phase(phase_type)

        if existing_phase and existing_phase.status == "inferred":
            # Fill in the inferred phase slot
            return existing_phase

        elif existing_phase and existing_phase.status == "active":
            # Temporal overlap check
            if temporal_overlap(new_claims, existing_phase) and semantic_fit(new_claims, existing_phase):
                return existing_phase

        else:
            # Create new phase
            new_phase = create_phase(
                event_id=event.id,
                phase_type=phase_type,
                sequence=infer_sequence(event, phase_type)
            )
            return new_phase
```

### 6. Event Evolution States

Events transition through states as evidence accumulates:

```
provisional (1 page)
    ↓
emerging (2-4 pages, structure becoming clear)
    ↓
stable (5+ pages, multi-phase structure confirmed)
    ↓
mature (long-term tracking, political/reform phases)
```

**State transitions trigger**:
- Scaffold refinement (provisional → emerging: confirm phase structure)
- Canonical title synthesis (emerging → stable: synthesize from 5+ sources)
- Sub-event extraction (stable → mature: investigation becomes sub-event)

### 7. Umbrella Event Strategy

**When first page detects multiple distinct phases:**

```python
# Heuristic: If >= 3 phases detected from first page
if len(detected_phases) >= 3:
    # This is comprehensive reporting of complex event
    # Create UMBRELLA event structure

    umbrella_event = create_umbrella_event(
        title="2025 Hong Kong Tai Po Fire",
        scale="meso",  # or macro if political/reform phases detected
        coherence=0.4  # LOW - structure uncertain from single source
    )

    # Create SUB-EVENTS for each major phase cluster
    for phase_cluster in group_phases_by_topic(detected_phases):
        sub_event = create_sub_event(
            parent=umbrella_event,
            phase_cluster=phase_cluster
        )
        umbrella_event.add_child(sub_event, relationship="PART_OF")
```

**Example Decision Tree:**

```
IF first_page detects:
  - Only INCIDENT → Create micro event (single phase)
  - INCIDENT + RESPONSE → Create meso event (2 phases, sequential)
  - INCIDENT + CONSEQUENCE → Create meso with inferred RESPONSE between
  - INCIDENT + RESPONSE + CONSEQUENCE → Create UMBRELLA with potential sub-events
  - CONSEQUENCE only → Infer earlier INCIDENT (not observed)
  - INVESTIGATION + POLITICAL → Create umbrella, infer main event happened earlier
```

### 8. Practical Example: Hong Kong Fire

**Page 1** (DW, Nov 26 09:39): "Death toll rises as blaze engulfs high-rise"

```python
# Claim Analysis (keyword detection)
claims = [
    "Four people confirmed dead",           # CONSEQUENCE
    "blaze engulfs high-rise",              # INCIDENT (present tense!)
    "firefighters battle flames",           # RESPONSE
    "occurred in Tai Po district",          # INCIDENT
    "evacuations underway",                 # RESPONSE
    "death toll expected to rise"           # CONSEQUENCE
]

# Phase Detection
detected_phases = detect_phases_from_claims(claims)
# → ["INCIDENT", "RESPONSE", "CONSEQUENCE"]

# Common-sense check: 3 phases from single page
# This is comprehensive reporting of ongoing major event
# Decision: Create UMBRELLA event

# Structure Created:
umbrella = Event(
    title="2025 Hong Kong Tai Po Fire",
    scale="meso",
    coherence=0.45,  # LOW - only one source, but comprehensive
    status="provisional"
)

# Create sub-structure (flat for now, may spawn sub-events later)
umbrella.phases = [
    Phase(
        name="Fire Breakout",
        type="INCIDENT",
        status="observed",  # "blaze engulfs" is PRESENT TENSE
        claims=["blaze engulfs high-rise", "occurred in Tai Po"],
        start_time="inferred from temporal markers",
        confidence=0.7
    ),
    Phase(
        name="Emergency Response",
        type="RESPONSE",
        status="observed",
        claims=["firefighters battle flames", "evacuations underway"],
        confidence=0.8
    ),
    Phase(
        name="Casualty Assessment",
        type="CONSEQUENCE",
        status="active",
        claims=["Four people confirmed dead", "death toll expected to rise"],
        confidence=0.9
    ),
    Phase(
        name="Investigation",
        type="INVESTIGATION",
        status="pending",  # Expected later
        claims=[],
        confidence=0.0
    )
]

# Graph structure:
(umbrella:Event {coherence: 0.45})
  -[:HAS_PHASE {sequence: 1}]->(breakout:Phase)
      -[:SUPPORTED_BY]->(c1:Claim {text: "blaze engulfs..."})
  -[:HAS_PHASE {sequence: 2}]->(response:Phase)
      -[:SUPPORTED_BY]->(c2:Claim {text: "firefighters battle..."})
  -[:HAS_PHASE {sequence: 3}]->(casualties:Phase)
      -[:SUPPORTED_BY]->(c3:Claim {text: "Four people confirmed dead"})
  -[:HAS_PHASE {sequence: 4, status: "pending"}]->(investigation:Phase)
```

**Page 2** (NYPost, Nov 26 10:15): "36 dead as blaze rips through towers"

```python
# Phase Detection
detected_phases = detect_phases_from_claims(claims)
# → ["CONSEQUENCE", "RESPONSE"]  # Casualty update + ongoing fire

# Matching against umbrella event
embedding_sim = 0.82  # Very similar semantics
entity_overlap = 0.71  # Hong Kong, Tai Po, Fire Services
temporal_diff = 0.6 hours  # Very close in time
→ Total score: 0.68 → ATTACH

# Common-sense decision tree:
# Q1: Are these claims about same PHASE or different phase?
if detected_phases overlap with umbrella.active_phases:
    # Same phase update → ATTACH to existing phase
    decision = "ATTACH_TO_PHASE"
    target_phase = umbrella.get_phase("CONSEQUENCE")

    # Add claims to phase
    target_phase.add_claims(filtered_claims)

    # Track evolution
    create_evolution_edge(
        old_claim="Four people confirmed dead",
        new_claim="36 dead as blaze rips through towers",
        relationship="EVOLVED_TO"
    )

    # Update umbrella coherence (more sources = higher confidence)
    umbrella.coherence = 0.45 + 0.15 = 0.60  # 2 sources now

elif detected_phases are NEW phases not in umbrella:
    # New phase emerging → Could be sub-event or new phase
    # Check temporal gap and semantic shift

    if temporal_gap > 3_days and semantic_shift > 0.3:
        # This might be SUB-EVENT (e.g., Investigation)
        decision = "CREATE_SUB_EVENT"

        sub_event = create_sub_event(
            parent=umbrella,
            title="2025 Hong Kong Fire Investigation",
            scale="meso",
            relationship="CAUSED"  # Main fire CAUSED investigation
        )
        umbrella.add_child(sub_event, "CAUSED")

    else:
        # Just new phase in same umbrella
        decision = "ADD_PHASE"
        new_phase = create_phase(umbrella, phase_type)
```

**Page 3** (BBC Live, Nov 26 09:33 - earliest!): "Fire started at approximately 17:59"

```python
# Matching
embedding_sim = 0.75
entity_overlap = 0.68
temporal_diff = EARLIER than event.earliest_time!
→ Total score: 0.64 → ATTACH

# Phase assignment
detected_phases = ["INCIDENT"]
best_phase = event.get_phase("INCIDENT")  # Currently inferred
→ FILL the inferred phase with actual claims!
→ Update event.earliest_time = 2025-11-26 17:59
→ Phase: Fire Breakout now has observed claims
```

**Page 4** (Global Voices, Dec 2): "Critics say fire was due to negligence"

```python
# Matching
embedding_sim = 0.58
entity_overlap = 0.45
temporal_diff = 6 days
→ Total score: 0.51 → ATTACH (still within window)

# Phase assignment
detected_phases = ["POLITICAL", "INVESTIGATION"]
best_phase = None  # No political phase exists yet
→ Create new phase: Political Response
→ Add claims about negligence, scaffolding criticism
```

## Implementation Plan

### Phase 1: Core Scaffold (This Session)
1. ✅ Neo4j service layer
2. ✅ Canonical title synthesis
3. ⏳ Multi-signal event matching
4. ⏳ Phase detection from claims
5. ⏳ Scaffold building with inferred phases
6. ⏳ Decision math (energy/continuity/contradiction policy)

### Phase 2: Phase Intelligence
1. Phase taxonomy implementation
2. Temporal inference logic
3. Phase assignment algorithm
4. Claim evolution tracking (EVOLVED_TO edges)

### Phase 3: Event Evolution
1. State transitions (provisional → emerging → stable)
2. Sub-event extraction (Investigation as separate meso event)
3. Causality detection (CAUSED, TRIGGERED relationships)
4. Narrative generation from graph traversal

### Phase 4: Frontend Integration
1. Neo4j API endpoints
2. Timeline visualization (phases on timeline)
3. Graph explorer (D3.js network view)
4. Narrative generator (prose from graph)

### 9. Common-Sense Heuristics (Rigorous Checking)

**Principle: Let data guide structure, but apply domain knowledge**

#### Heuristic 1: Multi-Phase First Page → Umbrella Event
```python
if len(detected_phases) >= 3 from single_page:
    # Comprehensive reporting → Major event
    create_umbrella_event(coherence=0.4)  # Low initially, will increase
else:
    # Focused reporting → Simple event
    create_simple_event()
```

#### Heuristic 2: Temporal Gap → Sub-Event Boundary
```python
if temporal_gap(new_page, umbrella) > 3_days:
    if semantic_shift > 0.3:  # Topic changed significantly
        # Investigation, Political Response, etc.
        create_sub_event_under_umbrella()
    else:
        # Just delayed reporting of same event
        attach_to_existing_phase()
```

#### Heuristic 3: Contradictory Claims → Separate Phases
```python
if detect_contradiction(new_claims, existing_claims):
    # E.g., "4 dead" vs "128 dead"
    # Check if this is EVOLUTION or CONTRADICTION

    if temporal_order_makes_sense:
        # Evolution: 4→36→128 over time
        create_evolution_edge(old, new, "EVOLVED_TO")
    else:
        # Contradiction: competing reports
        mark_contested(field="death_toll", values=[old, new])
        # Only branch/split if an alternate hypothesis lowers energy and continuity is low
```

#### Heuristic 4: Scale Promotion
```python
# Start conservative, promote as evidence accumulates
if umbrella.coherence < 0.5:
    # Still uncertain, keep as provisional
    umbrella.status = "provisional"

elif umbrella.coherence >= 0.5 and umbrella.pages_count >= 3:
    # Structure becoming clear
    umbrella.status = "emerging"

elif umbrella.coherence >= 0.7 and umbrella.pages_count >= 5:
    # Well-established multi-source event
    umbrella.status = "stable"

    # Check if phases should become sub-events
    for phase in umbrella.phases:
        if phase.claims_count > 10 and phase.temporal_span > 2_days:
            # Phase is substantial enough to be own event
            promote_phase_to_sub_event(phase)
```

#### Heuristic 5: Location Hierarchy → Event Nesting
```python
# "Hong Kong Fire" vs "Tai Po Fire" vs "Wang Fuk Court Fire"
if location_is_more_specific(new_page, umbrella):
    # New page is about specific sub-location
    # This might be micro event within meso event

    if umbrella.scale == "meso":
        # Check if this is the SAME fire or DIFFERENT fire
        if same_time_window and entity_overlap > 0.6:
            # Same fire, just more specific reporting
            attach_to_umbrella()
        else:
            # Different fire in same city
            create_separate_event()
```

#### Heuristic 6: Next Page Decision Flow
```
For each new page:
  ├─ Find candidate events (embedding + entity + temporal)
  │
  ├─ IF best_match_score > 0.50:
  │   ├─ Detect phases in new page
  │   │
  │   ├─ IF phases overlap with existing:
  │   │   └─ ATTACH to existing phase
  │   │
  │   ├─ ELIF phases are NEW but temporal_gap < 3 days:
  │   │   └─ ADD new phase to umbrella
  │   │
  │   └─ ELSE temporal_gap > 3 days AND semantic_shift > 0.3:
  │       └─ CREATE sub-event (Investigation, Political)
  │
  ├─ ELIF best_match_score 0.30-0.50:
  │   └─ CREATE relationship (RELATED_TO, SIMILAR)
  │
  └─ ELSE best_match_score < 0.30:
      └─ CREATE new umbrella event
```

## Open Questions

1. **Phase overlap handling**: How to handle concurrent phases? (Response + Casualties happening simultaneously)
   - **Answer**: Allow concurrent phases, mark with temporal overlap in graph

2. **Threshold tuning**: What are the right values for ATTACH/RELATE/SPAWN?
   - **Proposal**: 0.50 attach, 0.30 relate, <0.30 spawn (tune empirically)

3. **Embedding updates**: Should event centroid embedding update as pages attach?
   - **Answer**: Yes, weighted average of page embeddings, decay older pages

4. **Phase sequence flexibility**: Some events don't follow standard sequence
   - **Answer**: Don't enforce sequence, use temporal markers and keywords

5. **Sub-event criteria**: When should a phase become its own meso event?
   - **Answer**: When phase has >10 claims, >2 day span, distinct participants

## Success Metrics

- **Consolidation rate**: % of related pages correctly attached to same event
- **Phase accuracy**: % of claims placed in correct phase
- **Temporal coherence**: Event timeline makes chronological sense
- **Narrative quality**: Can generate readable story from graph traversal
- **Graph completeness**: % of events with multi-phase structure vs flat

## Next Steps

Should we proceed with implementing:
1. Multi-signal matching algorithm (embedding + entities + temporal + location)?
2. Phase detection from claim analysis?
3. Scaffold building with inferred phases?
4. Decision math: merge/split symmetry, continuity guard, contradiction policy?

Or do you want to refine the design further before coding?

## Decision Math (symmetry, stability, contradiction policy)

### Energy-based merge/split (symmetric)
- Define event/phase energy:  
  `E = λ1·temporal_dispersion + λ2·entity_entropy + λ3·contradiction_load – λ4·coherence_gain`
- Merge if `E(A∪B) < E(A)+E(B)`; split if children lower `E`. Log components for inspectability.
- Intuition: don’t over-merge if it increases dispersion/entropy/contradiction; do merge when coherence gain outweighs costs.

### Continuity guard (block oscillations)
- Define continuity between successive states:  
  `continuity(E_t, E_t+1) = f(overlap of claims/entities, causal continuity, entropy delta)`
- If continuity is high, block merge→split→merge oscillations; require a significant energy decrease and low continuity to split.
- Intuition: keep stable IDs unless evidence genuinely diverges.

### Contradiction handling (built-in, not branching by default)
- Track fact evolution per field (e.g., death_toll: 4→36→44→128) with timestamps/sources; latest resolved value surfaces in summary; prior/conflicting stay in a contested/evolution panel.
- Apply `coherence_penalty` in likelihood; `contradiction_load` in energy.
- Split/branch only when a rival hypothesis lowers energy AND continuity is low; otherwise mark contested, don’t proliferate phases.

### Priors and thresholds (bounded, tunable)
- Priors: recency, structural type (anchor vs emergent), entropy (lower H = higher prior), source diversity. Never 0/1.
- Attach/relate/spawn thresholds remain tunable (e.g., attach ~0.50, relate ~0.30); cap per-claim contribution and group by source/page to avoid correlated inflation.

### Compute guardrails (economic)
- Recluster only the local connected component on new claims; defer global reconciliation to scheduled passes.
- Size gates: skip heavy steps (causal extraction, full synthesis) on tiny/low-confidence events; promote when evidence crosses thresholds.
- DAG relationships: allow multiple parents; prevent cycles when adding PHASE_OF/PART_OF/CAUSES/RELATED edges.
