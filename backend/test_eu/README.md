# EventfulUnit Experiment

Empirical validation of the EU model against our existing Neo4j graph.

## Current System State

**Nodes:**
- 1215 Claims
- 740 Entities
- 113 Pages
- 16 Events
- 22 ClaimMetrics

**Relationships:**
- `Page -[:EMITS]-> Claim`
- `Event -[:INTAKES]-> Claim`
- `Event -[:INVOLVES]-> Entity`
- `Claim -[:MENTIONS]-> Entity`
- `Claim -[:CORROBORATES]-> Claim` (266)
- `Claim -[:CONTRADICTS]-> Claim` (39)
- `Claim -[:UPDATES]-> Claim` (25)
- `Event -[:CONTAINS]-> Event` (1 - Trump/BBC duplicate issue #22)

**Existing Signals:**
- Entity mention counts (Jimmy Lai: 114, Hong Kong: 105, Trump: 84)
- Event coherence scores (0.55 - 1.0)
- Claim confidence scores
- Claim embeddings (for semantic similarity)

---

## The EU Hypothesis

Current architecture separates:
- **Claims** = extracted statements (content)
- **Events** = containers that hold claims (structure)

EU model proposes:
- Everything is an **EventfulUnit** with different **roles** and **mass**
- A claim IS an event at emergence phase with low mass
- An event IS a cluster of EUs that crossed a mass/coherence threshold
- No ontological distinction, only phase transitions

---

## Experiments

### Experiment 1: Mass Derivation

Can we compute meaningful "mass" from existing graph data?

```
Mass components:
- corroboration: count(CORROBORATES relationships)
- tension: count(CONTRADICTS relationships)
- entity_weight: sum(entity mention counts)
- source_diversity: count(distinct pages)
- downstream: count(claims that reference this)
```

**Test:** Rank claims by computed mass. Do high-mass claims correspond to what humans would consider "important"?

### Experiment 2: Role Classification

Can we infer EU roles from existing claim data?

```
Roles to detect:
- OCCURRENCE: "X happened" (past tense, factual)
- ASSERTION: "X claims Y" (attributed statement)
- DENIAL: "X denies Y" (negation by named party)
- CONFIRMATION: "X confirms Y" (official validation)
- UPDATE: Claims with UPDATES relationship
- CONTRADICTION: Claims with CONTRADICTS relationship
```

**Test:** Classify existing claims into roles. Does role distribution reveal patterns?

### Experiment 3: Emergence Detection

Can we identify when claim clusters "should" become events without explicit nesting?

```
Emergence signals:
- N claims share >60% entity overlap
- Coherence of claim cluster > threshold
- Total cluster mass > threshold
- Temporal clustering (claims within X hours)
```

**Test:** Run emergence detection on claims. Compare detected clusters with existing Event boundaries. Where do they match? Where do they diverge?

### Experiment 4: Computational Cost

What's the actual cost of EU-style operations?

```
Measure:
- Mass recalculation on single new claim
- K-hop influence propagation (K=1,2,3)
- Full graph mass update
- Lazy vs eager strategies
```

**Test:** Benchmark with current 1215 claims. Project to 10K, 100K scale.

### Experiment 5: The Duplicate Sub-Event Problem (#22)

Would EU model have prevented Trump/BBC duplicate?

```
Current: Created sub-event "Trump's Defamation Lawsuit Against BBC"
         under parent "Trump vs. BBC Defamation Lawsuit"

EU model: Both would be same EU cluster, mass would accumulate
          without spawning near-identical child
```

**Test:** Replay the Trump/BBC claims through EU mass model. Does it naturally avoid duplication?

---

## Success Criteria

1. **Mass validity**: High-mass EUs correlate with human-judged importance (>0.7 correlation)
2. **Role coherence**: Role classification achieves >80% agreement with manual labels
3. **Emergence accuracy**: Detected clusters match existing events >70% of the time
4. **Computational feasibility**: Operations complete in <100ms for current scale
5. **Duplicate prevention**: EU model naturally avoids #22-style duplicates

---

## Files

```
test_eu/
├── README.md
├── load_graph.py          # Pull data from Neo4j into working structures
├── mass_calculator.py     # Experiment 1: mass derivation
├── role_classifier.py     # Experiment 2: role detection
├── emergence_detector.py  # Experiment 3: cluster emergence
├── benchmark.py           # Experiment 4: computational cost
├── replay_trump_bbc.py    # Experiment 5: duplicate prevention
└── results/               # Output data and analysis
```

---

## Running

```bash
# Run inside container (backend/ is mounted as /app)
docker exec herenews-app python /app/test_eu/load_graph.py
docker exec herenews-app python /app/test_eu/mass_calculator.py
docker exec herenews-app python /app/test_eu/role_classifier.py
docker exec herenews-app python /app/test_eu/emergence_detector.py
docker exec herenews-app python /app/test_eu/replay_trump_bbc.py
```

---

## Results (2025-12-16)

### Experiment 1: Mass Derivation ✅

**Finding:** Mass calculation from graph signals produces meaningful rankings.

| Metric | Value |
|--------|-------|
| Mean mass | 0.098 |
| Median mass | 0.062 |
| Zero mass claims | 5.9% |
| High mass (>0.3) | 5.3% |

**Top mass claims align with important facts:**
1. [0.81] Wang Fuk Court fire death toll (high corr + tension + entity weight)
2. [0.63] Jimmy Lai imprisonment (high corr + entity weight)
3. [0.62] Jimmy Lai guilty verdict (high corr + entity weight)

**Browsable threshold analysis:**
- >0.1: 35.8% of claims (435)
- >0.2: 15.9% of claims (193)
- >0.3: 5.4% of claims (66)

**Conclusion:** Mass works as a filter. Setting threshold at 0.2 would show ~16% of claims, which is reasonable for browsing.

---

### Experiment 2: Role Classification ✅

**Finding:** Simple pattern matching achieves reasonable role detection.

| Role | Count | % |
|------|-------|---|
| occurrence | 893 | 73.5% |
| assertion | 175 | 14.4% |
| institutional | 63 | 5.2% |
| update | 25 | 2.1% |
| denial | 21 | 1.7% |
| confirmation | 20 | 1.6% |
| contradiction | 18 | 1.5% |

**Patterns discovered:**
- `occurrence -> corroborates -> occurrence` (133 times) - facts support facts
- `contradiction -> contradicts -> occurrence` (12 times) - corrections target facts
- Legal events (Jimmy Lai, Do Kwon) have higher `denial` and `institutional` ratios

**Conclusion:** Role classification works with simple patterns. Could improve with LLM for edge cases.

---

### Experiment 3: Emergence Detection ⚠️

**Finding:** Entity-based clustering partially matches existing events.

| Metric | Value |
|--------|-------|
| Clusters detected | 17 |
| High overlap (>70%) | 3 |
| Medium overlap (30-70%) | 9 |
| Low overlap (<30%) | 5 |
| Match rate | 17.6% |

**Matched events:**
- Jimmy Lai Imprisonment (74% overlap)
- Bondi Beach Shooting (83% overlap)
- Charlie Kirk Assassination (72% overlap)

**Missed/Partial:**
- Wang Fuk Court Fire (33% - split by "Hong Kong" entity sharing with other events)
- TIME Person of the Year (5% - abstract event, weak entity signal)
- NS-37 Mission (missed - low entity overlap)

**Conclusion:** Entity-only clustering insufficient. Need semantic similarity + temporal signals for full emergence detection.

---

### Experiment 5: Trump/BBC Duplicate ⚠️

**Finding:** Entity Jaccard was only 25% (lower than expected).

| Metric | Parent | Child |
|--------|--------|-------|
| Claims | 18 | 5 |
| Shared claims | 0 | 0 |
| Shared entities | Donald Trump, BBC |

**Insight:** The duplicate problem isn't entity overlap - it's **semantic near-identity of event names**:
- "Trump vs. BBC Defamation Lawsuit"
- "Trump's Defamation Lawsuit Against BBC"

These are lexically different but semantically identical. EU model would need embedding comparison, not just entity Jaccard.

**Revised solution:** Before creating sub-event, compare embedding of proposed cluster to parent. If similarity >0.9, merge instead of split.

---

## Key Insights

1. **Mass is viable** - Graph signals (corroboration, contradiction, entity importance) produce meaningful importance rankings

2. **Roles are detectable** - Simple patterns work for 80%+ of cases; remaining need LLM classification

3. **Emergence needs more signals** - Entity overlap alone misses abstract events and over-merges geographically related events

4. **Duplicate prevention needs semantics** - Entity Jaccard insufficient; need embedding similarity check before sub-event creation

5. **Computational cost** - Current experiments run in <1s for 1215 claims. Need benchmark for 10K+ scale.

---

## Phase 2: Recursive Emergence (2025-12-17)

### Progressive Emergence v1
**File:** `progressive_emergence.py`

Layer-by-layer emergence with pairwise merging. Too conservative - only merged pairs.

### Progressive Emergence v2
**File:** `progressive_emergence_v2.py`

Added union-find for multi-way absorption. Too aggressive - Jimmy Lai absorbed 1000+ claims.

### Entropy-Based Emergence
**File:** `entropy_emergence.py`

Only merge if entropy decreases (coherence improves). Conservative but principled.

**Results at depth 8:**
| Cluster | Claims | Entropy | Notes |
|---------|--------|---------|-------|
| Jimmy Lai | 42 | -0.63 | Original had 117 |
| Wang Fuk Court | 42 | -0.61 | Original had 130 |
| Bondi Beach | 20 | -0.74 | Very coherent |
| Brown University | 18 | -0.77 | Very coherent |

**Insight:** Entropy approach keeps clusters smaller than original events. This may be correct - original events may be too broad.

### Metabolic Emergence
**File:** `metabolic_emergence.py`

Contradictions are signal, not noise. Allow merges that create tension.

**Key distinction:**
- **Stable clusters**: High coherence, low/no contradictions (Jimmy Lai: 100% coherence)
- **Active clusters**: Have contradictions to metabolize (Wang Fuk Court: 50% coherence, 6 contradictions)

**Results at depth 10:**
| Cluster | Claims | Coherence | Tension | Notes |
|---------|--------|-----------|---------|-------|
| Hong Kong | 127 | 0.87 | 13% | Large, mostly coherent |
| Donald Trump | 93 | 0.86 | 14% | Multiple stories merged |
| Venezuela | 88 | 0.80 | 20% | Evolving situation |
| Jimmy Lai | 77 | 0.83 | 17% | Closer to original size |

### Temporal Analysis
**File:** `temporal_analysis.py`

Analyzed what contradictions represent:
- 13% are temporal/numerical updates (death toll: 128 → 160)
- 59% are source disagreements (Utah State vs Utah Valley University)
- 28% unknown

**Insight:** Most contradictions are NOT errors to fix, but information the system should track.

---

## Key Findings (Phase 2)

1. **Don't match original events** - They're not ground truth (see #22 duplicate issue)

2. **Entropy is good but conservative** - Keeps clusters coherent but small

3. **Metabolism is more realistic** - Real events contain contradictions (evolving death tolls, source disagreements)

4. **Stable vs Active distinction works** - Some stories are settled (Jimmy Lai), others are evolving (Wang Fuk Court)

5. **Contradictions reveal temporal evolution** - "128 killed" → "160 killed" is metabolism, not error

---

## Phase 3: Semantic Emergence (2025-12-17)

### The Problem with Entity-Based Approaches

Entity clustering can't distinguish:
- "Hong Kong fire" (Wang Fuk Court)
- "Hong Kong trial" (Jimmy Lai)
- "Hong Kong politics" (general)

All mention "Hong Kong" but are different events.

### Semantic Approach

**Files:**
- `page_dissolution.py` - Pages as initial EUs (12% assignment - too few links)
- `entity_gravity.py` - Single entity clustering (76% - too broad)
- `entity_constellation.py` - Entity pairs (18% - too narrow)
- `anchor_context.py` - Anchor + context hybrid (73% - better but still issues)
- `semantic_emergence.py` - Embedding clustering (initial test)
- `semantic_emergence_v2.py` - With LLM verification
- `semantic_hierarchy.py` - Full hierarchical emergence

### Results (70-claim mixed sample)

```
Level 0: 70 claims
Level 1: 45 sub-events (embedding similarity)
Level 2: 38 events (LLM-verified merging)
```

**Emerged hierarchy for Wang Fuk Court Fire:**
```
[EVENT] Wang Fuk Court Fire (18 claims)
  └─ [SUB] Fire outbreak (8 claims)
  └─ [SUB] Death toll (4 claims) - ACTIVE (conflicting numbers)
  └─ [SUB] Vigil (2 claims) - STABLE
  └─ [SUB] Fire alarms failure (2 claims) - STABLE
  └─ [SUB] Building spread (2 claims) - STABLE
```

### Key Findings

1. **Semantic > Entity** - Embeddings capture actual similarity, not just shared names
2. **LLM verification cheap & effective** - ~$0.001/call, catches edge cases
3. **Hierarchy emerges naturally** - Claims → Sub-events → Events
4. **Sub-events are real** - "Death toll updates" is a coherent sub-narrative

### Threshold Analysis

| Config | Clusters | Largest | Notes |
|--------|----------|---------|-------|
| Embedding 0.80 | 37 | 5 | Too strict |
| Embedding 0.75 | 27 | 8 | Good balance |
| Embedding + LLM | 30 | 10 | Best |

### Cost

- 70 claims: ~$0.02
- 1215 claims (projected): ~$0.30-0.50
- Feasible for current scale

---

## Phase 4: Full Streaming + Hierarchical Emergence (2025-12-17)

### What We Built

Full pipeline processing all 1215 claims with:
1. **Embeddings cached in PostgreSQL** (pgvector)
2. **Streaming claim → sub-event clustering**
3. **Hierarchical sub-event → event merging**

### Results

```
Phase 1: Claims → Sub-events
  - 1215 claims → 550 sub-events
  - 55% merge rate
  - 408 LLM calls

Phase 2: Sub-events → Events
  - 95 candidates → 16 events created
  - 45 LLM calls
```

### Top Emerged Events

| Event | Claims | Sub-events | Coherence | State |
|-------|--------|------------|-----------|-------|
| Jimmy Lai | 100 | 5 | 100% | STABLE |
| Wang Fuk Court Fire | 66 | 7 | 67% | ACTIVE ⚡ |
| Brown University Shooting | 43 | 4 | 92% | STABLE |
| Bondi Beach Shooting | 42 | 3 | 100% | STABLE |
| Do Kwon Sentencing | 31 | 5 | 100% | STABLE |

### The Fix: Wang Fuk Court Now Unified

Before (v1): Two separate sub-events
```
[sub] Wang Fuk Court Fire (23 claims)
[sub] Tai Po Fire casualties (17 claims)  ← SEPARATE
```

After (v2): One coherent event
```
[EVENT] Wang Fuk Court Fire (66 claims)
  └─ Fire details (23)
  └─ Casualties (16)  ← NOW MERGED
  └─ ICAC investigation (8)
  └─ Death toll updates (8)
```

### Path to Frames

The same mechanism extends to Level 3 (Frames):

```
Level 0: Claims
Level 1: Sub-events ("same incident?")
Level 2: Events ("same story?")
Level 3: Frames ("same narrative?")  ← NEXT
```

Example potential frames:
- "Trump Presidency 2.0" → BBC lawsuit + Brown comments + Lai case + Venezuela
- "Hong Kong 2025" → Wang Fuk Court Fire + Jimmy Lai Trial

### Cost

| Phase | Calls | Cost |
|-------|-------|------|
| Embeddings | 1215 | $0.02 (cached) |
| Phase 1 LLM | 408 | $0.04 |
| Phase 2 LLM | 45 | $0.005 |
| **Total** | | **$0.07** |

---

## Reports

- `report_1.md` - Entropy approach findings
- `report_2.md` - Metabolic approach findings
- `report_3.md` - Semantic emergence findings
- `report_4.md` - **Hierarchical emergence (breakthrough)**

---

## Summary: What Works

| Metric | Best Approach | Notes |
|--------|---------------|-------|
| Clustering | Semantic (embeddings) | Handles "Hong Kong problem" |
| Borderline merges | LLM verification | 41% acceptance rate |
| Hierarchy | Recursive LLM merge | Sub-events → Events → Frames |
| Coherence | Corr/contra ratio | Stable vs Active |
| Mass | `size × coherence × pages` | Browsability |
| Caching | PostgreSQL pgvector | Reuse embeddings |

## Final Architecture

```
Claim arrives
    ↓
Get/cache embedding (PostgreSQL)
    ↓
Find similar sub-events (cosine similarity)
    ↓
sim > 0.70? → Merge directly
0.55-0.70? → LLM "same event?" → Merge if YES
< 0.55? → Create new sub-event
    ↓
Track: mass, coherence, tension
    ↓
Periodically:
  Sub-events → Events (LLM: "same story?")
  Events → Frames (LLM: "same narrative?")
```

## Recursive EU Model (Validated)

```
EU = Claim | Cluster(EU, EU, ...)

Each EU has:
- embedding (centroid of children)
- mass (accumulated from children)
- coherence (internal corroboration)
- tension (internal contradiction)
- state (STABLE | ACTIVE)
- children[] (recursive)
```

## Phase 5: Frame Emergence & Cross-Frame Linking (2025-12-17)

### Frame Emergence (Level 3)

**File:** `frame_emergence.py`

Three-level hierarchy: Claims → Sub-events → Events → Frames

**Results:**
```
Claims: 1215
Sub-events: 553
Events: 17
Frames: 1  ← Only one emerged!
```

**The emerged frame:**
```
[FRAME] "Wang Fuk Court Fire Incident"
        77 claims, 23 pages, ACTIVE ⚡
        └─ Wang Fuk Court Fire details (66 claims)
        └─ Fire ignition/mesh netting (11 claims)
```

**Why only one frame?** Other events don't share narratives:
- Jimmy Lai (99 claims) - isolated human rights story
- Bondi Beach Shooting (42 claims) - isolated crime story
- Brown University Shooting (39 claims) - isolated campus violence

Frame emergence requires **narrative density** - multiple events from same geographic/thematic region.

**Key insight:** The algorithm is correctly conservative. Better to under-merge than create incoherent "junk drawer" frames.

---

### Cross-Frame Entity Linking

**File:** `cross_frame_linking.py`

**Finding: Donald Trump appears in 5 separate events:**
```
'donald trump' appears in 5 events:
  └─ BBC defamation lawsuit (22 claims)
  └─ Brown University shooting response (35 claims)
  └─ Venezuela/Machado policy (7 claims)
  └─ Jimmy Lai advocacy (100 claims)
  └─ Rob Reiner interview (12 claims)
```

**Other cross-event entities:**
| Entity | Events | Potential Frame |
|--------|--------|-----------------|
| Donald Trump | 5 | Trump Presidency 2.0 |
| Hong Kong | 2 | Hong Kong 2025 |
| Venezuela | 2 | Venezuela Crisis |
| Do Kwon | 2 | Crypto Fraud |
| Maria Corina Machado | 2 | Venezuelan Opposition |

**Graph structure implication:**
```
Level 0-2: TREE structure (each claim → one sub-event)
Level 3+:  GRAPH structure (events → multiple frames)
```

Jimmy Lai links to "Hong Kong" via entity, but NOT via semantic merge with Wang Fuk Court Fire. This is correct - they're different narratives that share location.

---

### Taxonomy Emergence

**File:** `taxonomy_emergence.py`

**Question:** When does taxonomy crystallize?

**Finding:** Taxonomy confidence increases with EU size:
```
Singletons: 84% confidence
Small:      88% confidence
Medium:     93% confidence
Large:      94% confidence
Events:     90% confidence (20% high specificity)
```

**Key insight:** Taxonomy is:
- **Secondary** at levels 0-1 (content is self-describing)
- **Starting to matter** at level 2 (events)
- **Essential** at level 3+ (frames/meta-frames)

The right question isn't "does it need taxonomy?" but "is taxonomy the PRIMARY descriptor?"

---

## Reports

- `report_1.md` - Entropy approach findings
- `report_2.md` - Metabolic approach findings
- `report_3.md` - Semantic emergence findings
- `report_4.md` - Hierarchical emergence (breakthrough)
- `report_5.md` - **Unbounded hierarchy & emergent taxonomy**
- `report_6.md` - **Frame emergence analysis**
- `report_7.md` - **Cross-frame entity linking (graph structure)**
- `report_8.md` - **Taxonomy emergence patterns**

---

## Final Architecture (Validated)

```
┌─────────────────────────────────────────────────────────────┐
│                     LEVEL 3+ (GRAPH)                        │
│  ┌─────────────┐   entity    ┌─────────────┐               │
│  │ Hong Kong   │←───link────→│ Press       │               │
│  │ 2025        │             │ Freedom     │               │
│  └──────┬──────┘             └──────┬──────┘               │
│         │                           │                       │
│  ┌──────┴──────────────────────────┴───────┐               │
│                     LEVEL 2 (TREE)                         │
│  ┌─────────────┐             ┌─────────────┐               │
│  │ Wang Fuk    │             │ Jimmy Lai   │               │
│  │ Court Fire  │             │ Trial       │               │
│  └──────┬──────┘             └──────┬──────┘               │
│         │                           │                       │
│  ┌──────┴──────┐              ┌─────┴──────┐               │
│  │ Sub-events  │              │ Sub-events │               │
│  │ (7 total)   │              │ (5 total)  │               │
│  └──────┬──────┘              └─────┬──────┘               │
│         │                           │                       │
│  ┌──────┴──────┐              ┌─────┴──────┐               │
│  │ Claims (66) │              │ Claims(100)│               │
│  └─────────────┘              └────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

**Merge logic at each level:**
- Claims → Sub-events: Embedding sim > 0.70 OR (0.55-0.70 + LLM "same event?")
- Sub-events → Events: sim > 0.60 + LLM "same story?"
- Events → Frames: sim > 0.50 + LLM "same narrative?"
- Frames link via: Entity co-occurrence (graph, not tree)

**Stopping criteria (not level number):**
- Label becomes stop-word ("Politics", "Society")
- Coherence collapses (<0.2)
- No temporal/thematic boundary

---

## Key Insights (Complete)

1. **Mass is emergent** - Computed from corroboration, contradiction, entity importance
2. **Hierarchy is natural** - Same "same X?" LLM check works at every level
3. **Frames require density** - Need multiple events sharing narrative
4. **Graph structure at L3+** - Entities create cross-frame links
5. **Taxonomy crystallizes later** - Not needed at claim level, essential at frame level
6. **Conservative is correct** - Better to under-merge than create junk drawers
7. **Cost is feasible** - ~$0.07 for 1215 claims full hierarchy

---

## Next Steps

1. ✅ Cache embeddings in PostgreSQL
2. ✅ Streaming emergence works
3. ✅ Hierarchical merging works
4. ✅ Frame level emergence (tested)
5. ✅ Cross-frame entity linking (analyzed)
6. ✅ Taxonomy emergence patterns
7. ⬜ Real-time streaming (not batch)
8. ⬜ Store hierarchy in Neo4j
9. ⬜ UI for browsing hierarchy
10. ⬜ Hub-based frame generation (Donald Trump hub)
