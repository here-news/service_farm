# Belief Kernel Upgrade Layers

## Status: Kernel Core is SOLID

The 5-relationship model works well:
- **COMPATIBLE** - New fact, add to beliefs
- **REDUNDANT** - Already known, add source
- **REFINES** - More specific, replace
- **SUPERSEDES** - Updated info, replace with history
- **CONFLICTS** - Cannot resolve, flag for review

RICH mode (temporal + modality context) improves results significantly:
- 50% fewer conflicts (2 vs 4)
- +24% quality score
- Better temporal update detection

**The kernel stays untouched. Upgrades are to layers above it.**

---

## Layer 3: Topology Storage (UPGRADE)

### Current State
```python
belief = {
    'text': '160 people died',
    'sources': ['bbc.com', 'reuters.com'],
    'supersedes': '128 people died'  # plain text
}
```

### Upgraded State
```python
belief = {
    'id': 'bl_abc123',              # belief ID for citations
    'text': '160 people died',
    'sources': ['bbc.com', 'reuters.com'],
    'claim_ids': ['cl_xyz', 'cl_abc'],  # provenance to original claims
    'entity_ids': ['en_tsrl5p2z'],      # entities mentioned
    'supersedes_id': 'bl_def456',       # belief ID chain (not text)
    'supersedes_text': '128 people died',  # keep for display
    'certainty': 0.88,                  # computed from source count
    'category': 'casualties',           # thematic grouping
    'last_updated': '2025-12-11T09:00:00Z'
}
```

### Implementation
- Generate belief IDs when creating beliefs
- Track claim IDs that contributed to each belief
- Extract entity IDs from source claims
- Compute certainty from source count: `min(source_count / 4, 1.0)`
- Auto-categorize beliefs into themes

---

## Layer 4: Prose Generation (UPGRADE)

### Current Prompt (simple)
```
CONFIRMED (3+ sources):
- 160 people died
- Fire alarm failed

Write news summary...
```

### Upgraded Prompt (webapp-style)
```
Synthesize beliefs into narrative with embedded references.

BELIEFS (grouped by theme):

## Casualties [high certainty]
[bl_001] 160 people died in the Wang Fuk Court fire (4 sources)
[bl_002] 76 people were injured (3 sources)
[bl_003] 30 people are missing (2 sources)

## Emergency Response [high certainty]
[bl_004] 200 fire trucks deployed (3 sources)
[bl_005] Fire alarm systems were not functioning (4 sources)

## Investigation [medium certainty]
[bl_006] 15 people arrested for manslaughter (2 sources)

## Unresolved
[conflict] Fire origin: 15th floor vs 14th floor

ENTITIES:
Wang Fuk Court → en_tsrl5p2z
Hong Kong → en_lo521i3d
John Lee → en_tq6x7efn

Write narrative that:
1. Marks entities on first mention: "Wang Fuk Court [en_tsrl5p2z]"
2. Cites beliefs: "killed 160 people [bl_001]"
3. Uses thematic sections with headers
4. Notes certainty: "confirmed" (3+), "reported" (2), "according to one source" (1)
5. Addresses conflicts in a "Developing" section
```

---

## Layer 5: UI Presentation (NEW)

### Based on event-evolution.html prototype

#### 5.1 Fact Highlighting in Prose
```html
<span class="fact-highlight certain" data-belief-id="bl_001">
    160 people
    <span class="fact-tooltip">
        <div>Certainty: 88%</div>
        <div>Sources: 4 independent</div>
        <div>History: 36 → 128 → 156 → 160</div>
    </span>
</span>
```

CSS classes based on source count:
- `.certain` (green) - 3+ sources
- `.likely` (blue) - 2 sources
- `.uncertain` (yellow) - 1 source
- `.contested` (red) - conflicting

#### 5.2 Belief States Panel
```javascript
const beliefStates = topology.beliefs
    .filter(b => isNumericMetric(b.text))
    .map(b => ({
        label: extractMetricLabel(b.text),  // "Deaths", "Injured"
        value: extractNumber(b.text),        // 160
        certainty: b.certainty,
        history: buildHistory(b),            // from supersedes chain
        sources: b.sources.length
    }));
```

#### 5.3 Active Contradictions Panel
```javascript
const contradictions = topology.conflicts.map(c => ({
    topic: inferTopic(c),
    values: [c.new_claim, c.existing_belief],
    support: [1, countSources(c.existing_belief)],
    status: 'unresolved'
}));
```

#### 5.4 Convergence Chart
Track entropy at each claim:
```javascript
const trajectory = topology.history.map((h, i) => ({
    claim: i + 1,
    entropy: computeEntropyAtPoint(topology, i),
    beliefCount: countBeliefsAtPoint(topology, i)
}));
```

---

## Implementation Order

### Phase 1: Topology Enrichment
1. Add belief ID generation
2. Track claim IDs per belief
3. Extract entity IDs from claims
4. Compute certainty scores
5. Auto-categorize beliefs

### Phase 2: Prose Upgrade
1. Create thematic grouping function
2. Build entity lookup from claims
3. Upgrade prompt with citations
4. Post-process to add entity IDs

### Phase 3: API Layer
1. Create `/api/kernel/events` endpoint
2. Return full topology with UI-ready data
3. Include entropy trajectory
4. Include conflict details

### Phase 4: UI Integration
1. Adapt event-evolution.html to consume API
2. Implement fact highlighting
3. Implement belief states panel
4. Implement convergence chart

---

## File Structure

```
backend/test_eu/
├── belief_kernel.py          # UNCHANGED - core engine (5 relationships)
├── kernel_enriched.py        # Layer 3 - EnrichedKernel wrapper
│   └── Adds: belief IDs, claim tracking, entity IDs, certainty, categories
├── kernel_prose.py           # Layer 4 - citation-rich prose generation
│   └── Adds: [bl_xxx] citations, [en_xxx] entity links, thematic sections
├── kernel_api.py             # Layer 5 - FastAPI endpoints
│   └── Endpoints: /process, /topology, /prose, /trajectory
└── KERNEL_UPGRADES.md        # This document

frontend/prototypes/
├── event-evolution.html      # UI patterns reference
└── kernel-ui.html            # Kernel-connected UI prototype
    └── Features: topology viewer, convergence chart, belief browser
```

---

## Implementation Status

| Layer | File | Status |
|-------|------|--------|
| 1. Kernel Core | belief_kernel.py | STABLE (unchanged) |
| 2. Claim Enrichment | test_kernel_full.py | STABLE (RICH mode) |
| 3. Topology Storage | kernel_enriched.py | COMPLETE |
| 4. Prose Generation | kernel_prose.py | COMPLETE |
| 5. API Layer | kernel_api.py | COMPLETE |
| 6. UI Prototype | kernel-ui.html | COMPLETE |
