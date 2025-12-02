# Section-Aware Event Architecture

**Date:** 2025-12-02
**Status:** Design → Implementation

---

## Core Principle

**Events grow through sections that can be promoted to separate events**

```
Event: Hong Kong Fire
├─ Section: "Outbreak" (promotion_score: 0.3)
│  ├─ casualties: {deaths: 44, ...}
│  ├─ timeline: {start: ..., milestones: [...]}
│  └─ 3 pages
├─ Section: "Rescue Operations" (promotion_score: 0.45)
│  ├─ response: {evacuations: 700, ...}
│  └─ 2 pages
└─ Section: "Investigation" (promotion_score: 0.78) ⚠️ PROMOTABLE!
   ├─ arrests: 3
   └─ 2 pages → Should become separate related event
```

---

## Ontology Structure

```python
event = {
    "event_id": "uuid",
    "sections": {
        "main": {
            # Always present - the core event narrative
            "name": "Main Event",
            "semantic_type": "primary",
            "start": FactBelief(...),
            "end": FactBelief(...),
            "summary": "2-3 sentence narrative",
            "ontology": {
                "casualties": {...},
                "timeline": {...},
                "locations": {...},
                "response": {...}
            },
            "page_ids": ["uuid1", "uuid2"],
            "page_count": 5,
            "promotion_score": 0.0,  # Main section never promotes
            "created_at": "...",
            "updated_at": "..."
        },

        "investigation_phase": {
            # Detected section - grows independently
            "name": "Criminal Investigation",
            "semantic_type": "investigation",  # LLM-detected
            "start": FactBelief("2025-11-27T09:00", ...),
            "end": None,
            "summary": "Police investigation into fire cause",
            "ontology": {
                "arrests": FactBelief(3, ...),
                "suspects": [...],
                "timeline": {...}
            },
            "page_ids": ["uuid6", "uuid7"],
            "page_count": 2,
            "promotion_score": 0.78,  # HIGH - promotable!
            "promotion_signals": {
                "temporal_gap": 0.6,      # 18+ hours from main event end
                "entity_divergence": 0.8,  # Different primary entities
                "semantic_shift": 0.9,     # LLM: "fire" → "investigation"
                "page_density": 0.7,       # 2 pages exclusively about this
                "human_weight": 0.0        # Not yet reviewed
            },
            "created_at": "...",
            "updated_at": "..."
        }
    },

    "coherence": 0.95,
    "artifact_count": 7,
    "enrichment_timeline": [...]
}
```

---

## Section Detection Logic

### When Processing Page → Event

```python
1. LLM analyzes page content against existing event sections:

   Prompt:
   """
   Event has these sections:
   - Main Event: Hong Kong fire at Wang Fuk Court (Nov 26, 2025)
   - Rescue Operations: Evacuation and firefighting

   New page: "Hong Kong Police Arrest 3 in Connection with Fire"

   Does this page:
   A) Enrich existing section (which one?)
   B) Create new section (what name? what semantic type?)

   Return:
   {
     "decision": "create_section",
     "section_name": "Criminal Investigation",
     "semantic_type": "investigation",
     "rationale": "Page focuses on criminal investigation, distinct from fire/rescue"
   }
   """

2. If create_section:
   - Create new section key (sanitized name)
   - Initialize ontology for that section
   - Set promotion_score = 0.5 (new sections start moderate)

3. If enrich_section:
   - Route to appropriate section
   - Update that section's ontology
   - Recalculate promotion_score
```

---

## Promotion Scoring

### Formula

```python
promotion_score = (
    temporal_gap * 0.25 +
    entity_divergence * 0.25 +
    semantic_shift * 0.20 +
    page_density * 0.20 +
    human_weight * 0.10
)
```

### Signals

1. **Temporal Gap** (0.0-1.0):
   - < 6 hours: 0.0
   - 6-24 hours: 0.3
   - 24-48 hours: 0.6
   - 48+ hours: 0.9

2. **Entity Divergence** (0.0-1.0):
   - Jaccard distance between section entities and main section entities
   - < 20% overlap: 0.9
   - 20-50% overlap: 0.6
   - 50-80% overlap: 0.3
   - > 80% overlap: 0.1

3. **Semantic Shift** (0.0-1.0):
   - LLM classification: "Same topic" (0.0-0.3), "Related topic" (0.4-0.7), "New topic" (0.8-1.0)
   - Example: "fire" → "fire rescue" = 0.2
   - Example: "fire" → "investigation" = 0.8

4. **Page Density** (0.0-1.0):
   - Ratio of pages exclusively about this section vs main event
   - 1 page: 0.2
   - 2-3 pages: 0.5
   - 4-5 pages: 0.7
   - 6+ pages: 0.9

5. **Human Weight** (-0.3 to +0.3):
   - Manual override: "Keep together" (-0.3), "Neutral" (0.0), "Separate" (+0.3)
   - Allows human to influence but not fully control

### Thresholds

- **< 0.45**: Section stays within event (STABLE)
- **0.45-0.59**: "Review zone" - UI shows warning icon (REVIEW)
- **≥ 0.6**: Promotable - UI shows "Promote to Event" button (PROMOTABLE)

Note: Thresholds are intentionally permissive to tolerate missing data and encourage natural section evolution.

---

## Promotion Action

When section promoted:

```python
1. Create new event from section:
   - Title: section['name']
   - Ontology: section['ontology']
   - Pages: section['page_ids']
   - Coherence: recalculate

2. Create relationship:
   - Type: PHASE_OF if temporal continuation
   - Type: CAUSED_BY if causal
   - Type: EXTENDS if elaboration
   - Confidence: 1.0 - promotion_score (inverse)

3. Remove section from original event:
   - Keep reference in event['promoted_sections']
   - Update event coherence

4. Link pages to new event:
   - page_events table gets new entries
```

---

## Section Naming Strategy

### Automatic Detection

LLM identifies semantic types:
- `outbreak` - initial incident
- `response` - emergency response, rescue, evacuation
- `investigation` - criminal/official investigation
- `aftermath` - long-term consequences, rebuilding
- `legal_action` - lawsuits, charges, trials
- `policy_response` - government policy changes
- `memorial` - commemorations, remembrance

### Name Format

- **Semantic type + Context**: "Criminal Investigation", "Government Response", "Rescue Operations"
- Not generic: Not "Phase 1", "Section B"
- Not temporal: Not "November 27 Events" (unless truly time-boxed)

---

## Web UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│ Event: Hong Kong Fire at Wang Fuk Court                    │
├──────────┬──────────────────────────────┬──────────────────┤
│          │                              │                  │
│ SECTIONS │      MAIN CONTENT            │  ARTIFACT        │
│          │                              │  SUBMISSION      │
│          │                              │                  │
│ ▼ Main   │  [Section Content Display]  │  ┌────────────┐  │
│   Event  │                              │  │ Paste URL  │  │
│   ────── │  Casualties:                 │  │            │  │
│   5 pgs  │   Deaths: 44 (4 sources)     │  └────────────┘  │
│   0.95   │   Missing: 279               │                  │
│          │                              │  ┌────────────┐  │
│ ▼ Rescue │  Timeline:                   │  │  [Submit]  │  │
│   Ops    │   • 14:51 - Fire starts      │  └────────────┘  │
│   ────── │   • 18:22 - Level 5 alarm    │                  │
│   2 pgs  │                              │  Recent:         │
│   0.45   │  [Story Narrative]           │  • LiveNow       │
│          │                              │  • NYPost        │
│ ⚠ Invest │  [Entity Graph]              │  • Newsweek      │
│   igation│                              │                  │
│   ────── │                              │  [Promote?]      │
│   2 pgs  │                              │  ┌────────────┐  │
│   0.78   │                              │  │ Yes | No   │  │
│          │                              │  └────────────┘  │
└──────────┴──────────────────────────────┴──────────────────┘
```

**Section List (Left)**:
- Collapsible sections
- Show page count and coherence
- Color code by promotion score:
  - Green (< 0.5): Stable
  - Yellow (0.5-0.69): Review
  - Red (≥ 0.7): Promotable
- Click to view in main content

**Main Content (Center)**:
- Selected section's ontology
- Story narrative
- Entity graph
- Timeline visualization

**Artifact Submission (Right)**:
- URL paste box
- Submit button
- Recent artifacts list
- Promotion decision UI (if section in review zone)

---

## Implementation Plan

1. **Update holistic_enrichment.py**:
   - Add section detection prompt
   - Route page enrichment to specific section
   - Calculate promotion scores

2. **Update test_event_network.py**:
   - Load existing Hong Kong fire event
   - Reprocess pages with section awareness
   - Show section growth progression

3. **Create frontend/section_ui.html**:
   - Three-column layout
   - Section list with promotion indicators
   - Main content display
   - Artifact submission form

4. **Add promotion endpoint**:
   - POST /api/events/{event_id}/sections/{section_key}/promote
   - Creates new event, updates relationships
   - Returns new event ID

---

## Test Scenario: Hong Kong Fire

**Expected sections after processing 5 pages:**

1. **Main Event** (5 pages):
   - All factual details about fire
   - Casualties, timeline, locations
   - Promotion score: 0.0 (never promotes)

2. **Rescue Operations** (2-3 pages mention):
   - Evacuation details
   - Firefighter deployment
   - Promotion score: 0.45 (stable)

3. **Investigation** (if enough detail):
   - 3 arrests
   - Bamboo scaffolding as factor
   - Promotion score: 0.65-0.75 (review/promotable)

Let's test and see what naturally emerges!
