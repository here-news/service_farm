# Narrative Structure Improvement Notes

## Current State (Dec 2025)

### What Works Well

1. **Claim Reference Embedding** - `[cl_xxx]` markers inline with statements
2. **Entity ID Marking** - `[en_xxx]` on first mention of entities
3. **Plausibility-Based Ordering** - High-plausibility claims treated as facts
4. **Update Chain Detection** - Superseded claims identified and deprioritized
5. **Pattern Detection** - Progressive/contradictory/consensus classification
6. **Markdown Headers** - `**Section**` for topic breaks

### Current Narrative Flow
```
**Overview** → **Casualties** → **Community Response** → **Investigation** → **Government Reactions** → **Conclusion**
```

---

## Key Insights from This Session

### 1. Topology Drives Narrative Quality
The Bayesian topology analysis is the foundation. Better topology = better narrative.

- **Update chains** must be detected with direction (which claim is newer)
- **Superseded claims** need explicit penalty, not just lower corroboration
- **Progressive pattern** requires special handling - use latest figures, reference earlier as "initial reports"

### 2. Prompt Engineering is Fragile
Ad-hoc prompt instructions (like "for progressive events, do X") are brittle. Instead:
- Pre-compute the structure in code
- Pass explicit data (superseded_by map, sorted claims)
- Let LLM synthesize, not decide structure

### 3. Entity Hydration is Critical
Without `canonical_name` on entities, narratives lose the `[en_xxx]` markers.
- Must hydrate entities when loading claims
- Entity lookup should come from hydrated Entity objects, not metadata

### 4. Plausibility Normalization Matters
Raw posteriors vary wildly. Normalization (log-scale sigmoid) keeps them in useful range (0.1-0.95).

---

## Ideas for Next Session

### A. Structured Narrative Schema

Instead of free-form LLM generation, define a schema:

```python
@dataclass
class NarrativeSection:
    title: str  # e.g., "Casualties"
    topic_key: str  # e.g., "casualties", "response", "investigation"
    claims: List[Claim]  # Claims relevant to this section
    summary: str  # Generated summary for this section

@dataclass
class StructuredNarrative:
    headline: str
    sections: List[NarrativeSection]
    timeline: List[TimelineEntry]  # Key events in chronological order
    key_figures: Dict[str, str]  # "death_toll" -> "160", "injuries" -> "76"
```

**Benefits:**
- Frontend can render sections independently
- Easier to update individual sections when new claims arrive
- Key figures extracted explicitly (not buried in prose)

### B. Topic Clustering Before Narrative

Cluster claims by topic before generating narrative:
1. Extract topic from each claim (casualties, response, investigation, etc.)
2. Group claims by topic
3. Generate section-by-section
4. Assemble into coherent narrative

Could use claim embeddings + clustering, or LLM classification.

### C. Temporal Layering

For progressive events, show the evolution:
```
Initial Reports (Nov 26 AM): 36 dead, 279 missing
Updated (Nov 26 PM): 83 dead confirmed
Current (Dec 9): 160 confirmed dead after DNA tests
```

This could be a separate "Timeline" section or inline annotations.

### D. Contradiction Highlighting

When contradictions exist, make them explicit:
```
⚠️ Conflicting Reports:
- Source A claims X [cl_xxx]
- Source B claims Y [cl_yyy]
```

Currently contradictions are in `TopologyResult` but not well-surfaced in narrative.

### E. Source Attribution

Add source credibility to narrative:
```
"According to official government statements [high confidence], the death toll..."
"Local media reports [medium confidence] suggest..."
```

We have `publisher_priors` - could surface this in narrative.

### F. Claim Grouping by Semantic Similarity

Instead of listing similar claims separately, group them:
```
Multiple sources confirm the death toll reached 160 [cl_a][cl_b][cl_c].
```

Currently done somewhat, but could be more systematic using the similarity network.

---

## Technical Debt to Address

1. **Duplicate SUPPORTS relationships** - Some claims have multiple SUPPORTS edges to same event (seen in query results with duplicate rows)

2. **DateTime handling** - Neo4j DateTime vs Python datetime causes issues. Should standardize on UTC-aware Python datetime everywhere.

3. **Two narrative generation paths** - `_generate_event_narrative` vs `_generate_event_narrative_with_topology`. Should consolidate.

4. **Claim.entity_names from metadata** - Falls back to empty list if entities not hydrated. Should fail loudly or always hydrate.

---

## Files to Review

- `backend/services/claim_topology.py` - Bayesian analysis, update detection
- `backend/services/event_service.py` - Narrative generation (`_generate_event_narrative_with_topology`)
- `backend/models/domain/live_event.py` - Command handlers, hydration
- `backend/models/domain/claim.py` - `entity_names` property

---

## Test Event

`ev_pth3a8dc` - Wang Fuk Court Fire
- 89 claims
- Progressive pattern (death toll evolved from 36 → 83 → 156 → 160)
- Multiple entities (John Lee, Xi Jinping, locations)
- Good test case for all narrative features

Command to re-test:
```bash
docker exec herenews-app python send_event_command.py ev_pth3a8dc /retopologize
```

---

## Questions to Explore

1. Should narrative structure be event-type specific? (Disaster vs. Political vs. Crime)
2. How to handle very long events with 200+ claims? Summarization? Sub-events?
3. Should we store generated narrative sections separately for incremental updates?
4. How to surface uncertainty to end users without cluttering the narrative?
