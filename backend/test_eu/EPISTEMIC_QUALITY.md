# Complete Epistemic Quality Framework

## The Goal

Given N claims about an event, produce:
1. A **coherent** picture of what happened
2. With **calibrated** confidence (Jaynes)
3. That is **complete** (covers all aspects)
4. And **traceable** (every belief → claims)

## Quality Dimensions

### 1. COMPLETENESS (Event Coverage)

Does the kernel answer the fundamental questions?

| Aspect | Question | Required for Complete Picture |
|--------|----------|------------------------------|
| WHAT | What happened? | ✓ Essential |
| WHEN | When did it happen? | ✓ Essential |
| WHERE | Where did it happen? | ✓ Essential |
| WHO_AFFECTED | Who was affected? | ✓ Essential |
| WHO_ACTORS | Who responded/caused it? | ○ Important |
| OUTCOMES | What were the consequences? | ✓ Essential |
| CAUSE | Why did it happen? | ○ Important |
| RESPONSE | How did authorities respond? | ○ Important |
| ACCOUNTABILITY | Who is responsible? | ○ If applicable |
| UNRESOLVED | What remains unknown? | ✓ Essential |

**Metric**: % of essential aspects covered with at least one corroborated belief

### 2. COHERENCE (Internal Consistency)

Are the beliefs mutually consistent?

| Check | Description | Failure Mode |
|-------|-------------|--------------|
| No contradictions | Beliefs don't conflict | "Fire on floor 3" AND "Fire on floor 8" |
| Temporal consistency | Timeline makes sense | Death toll decreasing over time |
| Logical consistency | Beliefs entail each other | "160 dead" but "only 40 injured" |
| Entity consistency | Same entities referenced | "John Lee" = "Hong Kong leader" |

**Metric**: Number of unresolved contradictions / Total beliefs

### 3. CONFIDENCE CALIBRATION (Jaynes)

Is uncertainty proportional to evidence?

| Sources | Expected Confidence | Expected Entropy |
|---------|--------------------|--------------------|
| 0 | 0.0 | 1.0 (unknown) |
| 1 | 0.15-0.25 | 0.8 (unconfirmed) |
| 2 | 0.25-0.35 | 0.5 (corroborated) |
| 3+ | 0.35-0.50 | 0.1-0.4 (confirmed) |

**Metric**: Correlation between source count and confidence

### 4. CONSOLIDATION (Semantic Deduplication)

Are semantically equivalent claims merged?

| Input Claims | Should Become | Current Behavior |
|--------------|---------------|------------------|
| "160 dead", "death toll 160", "160 killed" | ONE belief: death_toll=160 | 3 separate topics |
| "Wang Fuk Court", "Tai Po building" | ONE belief: location=Wang Fuk Court, Tai Po | 2 topics |

**Metric**: Number of topics / Expected canonical topics (lower is better)

### 5. VALUE EXTRACTION (Structured Data)

Are values properly extracted from claims?

| Claim | Expected Extraction |
|-------|---------------------|
| "At least 160 people died" | {topic: death_toll, value: 160, qualifier: "at least"} |
| "Fire broke out at 3:30am" | {topic: time, value: "03:30", date: inferred} |
| "Police arrested 15 people" | {topic: arrests, value: 15, actor: "police"} |

**Metric**: % of claims with structured value extraction

### 6. TEMPORAL EVOLUTION (Update Handling)

Are temporal updates handled correctly?

| Claim Sequence | Expected Behavior |
|----------------|-------------------|
| "4 dead" → "13 dead" → "160 dead" | TEMPORAL_UPDATE, not CONTRADICT |
| "Fire ongoing" → "Fire contained" | STATUS_UPDATE, not CONTRADICT |
| "3 arrested" → "15 arrested" | TEMPORAL_UPDATE (count increased) |

**Metric**: % of numeric updates correctly classified as updates (not contradictions)

### 7. TRACEABILITY (Provenance)

Can every belief be traced to source claims?

| Requirement | Check |
|-------------|-------|
| Every belief has ≥1 supporting claim | belief.supporting_claims not empty |
| Every claim is placed or excluded | claim.status != "pending" |
| Every action has justification | action.rule and action.reasoning |

**Metric**: % of beliefs with complete provenance chain

## Composite Quality Score

```
Quality = w1*Completeness + w2*Coherence + w3*Calibration +
          w4*Consolidation + w5*Extraction + w6*Temporal + w7*Traceability

Where:
  w1 = 0.20 (Completeness)
  w2 = 0.20 (Coherence - no contradictions)
  w3 = 0.15 (Calibration - Jaynes)
  w4 = 0.15 (Consolidation)
  w5 = 0.10 (Value extraction)
  w6 = 0.10 (Temporal handling)
  w7 = 0.10 (Traceability)
```

## Current Kernel Assessment

Based on the Hong Kong Fire test (73 claims):

| Dimension | Score | Issue |
|-----------|-------|-------|
| Completeness | 40% | Missing WHEN, UNRESOLVED |
| Coherence | 70% | 5 contradictions (death_toll, etc) |
| Calibration | 95% | ✓ Jaynes aligned |
| Consolidation | 20% | 47 topics instead of ~10 |
| Extraction | 30% | Full text stored, not values |
| Temporal | 40% | Updates marked as contradictions |
| Traceability | 90% | ✓ Claims linked to beliefs |

**Composite Score: ~50%** (needs improvement)

## Required Enhancements

### Priority 1: Canonical Topic Mapping
```python
# Instead of LLM generating arbitrary topics, map to canonical set
CANONICAL_TOPICS = {
    'outcomes': ['death_toll', 'injury_count', 'missing_count', 'damage'],
    'where': ['location', 'address', 'district'],
    'when': ['date', 'time', 'duration'],
    ...
}
```

### Priority 2: Value Normalization
```python
# Extract structured values from claims
async def extract_value(claim_text, topic):
    if topic in ['death_toll', 'injury_count', 'arrests']:
        # Extract number
        return extract_number(claim_text)
    elif topic in ['location', 'address']:
        # Extract place name
        return extract_location(claim_text)
    ...
```

### Priority 3: Temporal Update Detection
```python
# Recognize updates vs contradictions for numeric topics
if is_numeric_topic(topic) and new_value > old_value:
    return "updates", Rule.TEMPORAL_UPDATE
```

### Priority 4: Semantic Consolidation
```python
# Before creating new topic, check if semantically equivalent exists
existing_topic = find_equivalent_topic(new_topic_name, aspect)
if existing_topic:
    return existing_topic  # Merge, don't create new
```

## Success Criteria

A "complete" epistemic summary should:

1. Answer: **What happened?** (the incident)
2. Answer: **When?** (date, time)
3. Answer: **Where?** (location)
4. Answer: **Who was affected?** (casualties, displaced)
5. Answer: **What's the current status?** (latest death toll, investigation)
6. Acknowledge: **What's uncertain?** (cause, responsibility)
7. Acknowledge: **What's unknown?** (still missing info)

All with appropriate confidence hedging based on source count.
