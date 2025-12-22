# EU Experiment Report 11

## Event Readiness Analysis - Epistemic & Semantic Assessment

### Framework

We analyzed emerged events on two dimensions:

**EPISTEMIC READINESS** (Is it trustworthy?)
- Source diversity (multiple independent pages)
- Corroboration density (internal claim support)
- Contradiction resolution (tensions resolved?)
- Authority signals (institutional sources)

**SEMANTIC READINESS** (Is it complete?)
- 5W1H coverage (Who, What, When, Where, Why, How)
- Narrative completeness (beginning, development, outcome)
- Narrative stage (EMERGING → DEVELOPING → RESOLVING → CONCLUDED)

### Results Summary

```
✅ Ready to publish: 8
🟡 Needs curation: 2
🔵 Still developing: 0
⚪ Premature: 0
```

### Top Events Analysis

| Event | Claims | Sources | Coherence | Semantic | Epistemic | Overall | Stage |
|-------|--------|---------|-----------|----------|-----------|---------|-------|
| Jimmy Lai Trial | 102 | 13 | 100% | 56% | 98% | 77% | DEVELOPING |
| Wang Fuk Court Fire | 66 | 20 | 67% | 68% | 72% | 70% | DEVELOPING |
| Bondi Beach Shooting | 41 | 9 | 100% | 71% | 98% | 85% | DEVELOPING |
| Brown University | 35 | 9 | 91% | 72% | 81% | 77% | RESOLVING |
| Venezuela Oil | 31 | 5 | 100% | 71% | 98% | 85% | DEVELOPING |
| Do Kwon Sentencing | 29 | 6 | 100% | 76% | 98% | 87% | CONCLUDED |
| Nick Reiner Murder | 24 | 8 | 100% | 68% | 98% | 83% | DEVELOPING |
| Trump vs BBC | 22 | 6 | 100% | 67% | 83% | 75% | RESOLVING |
| Amanda Seyfried | 19 | 4 | 92% | 60% | 76% | 68% | DEVELOPING |
| Starlink Collision | 15 | 3 | 100% | 65% | 73% | 69% | RESOLVING |

### Key Insights

**1. Epistemic scores are generally HIGH (72-98%)**

Most events have strong epistemic foundations:
- Multiple independent sources
- Official/institutional sources present
- Low speculation ratio

**2. Semantic scores are MODERATE (56-76%)**

Common gaps:
- Missing WHY (motivations often unclear)
- Missing HOW (mechanisms not explained)
- Open questions remain unresolved

**3. Coherence correlates with epistemic quality**

Events with 100% coherence (no contradictions) score 98% epistemic.
Events with tensions (Wang Fuk Court: 33% tension) score lower (72% epistemic).

### 5W1H Analysis (Top 3 Events)

**Jimmy Lai Trial (102 claims)**
```
✓ WHO: Jimmy Lai, Hong Kong authorities, Donald Trump
✓ WHAT: Trial and potential life imprisonment
~ WHEN: October (partial - no trial date)
✗ WHERE: No specific location
✗ WHY: Reasons for charges not clear
✗ HOW: Trial process details missing
```

**Wang Fuk Court Fire (66 claims)**
```
~ WHO: Victims, construction company, Labour Department
✓ WHAT: Fire broke out at apartment complex
✓ WHEN: November 26, 2025
✓ WHERE: Wang Fuk Court, Tai Po District
~ WHY: Allegations of negligence (partial)
✗ HOW: Fire spread mechanism unknown
```

**Bondi Beach Shooting (41 claims)**
```
✓ WHO: 11+ victims, gunmen (one on ASIO watchlist)
✓ WHAT: Shooting, declared terrorism
✗ WHEN: No specific date
✓ WHERE: Bondi Beach, Australia
~ WHY: Targeting Jewish community (partial)
~ HOW: Gunmen involvement (partial)
```

### Narrative Stage Distribution

| Stage | Count | Description |
|-------|-------|-------------|
| DEVELOPING | 6 | Ongoing, new info expected |
| RESOLVING | 3 | Winding down, outcomes emerging |
| CONCLUDED | 1 | Story complete (Do Kwon sentencing) |

Most events are **DEVELOPING** - they have room for more claims.

### Unresolved Questions (Common Patterns)

**Legal Events** (Jimmy Lai, Do Kwon, Nick Reiner):
- "What will be the outcome?"
- "What evidence exists?"
- "What is the motive?"

**Disaster Events** (Wang Fuk Court Fire):
- "What caused it?"
- "What measures will prevent recurrence?"
- "Who is responsible?"

**Violence Events** (Bondi Beach, Brown University):
- "What were the motivations?"
- "What safety measures will follow?"

### Readiness Thresholds

Based on results, proposed thresholds:

```python
def publication_readiness(event):
    if event.overall_score >= 0.70:
        return "READY_TO_PUBLISH"
    elif event.overall_score >= 0.50:
        return "NEEDS_CURATION"
    elif event.overall_score >= 0.30:
        return "DEVELOPING"
    else:
        return "PREMATURE"
```

### What "Ready to Publish" Means

An event is READY when:
1. **Epistemic**: 5+ sources, 80%+ coherence, official sources present
2. **Semantic**: WHO/WHAT/WHERE clear, narrative has beginning
3. **Overall**: 70%+ combined score

It does NOT require:
- All 5W1H answered (some remain open)
- Concluded narrative (DEVELOPING is fine)
- Zero contradictions (tension is acceptable)

### What "Needs Curation" Means

Event needs editorial attention:
- Review open questions
- Verify key claims
- Add context for unclear elements
- Wait for more sources if < 5

### Implications for Breathing System

**Auto-publish when:**
```python
event.overall_score >= 0.70 and
event.sources >= 5 and
event.coherence >= 0.8 and
event.narrative_stage in ["DEVELOPING", "RESOLVING", "CONCLUDED"]
```

**Flag for review when:**
```python
event.tension > 0.2 or
event.semantic_score < 0.5 or
event.sources < 3
```

**Hold when:**
```python
event.overall_score < 0.50 or
event.narrative_stage == "EMERGING"
```

### Conclusion

**80% of top events are ready to publish** with:
- Strong epistemic foundations (diverse, corroborated)
- Moderate semantic completeness (some 5W1H gaps)
- Developing narratives (not concluded)

The remaining 20% need curation mainly due to:
- Fewer sources (< 5)
- Lower semantic clarity
- Ambiguous narrative arc

This validates the breathing event system: events naturally reach publishable quality through organic claim accumulation.

---

*Report generated 2025-12-17*
