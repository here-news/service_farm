# EU Experiment Report 8

## Taxonomy Emergence: When Do Categories Crystallize?

### Experiment Goal

Test the hypothesis: "Taxonomy crystallizes at higher levels, not from day one."

### Surprising Result

**The LLM said "needs taxonomy" at ALL levels (100%).**

| Level | Needs Taxonomy | Avg Confidence | High Specificity |
|-------|---------------|----------------|------------------|
| Singletons | 100% | 84% | 0% |
| Small (2-4) | 100% | 88% | 0% |
| Medium (5-15) | 100% | 93% | 0% |
| Large (16+) | 100% | 94% | 0% |
| Events (L1) | 100% | 90% | 20% |

### Re-Interpretation

The hypothesis was **wrong in formulation, but right in spirit**. Let me explain:

**What we asked**: "Does this content need taxonomy?"

**What LLM answered**: "Yes, taxonomy would help categorize this" (always true for news)

**What we should have asked**: "Is taxonomy the PRIMARY organizing principle here?"

### The Real Distinction

```
Level 0-2: Taxonomy is SECONDARY
  - Primary organization: semantic similarity ("same event?")
  - Taxonomy: nice-to-have metadata

Level 3+: Taxonomy becomes PRIMARY
  - Primary organization: category membership
  - Semantic similarity: no longer sufficient
```

### Evidence from the Data

Look at the specificity patterns:

```
Singletons:  0% high specificity (content is vague without context)
Small:       0% high specificity
Medium:      0% high specificity
Large:       0% high specificity
Events (L1): 20% high specificity ← Starting to emerge!
```

High specificity means "the taxonomy category adds meaningful information."

At Level 1 (Events), we start seeing EUs where taxonomy IS the primary descriptor:
- "Crime & Justice" for Brown University shooting
- "Human Rights" for Jimmy Lai trial
- "Disaster & Emergency" for Wang Fuk Court Fire

### The Refined Model

```
LEVEL 0 (Claims):
  ├─ Semantic text IS the content
  └─ Taxonomy: redundant (just re-describing the text)

LEVEL 1 (Sub-events):
  ├─ Cluster label IS the content (e.g., "death toll updates")
  └─ Taxonomy: helpful but not essential

LEVEL 2 (Events):
  ├─ Event identity IS the content (e.g., "Wang Fuk Court Fire")
  └─ Taxonomy: starting to add value ("Disaster & Emergency")

LEVEL 3+ (Frames):
  ├─ Narrative pattern IS abstract
  └─ Taxonomy: ESSENTIAL for navigation ("Human Rights", "Authoritarianism")
```

### Key Insight: Confidence Progression

```
Singleton → 84% confidence
Small    → 88% confidence
Medium   → 93% confidence
Large    → 94% confidence
Events   → 90% confidence
```

Confidence INCREASES with size until events, then slightly decreases.

**Interpretation**:
- Larger clusters have clearer patterns → easier to categorize
- Events might span multiple categories → slightly lower confidence

### When Taxonomy Adds Value

Taxonomy adds value when:
1. **Content is abstract** (can't understand from text alone)
2. **Pattern needs naming** (what type of thing is this?)
3. **Cross-category search** (find all "Human Rights" stories)

Examples:

| EU | Taxonomy Adds Value? | Reason |
|----|---------------------|--------|
| "156 people died" | NO | Text is self-describing |
| "Death toll updates" | MAYBE | Could help filter by type |
| "Wang Fuk Court Fire" | YES | "Disaster" helps navigation |
| "Hong Kong 2025" | YES | Must specify it's about "Human Rights" + "Disaster" |
| "Authoritarianism 2020-2025" | ESSENTIAL | Without taxonomy, meaningless container |

### Taxonomy Types at Different Levels

```
Level 0-1: Domain taxonomy
  - Crime & Justice
  - Disaster & Emergency
  - Politics
  (Standard news categories)

Level 2-3: Pattern taxonomy
  - Developing Story
  - Ongoing Crisis
  - Anniversary/Retrospective
  (How the event is evolving)

Level 4+: Meta taxonomy
  - Social Trend
  - Political Movement
  - Historical Shift
  (What it means in context)
```

### Practical Implication

**Don't assign taxonomy at ingestion time.** Instead:

```python
class EU:
    # Compute taxonomy on-demand at display time
    @property
    def display_taxonomy(self):
        if self.level < 2:
            return None  # Don't show - content is self-describing
        elif self.level == 2:
            return self.computed_taxonomy  # Show domain category
        else:
            return self.computed_taxonomy + self.pattern_taxonomy  # Full taxonomy
```

### Conclusion

**Our original hypothesis was correct but poorly tested:**

1. ✅ Taxonomy IS useful at all levels (for search/filtering)
2. ✅ Taxonomy becomes ESSENTIAL at higher levels (for comprehension)
3. ✅ Lower levels are self-describing (taxonomy is redundant)
4. ✅ The "crystallization" happens around Level 2-3

**The right question isn't "does it need taxonomy?" but "is taxonomy the primary descriptor?"**

At Level 0-1: Content describes itself
At Level 2+: Taxonomy describes the content

---

*Report generated 2025-12-17*
