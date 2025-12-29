# Epistemic Kernel: Semantic Pattern Analysis

## Overview

The kernel must distinguish between claims that:
1. **ADD** to our understanding (COMPATIBLE/COMPLEMENTARY)
2. **REINFORCE** our understanding (CORROBORATE)
3. **REFINE** our understanding (REFINEMENT - more specific)
4. **UPDATE** our understanding (SUPERSEDE - new information)
5. **CHALLENGE** our understanding (CONTRADICT - incompatible)

The current kernel fails to distinguish #1 from #5 for narrative claims.

---

## Pattern Taxonomy

### 1. COMPLEMENTARY (Currently Missing)

**Definition**: Claims about the SAME event that describe DIFFERENT aspects or details. Both can be true simultaneously.

**Examples**:
```
Claim A: "A major fire broke out in Hong Kong building"
Claim B: "Residents reported smelling smoke around 3am"
→ COMPLEMENTARY: B adds detail to A, doesn't conflict

Claim A: "13 people were killed in the fire"
Claim B: "Over 100 people were injured"
→ COMPLEMENTARY: Different outcome dimensions

Claim A: "Fire started on the 8th floor"
Claim B: "Flames spread to 10 floors total"
→ COMPLEMENTARY: Origin vs spread, both true
```

**Test**: Can both claims be true at the same time?
- YES → COMPLEMENTARY (add both)
- NO → Check for CONTRADICTION

**Current Kernel Gap**: No logic to test logical compatibility. It only compares extracted "values".

---

### 2. REFINEMENT (Partially Implemented)

**Definition**: A claim that provides MORE SPECIFIC information about the same fact. The general claim is still true, but the specific one is more informative.

**Examples**:
```
General: "Fire in Hong Kong"
Specific: "Fire in Tai Po, Hong Kong"
More Specific: "Fire at Wang Fuk Court, Tai Po District"
→ Each REFINES the previous (replaces with more specific)

General: "Multiple casualties reported"
Specific: "At least 10 dead"
More Specific: "13 confirmed dead, 8 missing"
→ Each REFINES with more precision
```

**Test**: Does the new claim ENTAIL the old claim?
- "Fire at Wang Fuk Court" ENTAILS "Fire in Hong Kong"
- If new → old is true, new REFINES old

**Current Kernel Gap**: Only compares extracted values, can't detect semantic entailment.

---

### 3. CONTRADICTION (Overactive)

**Definition**: Claims that CANNOT both be true. One must be false.

**Examples**:
```
Claim A: "Fire started on the 3rd floor"
Claim B: "Fire started on the 8th floor"
→ CONTRADICTION: Fire has one origin point

Claim A: "13 dead after DNA testing"
Claim B: "17 dead after DNA testing"
→ CONTRADICTION: Same metric, same timeframe, different values

Claim A: "Fire caused by electrical fault"
Claim B: "Fire caused by arson"
→ CONTRADICTION: Mutually exclusive causes
```

**Test**: Is there a logical impossibility if both are true?
- Can fire start in two places simultaneously? → Context dependent
- Can death toll be both 13 AND 17? → NO, contradiction

**Current Kernel Gap**: Flags ANY difference as contradiction for single-value topics.

---

## Decision Logic Needed

For each new claim vs existing belief:

```
                        ┌─────────────────────┐
                        │   Same Aspect?      │
                        └──────────┬──────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │ NO                 │ YES                │
              ▼                    ▼                    │
        ADD as new           Same Topic?               │
        aspect claim         ┌────┴────┐               │
                            NO        YES              │
                            ▼          ▼               │
                      ADD as new   Compare values      │
                      topic claim       │              │
                                        ▼              │
                    ┌───────────────────┴───────────────┐
                    │                                   │
            ┌───────┴───────┐                   ┌───────┴───────┐
            │ Semantic      │                   │ Semantic      │
            │ Similarity?   │                   │ Similarity?   │
            └───────┬───────┘                   └───────┬───────┘
                    │                                   │
           ┌────────┼────────┐               ┌──────────┼──────────┐
           HIGH    MED      LOW             SAME      ENTAILS    DIFFERENT
           ▼       ▼         ▼               ▼          ▼           ▼
       CORROBORATE REFINE?  Check         CORROBORATE REFINE    Check
       (same fact) (more    compatibility  (exact)    (specific) compatibility
                   detail)     │                                    │
                               ▼                                    ▼
                    ┌──────────┴──────────┐              ┌──────────┴──────────┐
                    │ Can both be true?   │              │ Can both be true?   │
                    └──────────┬──────────┘              └──────────┬──────────┘
                               │                                    │
                     ┌─────────┼─────────┐               ┌──────────┼──────────┐
                     YES               NO                YES                  NO
                     ▼                  ▼                 ▼                   ▼
                 COMPLEMENTARY    CONTRADICTION     COMPLEMENTARY      CONTRADICTION
                 (add detail)     (flag conflict)   (add detail)       (flag conflict)
```

---

## Required Kernel Enhancements

### Enhancement 1: Compatibility Test (LLM-based)

Before flagging contradiction, ask:
```
Can both of these statements be true simultaneously?

Belief: "{existing_belief}"
Claim: "{new_claim}"

Answer: YES (compatible, both can be true) / NO (one must be false)
If NO, explain which fact is incompatible.
```

### Enhancement 2: Entailment Test (for REFINEMENT)

Check if new claim is more specific:
```
Does the new claim entail (imply) the existing belief?

Existing: "Fire in Hong Kong"
New: "Fire at Wang Fuk Court, Tai Po, Hong Kong"

If New→Existing is true, then New REFINES Existing.
```

### Enhancement 3: Semantic Similarity (embedding-based)

Use embeddings to detect paraphrases:
- Similarity > 0.90: CORROBORATE_EXACT
- Similarity > 0.75: CORROBORATE_SIMILAR / check REFINES
- Similarity < 0.75: Check compatibility (might be COMPLEMENTARY or CONTRADICT)

### Enhancement 4: Slot Type Intelligence

Different topics need different handling:

| Topic Type | Multiple Values? | Example |
|------------|------------------|---------|
| `count` | NO (single, monotonic) | death toll |
| `location_origin` | NO (single) | where fire started |
| `location_spread` | YES (accumulates) | floors affected |
| `cause` | MAYBE (primary/contributing) | cause of fire |
| `response_actions` | YES (accumulates) | rescue efforts |
| `victims_names` | YES (accumulates) | identified victims |

---

## Test Cases for Validation

### Should be COMPLEMENTARY (not CONTRADICTION):

1. "Fire broke out" + "Residents smelled smoke" → COMPLEMENTARY
2. "13 dead" + "100+ injured" → COMPLEMENTARY (different metrics)
3. "Fire on 8th floor" + "Spread to 10 floors" → COMPLEMENTARY
4. "Police investigating" + "13 arrested" → COMPLEMENTARY (update)

### Should be REFINEMENT:

1. "Fire in Hong Kong" → "Fire in Tai Po, HK" → REFINEMENT
2. "Multiple casualties" → "At least 13 dead" → REFINEMENT
3. "Early morning" → "Around 3:30am" → REFINEMENT

### Should be CONTRADICTION:

1. "Fire started on floor 3" vs "Fire started on floor 8" → CONTRADICTION
2. "Caused by electrical" vs "Caused by arson" → CONTRADICTION
3. "13 dead (final)" vs "17 dead (final)" → CONTRADICTION

### Should be CORROBORATE:

1. "13 killed in fire" + "Death toll reaches 13" → CORROBORATE
2. "Fire at Wang Fuk Court" + "Wang Fuk Court blaze" → CORROBORATE

---

## Implementation Priority

1. **HIGH**: Compatibility test before contradiction (stop false positives)
2. **HIGH**: Slot type per topic (single vs multi value)
3. **MEDIUM**: Embedding similarity for corroboration
4. **MEDIUM**: Entailment test for refinement
5. **LOW**: Named entity extraction for value comparison
