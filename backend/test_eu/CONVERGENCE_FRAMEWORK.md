# Convergence to Truth: The General Framework

## The Core Problem

Given a stream of claims C = {c₁, c₂, ..., cₙ} about a topic, how do we:
1. Converge toward truth as evidence accumulates
2. Maintain consistency (same inputs → same outputs)
3. Surface uncertainty honestly
4. Resolve contradictions systematically

## Theoretical Foundation

### From Jaynes (Probability Theory)
- Probability quantifies plausibility given information
- P(H|E) = P(E|H) × P(H) / P(E)
- Beliefs update with evidence, not replace

### From Information Theory
- Entropy H = -Σ p log p measures uncertainty
- Knowledge = entropy reduction
- Convergence = entropy decreasing over time

### From the Calibration Experiments
```
H = 1.0 - 0.49×(n_corr × 0.35)^0.30 + 0.27×(n_contra)^0.31

Where:
- n_corr = corroborating claims
- 0.35 = independence ratio (65% are copies)
- n_contra = contradicting claims
```

## The General Solution

### 1. THE RELATE OPERATION

The atomic operation that determines how two claims relate:

```
relate(c₁, c₂) → {type, confidence}

Types:
- CORROBORATES: Same assertion, independent source
- UPDATES: New value supersedes old (temporal evolution)
- REFINES: Adds precision to existing claim
- CONTRADICTS: Incompatible assertions
- UNRELATED: Different topics
```

**Inputs to RELATE:**
1. Semantic similarity (embeddings)
2. Entity overlap
3. Temporal ordering (if available)
4. Domain priors (monotonicity, etc.)

**Key insight from experiments:**
> "Many 'corroborations' are actually updates (new details about same topic)"

### 2. DOMAIN PRIORS

Prior knowledge that informs relationship classification:

| Metric Type | Prior | Implication |
|-------------|-------|-------------|
| Death toll | Monotonic ↑ | Later higher value = UPDATE, not CONTRADICT |
| Missing count | Monotonic ↓ | Later lower value = UPDATE (people found) |
| Timestamps | Later supersedes | With temporal ordering, later claims dominate |
| Official sources | Higher authority | Official statement resolves uncertainty |

**When temporal ordering exists:**
- value₁ at t₁, value₂ at t₂, t₂ > t₁
- If monotonic prior applies and values follow direction → UPDATE
- If values violate direction → TRUE CONTRADICTION

**When temporal ordering is unknown:**
- Must treat as potential contradiction
- Entropy stays HIGH (honestly uncertain)
- Need more evidence to resolve

### 3. ENTROPY COMPUTATION

Only TRUE contradictions increase entropy:

```python
def compute_entropy(claims, belief, priors):
    # Count corroborations (within tolerance of belief)
    n_corr = count(c for c in claims if is_corroborating(c, belief))

    # Count TRUE contradictions (not updates)
    n_contra = count_true_contradictions(claims, priors)

    # Apply calibrated formula
    effective_corr = n_corr * independence_ratio
    H = 1.0 - 0.49 * (effective_corr ** 0.30) + 0.27 * (n_contra ** 0.31)

    return max(0.05, min(0.99, H))
```

### 4. CONVERGENCE CRITERIA

The system has converged when:

1. **Entropy is LOW** (< 0.3)
   - Strong corroboration, few contradictions

2. **Belief is STABLE**
   - Same value for N consecutive claims
   - New claims don't change the answer

3. **Contradictions are RESOLVED**
   - One side has 3x+ support of the other
   - Or official source provides definitive answer

### 5. THE COMPLETE PROCESS

```
For each new claim c:

1. EXTRACT: metrics, entities, temporal info

2. RELATE: for each existing claim cᵢ in event
   - Compute relationship type using:
     - semantic_similarity(c, cᵢ)
     - temporal_order(c, cᵢ)
     - domain_priors(metric_type)
   - Output: CORROBORATES | UPDATES | REFINES | CONTRADICTS

3. UPDATE BELIEF:
   - If CORROBORATES → n_corr++
   - If UPDATES → replace old value, no entropy penalty
   - If REFINES → increase precision
   - If CONTRADICTS → n_contra++, surface conflict

4. COMPUTE ENTROPY:
   - Using calibrated formula
   - With independence discounting
   - Only TRUE contradictions count

5. CHECK CONVERGENCE:
   - Entropy < threshold?
   - Belief stable for N claims?
   - Contradictions resolved?

6. OUTPUT:
   - Current belief
   - Entropy (uncertainty)
   - Active contradictions
   - What would resolve them
```

## Why This Works

### Mathematical Consistency
- Same inputs → same outputs (deterministic given priors)
- Entropy formula is calibrated against ground truth
- Independence discounting prevents false certainty

### Honest Uncertainty
- When temporal order unknown → entropy stays high
- When true contradictions exist → surfaced explicitly
- Never claims certainty we don't have

### Convergence Property
- Each corroborating claim reduces entropy
- Updates (temporal evolution) don't add uncertainty
- Only true contradictions block convergence
- Eventually settles on truth or identifies irreducible plurality

## Implementation Priorities

1. **RELATE operation** - General classifier for claim relationships
   - Use LLM for semantic understanding
   - Incorporate temporal ordering
   - Apply domain priors

2. **Temporal data** - Ensure pub_time flows through system
   - Pages have pub_time
   - Claims inherit from page
   - Sort by time for proper ordering

3. **Domain priors** - Configurable per metric type
   - Monotonic counts: deaths, injured, arrested
   - Decreasing counts: missing (people found)
   - Other types as identified

4. **Convergence tracking** - Metrics over time
   - Entropy trajectory
   - Belief stability
   - Contradiction resolution

## References

- `docs/20.theory.md` - Core theoretical foundation
- `docs/26.theory.calibration.md` - Experimental validation
- `test_eu/calibrated_entropy.py` - Calibrated formula
- `test_eu/unified_engine.py` - Multi-signal affinity
- `test_eu/relationship_classification.py` - Event relationship types
- `test_eu/basic_form_jaynes.py` - Jaynesian probability model
