# Jaynes Maximum Entropy Alignment Analysis

## The Principle

**Jaynes' Maximum Entropy Theorem**: Given prior information, the probability distribution that best represents our knowledge is the one with **maximum entropy** subject to the constraints imposed by what we actually know.

Key insight: **Don't assume what you don't know.**

## Topological/Hypergeometric View

The belief state can be viewed as a point in a high-dimensional space:

```
Belief Space B = (b₁, b₂, ..., bₙ) where each bᵢ ∈ [0, 1]

Each dimension represents:
- An aspect (WHAT, WHEN, WHERE, ...)
- A topic within aspect (death_toll, location, cause, ...)
- Confidence level
- Source agreement ratio
```

The "best truth surface" is the manifold that:
1. Satisfies all claim constraints
2. Maximizes entropy for unconstrained dimensions
3. Minimizes entropy only where evidence compels it

## Current Kernel Analysis

### What Aligns with Jaynes ✓

| Principle | Kernel Implementation |
|-----------|----------------------|
| **Start with maximum uncertainty** | New aspects start with `status=MISSING`, `entropy=1.0` |
| **Only constrain with evidence** | Beliefs only created when claims arrive |
| **Corroboration reduces uncertainty** | Multiple sources → higher confidence, lower entropy |
| **Preserve uncertainty where absent** | Aspects without claims stay `MISSING` |

### What VIOLATES Jaynes ✗

| Violation | Current Behavior | Jaynes-Correct Behavior |
|-----------|------------------|------------------------|
| **Overconfident seeding** | First claim → confidence = source_credibility | First claim → high entropy, tentative |
| **Binary belief states** | Value is either set or None | Should be probability distribution |
| **No prior integration** | Ignores domain priors | Should incorporate base rates |
| **Entropy calculation naive** | Based on source count only | Should model full uncertainty |

## Required Enhancements for Jaynes Alignment

### 1. Probabilistic Belief States

Instead of single values, beliefs should be distributions:

```python
@dataclass
class JaynesBeliefState:
    topic: str
    aspect: Aspect

    # Distribution over possible values (not single value)
    value_distribution: Dict[str, float]  # {value: probability}

    # Entropy of this belief
    entropy: float  # Computed from distribution

    # Constraints from claims
    constraints: List[Constraint]

    def max_entropy_update(self, claim_value: str, claim_weight: float):
        """
        Update distribution using MaxEnt principle:
        New distribution = argmax H(P) subject to new constraint
        """
        pass
```

### 2. Constraint-Based Updates

Each claim adds a constraint, not a replacement:

```python
@dataclass
class Constraint:
    """A constraint on the belief space from a claim."""
    claim_id: str
    constraint_type: str  # 'equals', 'greater_than', 'contains', 'excludes'
    value: Any
    weight: float  # Source credibility
    timestamp: str
```

### 3. Entropy Maximization

The kernel should maximize entropy subject to constraints:

```
H(B) = -Σ p(bᵢ) log p(bᵢ)

Subject to:
- All claim constraints
- Domain priors (base rates)
- Logical consistency
```

### 4. The Jaynes Update Rule

When new claim C arrives:

```
P_new(B) = P_old(B) × L(C|B) / Z

Where:
- L(C|B) = likelihood of claim given belief state
- Z = normalization constant
- P_new maximizes entropy subject to constraint from C
```

## Topological Interpretation

### The Belief Manifold

```
                    High Entropy
                    (Uncertain)
                         │
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    │     MISSING        │     PARTIAL        │
    │   (max entropy)    │   (constrained)    │
    │                    │                    │
    ├────────────────────┼────────────────────┤
    │                    │                    │
    │    CONTESTED       │   ESTABLISHED      │
    │  (multi-modal)     │   (min entropy)    │
    │                    │                    │
    └────────────────────┼────────────────────┘
                         │
                    Low Entropy
                    (Certain)
```

### Claim Processing as Geodesic Flow

Each claim moves the belief state along a geodesic in the belief manifold:

1. **SEED**: Jump from `∅` to initial point (high entropy region)
2. **CORROBORATE**: Move toward lower entropy (same direction, more certain)
3. **COMPLEMENT**: Expand to adjacent dimension (add constraint)
4. **CONTRADICT**: Create bifurcation (multi-modal distribution)
5. **UPDATE**: Slide along time axis (supersede with new constraint)

### The "Best Truth Surface"

The optimal belief state is the surface that:

```
min Σ (violation of constraints)²
max H(P) for unconstrained dimensions
```

This is a **saddle point** problem - we want maximum entropy (flatness) everywhere except where claims force curvature.

## Implementation Checklist

### Phase 1: Entropy-Aware States
- [ ] Add entropy computation to BeliefState
- [ ] Track constraint count per belief
- [ ] Entropy decreases only with corroboration

### Phase 2: Distributional Beliefs
- [ ] Replace single values with distributions
- [ ] Implement MaxEnt update rule
- [ ] Handle multi-modal distributions (conflicts)

### Phase 3: Topological Metrics
- [ ] Compute manifold curvature from constraint density
- [ ] Track geodesic distance between states
- [ ] Identify saddle points (contested beliefs)

### Phase 4: Jaynes-Optimal Inference
- [ ] Prior integration (domain base rates)
- [ ] Proper Bayesian update with MaxEnt
- [ ] Uncertainty quantification for each belief

## Test: Does Current Kernel Violate Jaynes?

### Test Case 1: Single Source Overconfidence

```
Claim: "13 dead" (Source: HKFP, credibility=0.7)
Current: confidence=0.7, entropy=low
Jaynes: confidence should be LOW (single source), entropy HIGH
```

**VIOLATION**: Kernel is overconfident on single-source claims.

### Test Case 2: Missing Aspect Uncertainty

```
No claims about CAUSE aspect
Current: status=MISSING, no belief
Jaynes: Correct - maximum entropy maintained
```

**ALIGNED**: Kernel correctly preserves uncertainty.

### Test Case 3: Contradicting Sources

```
Claim A: "Floor 3" (Source A)
Claim B: "Floor 8" (Source B)
Current: Creates CONTRADICTION flag
Jaynes: Should create bimodal distribution P(floor=3)=0.5, P(floor=8)=0.5
```

**PARTIAL**: Kernel flags but doesn't model uncertainty properly.

## Conclusion

The kernel is **partially aligned** with Jaynes:

| Aspect | Alignment |
|--------|-----------|
| Starting uncertainty | ✓ Aligned |
| Evidence-based constraints | ✓ Aligned |
| Corroboration reduces entropy | ✓ Aligned |
| Single-source confidence | ✗ Overconfident |
| Distributional beliefs | ✗ Missing |
| Proper Bayesian update | ✗ Missing |
| Multi-modal conflicts | ✗ Incomplete |

**Recommendation**: Enhance kernel with probabilistic belief states and MaxEnt updates for full Jaynes alignment.
