"""
REEE Theory Ledger
==================

This module documents the theoretical foundations and identifies where
higher-order structure TIGHTENS (not replaces) the existing system.

LOCKED RULES (2025-01)
======================

1. SINGLETON ANCHORS NEVER FORM CORES
   - They can only attach periphery
   - Cores require 2+ discriminative anchors with compatible context

2. BINDING ANCHOR MUST PASS CONTEXT-COMPATIBILITY TEST
   - For anchor e to bind scopes A and B:
     - Build P(C|e, scope=A) and P(C|e, scope=B) from companion sets
     - Compute Jaccard overlap (or Jensen-Shannon divergence)
     - If contexts are incompatible → e cannot count as binding signal

3. ENTROPY IS A PRIOR ON TRUST, NOT THE DECISION
   - High H(C|e) → "don't let e bind by itself; require more support"
   - But high entropy ≠ hub (Wang Fuk Court is rich backbone, not hub)
   - Use entropy for SCREENING, use compatibility for DECISIONS

4. EMIT META-CLAIMS WHEN TEST IS UNDERPOWERED
   - If companion set < min_companions → context_insufficient_support
   - REEE is honest about constraint scarcity
   - Underpowered tests allow binding but flag for human review

Why This Matters
----------------

The percolation problem: Bridge entities appear in multiple unrelated contexts.

Example (Hong Kong Fire case):
- John Lee appears in BOTH fire investigation AND Jimmy Lai trial
- IDF alone can't distinguish "bridge across topics" from "central to topic"
- Result without fix: mega-merge of unrelated events

The fix:
- John Lee in fire: companions = {Wang Fuk Court, Tai Po, Joe Chow}
- John Lee in Lai: companions = {Jimmy Lai, Esther Toh}
- Companion Jaccard = 0 → INCOMPATIBLE → John Lee cannot bind fire↔Lai

This is information-theoretic, not ad-hoc:
- P(C|e, scope) is a real conditional probability
- Context compatibility is a principled divergence measure
- We're checking if the entity's local context is consistent

What's NOT Valid (Pitfalls)
---------------------------

1. HIGH ENTROPY ≠ HUB
   - Wang Fuk Court has many companions within fire story
   - But it's still a legitimate backbone for that incident family
   - Entropy conflates "bridges topics" with "rich incident with aspects"

2. SAMPLE-SIZE / SPARSITY EFFECTS
   - With small counts (e.g., 4 vs 2 companions), raw Jaccard is brittle
   - Solution: Mark underpowered tests, allow binding but require audit

3. JAYNES INFERENCE SCOPE
   - MaxEnt/Bayesian inference is principled for TYPED VARIABLES
   - It's NOT principled for clustering (different mathematical object)
   - Clustering requires structural constraints, not just probabilistic

Implementation
--------------

The context compatibility check is inlined in aboutness/scorer.py:

```python
@dataclass
class ContextCompatibilityResult:
    compatible: bool
    overlap: float  # Jaccard overlap of companion sets
    companions1: Set[str]
    companions2: Set[str]
    underpowered: bool  # True if sample too small
    reason: str  # Human-readable explanation

def context_compatible(
    entity: str,
    surface1: Surface,
    surface2: Surface,
    min_overlap: float = 0.15,
    min_companions: int = 2,
) -> ContextCompatibilityResult:
    # Get companions in each surface
    companions1 = surface1.anchor_entities - {entity}
    companions2 = surface2.anchor_entities - {entity}

    # Check for underpowered test
    if len(companions1) < min_companions or len(companions2) < min_companions:
        return ContextCompatibilityResult(
            compatible=True,  # Allow but flag
            underpowered=True,
            reason="context_insufficient_support"
        )

    # Jaccard overlap
    overlap = len(companions1 & companions2) / len(companions1 | companions2)
    return ContextCompatibilityResult(
        compatible=overlap >= min_overlap,
        overlap=overlap,
        underpowered=False
    )
```

Integration Point: AboutnessScorer.score_pair()
- After mode-scoped filtering (for splittable anchors)
- Add context compatibility filter
- Emit meta-claims for audit trail

Future Work: Jensen-Shannon Divergence
--------------------------------------

Jaccard is a quick approximation. More principled:

```python
def jensen_shannon_divergence(p1: Dict[str, float], p2: Dict[str, float]) -> float:
    # JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    # where M = 0.5 * (P + Q)
    all_keys = set(p1.keys()) | set(p2.keys())
    m = {k: 0.5 * (p1.get(k, 0) + p2.get(k, 0)) for k in all_keys}

    def kl(p, q):
        return sum(p.get(k, 0) * math.log(p.get(k, 1e-10) / q.get(k, 1e-10))
                   for k in all_keys if p.get(k, 0) > 0)

    return 0.5 * kl(p1, m) + 0.5 * kl(p2, m)
```

JSD is bounded [0, 1], symmetric, and well-defined even with different supports.
With smoothing (add-one or Dirichlet), it handles sparse companions gracefully.

Test Cases
----------

1. John Lee fire↔lai:
   - companions_fire = {Wang Fuk Court, Tai Po, Joe Chow}
   - companions_lai = {Jimmy Lai, Esther Toh}
   - Jaccard = 0 → INCOMPATIBLE ✓

2. John Lee fire1↔fire2:
   - companions_fire1 = {Wang Fuk Court, Tai Po, Joe Chow}
   - companions_fire2 = {Wang Fuk Court, Fire Services}
   - Jaccard = 1/4 = 0.25 → COMPATIBLE ✓

3. Sparse surface (underpowered):
   - companions = {Unknown} (only 1)
   - Result: compatible=True, underpowered=True
   - Meta-claim emitted: "context_insufficient_support" ✓
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional, FrozenSet
import numpy as np


@dataclass
class MotifContext:
    """
    Higher-order context for an entity.

    For entity E, the motif context is the distribution of companions
    that appear with E across all claims/surfaces.

    IMPORTANT: This is for ANALYSIS only, not for decision-making.
    Use context_compatible() in scorer.py for binding decisions.

    Entropy interpretation:
    - Low H(C|e) → concentrated companions (often backbone-like)
    - High H(C|e) → dispersed companions (possibly hub-like)

    BUT: High entropy alone doesn't mean hub!
    Wang Fuk Court may have high entropy within fire story but is still backbone.
    """
    entity: str
    companions: Dict[str, int] = field(default_factory=dict)  # companion -> count
    total_occurrences: int = 0

    def add_companion(self, companion: str):
        self.companions[companion] = self.companions.get(companion, 0) + 1
        self.total_occurrences += 1

    def context_entropy(self) -> float:
        """
        Entropy of companion distribution (normalized [0,1]).

        Uses natural log, normalized by max possible entropy.

        WARNING: High entropy is a PRIOR on distrust, not a decision.
        Use this for screening, not for binding decisions.
        """
        if not self.companions or self.total_occurrences == 0:
            return 0.0

        entropy = 0.0
        for count in self.companions.values():
            p = count / self.total_occurrences
            if p > 0:
                entropy -= p * math.log(p)

        # Normalize by max possible entropy (uniform distribution)
        n = len(self.companions)
        max_entropy = math.log(n) if n > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def context_overlap(self, other: 'MotifContext') -> float:
        """Jaccard overlap between companion sets."""
        if not self.companions or not other.companions:
            return 0.0

        set1 = set(self.companions.keys())
        set2 = set(other.companions.keys())

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def weighted_overlap(self, other: 'MotifContext') -> float:
        """
        Bhattacharyya coefficient between companion distributions.

        More principled than Jaccard: accounts for how often
        companions appear, not just presence/absence.
        """
        if not self.companions or not other.companions:
            return 0.0

        # Normalize to probability distributions
        p1 = {k: v / self.total_occurrences for k, v in self.companions.items()}
        p2 = {k: v / other.total_occurrences for k, v in other.companions.items()}

        # Bhattacharyya coefficient: sum(sqrt(p1 * p2))
        all_companions = set(p1.keys()) | set(p2.keys())
        bc = sum(
            math.sqrt(p1.get(c, 0) * p2.get(c, 0))
            for c in all_companions
        )

        return bc


class MotifContextAnalyzer:
    """
    Analyzes higher-order motif context for entities.

    This is for ANALYSIS and EXPLORATION, not for production binding decisions.
    Production binding decisions use context_compatible() in scorer.py.
    """

    def __init__(self, surfaces: Dict[str, 'Surface']):
        self.surfaces = surfaces
        self.entity_contexts: Dict[str, MotifContext] = {}
        self._build_contexts()

    def _build_contexts(self):
        """Build motif context for each entity from surfaces."""
        for surface in self.surfaces.values():
            entities = list(surface.anchor_entities)

            for entity in entities:
                if entity not in self.entity_contexts:
                    self.entity_contexts[entity] = MotifContext(entity=entity)

                # Each other entity in this surface is a companion
                for companion in entities:
                    if companion != entity:
                        self.entity_contexts[entity].add_companion(companion)

    def entropy_screening(self, entity: str, threshold: float = 0.7) -> str:
        """
        Screen entity by entropy (PRIOR on trust, not decision).

        Returns:
            'low_entropy': Can trust as backbone candidate
            'high_entropy': Requires additional context check
            'unknown': No data for this entity
        """
        ctx = self.entity_contexts.get(entity)
        if not ctx:
            return 'unknown'

        entropy = ctx.context_entropy()
        return 'low_entropy' if entropy < threshold else 'high_entropy'

    def context_compatible_surfaces(
        self,
        entity: str,
        surface1_id: str,
        surface2_id: str,
        overlap_threshold: float = 0.15
    ) -> Tuple[bool, float, str]:
        """
        Check if entity's context is compatible between two surfaces.

        Returns:
            Tuple of (compatible, overlap, reason)
        """
        s1 = self.surfaces.get(surface1_id)
        s2 = self.surfaces.get(surface2_id)

        if not s1 or not s2:
            return False, 0.0, "surface_not_found"

        companions1 = s1.anchor_entities - {entity}
        companions2 = s2.anchor_entities - {entity}

        if not companions1 or not companions2:
            return True, 0.0, "insufficient_companions"

        intersection = len(companions1 & companions2)
        union = len(companions1 | companions2)
        overlap = intersection / union if union > 0 else 0.0

        if overlap >= overlap_threshold:
            return True, overlap, f"compatible: Jaccard={overlap:.3f}"
        else:
            return False, overlap, f"incompatible: Jaccard={overlap:.3f}, disjoint contexts"

    def get_analysis_report(self, entity: str) -> Dict:
        """Get full analysis report for an entity."""
        ctx = self.entity_contexts.get(entity)
        if not ctx:
            return {'entity': entity, 'status': 'not_found'}

        entropy = ctx.context_entropy()
        screening = self.entropy_screening(entity)
        top_companions = sorted(ctx.companions.items(), key=lambda x: -x[1])[:10]

        return {
            'entity': entity,
            'entropy': entropy,
            'screening': screening,
            'total_occurrences': ctx.total_occurrences,
            'unique_companions': len(ctx.companions),
            'top_companions': top_companions,
            'interpretation': (
                f"Entropy={entropy:.3f} → {screening}. "
                f"{'Requires context check for binding decisions.' if entropy >= 0.7 else 'Candidate for trusted backbone.'}"
            )
        }


# =============================================================================
# EXPERIMENTAL VALIDATION
# =============================================================================

async def validate_context_compatibility():
    """
    Validate the locked rules on Hong Kong Fire case.

    Expected outcomes:
    1. John Lee fire↔lai: INCOMPATIBLE (Jaccard=0)
    2. John Lee fire1↔fire2: COMPATIBLE (Jaccard>0.15)
    3. Underpowered tests: flagged with meta-claim
    """
    import sys
    sys.path.insert(0, '/app')

    from reee.aboutness.scorer import context_compatible
    from reee.types import Surface

    print("=" * 60)
    print("CONTEXT COMPATIBILITY VALIDATION")
    print("Locked Rules (2025-01)")
    print("=" * 60)

    # Mock surfaces
    s_fire = Surface(
        id='fire_1',
        anchor_entities={'John Lee', 'Wang Fuk Court', 'Tai Po', 'Joe Chow'}
    )
    s_lai = Surface(
        id='lai_1',
        anchor_entities={'John Lee', 'Jimmy Lai', 'Esther Toh'}
    )
    s_fire2 = Surface(
        id='fire_2',
        anchor_entities={'John Lee', 'Wang Fuk Court', 'Fire Services'}
    )
    s_sparse = Surface(
        id='sparse_1',
        anchor_entities={'John Lee', 'Unknown'}
    )

    print("\n1. John Lee fire↔lai (should be INCOMPATIBLE):")
    result = context_compatible('John Lee', s_fire, s_lai)
    print(f"   compatible={result.compatible}, overlap={result.overlap:.3f}")
    print(f"   reason: {result.reason}")
    assert not result.compatible, "John Lee should NOT bind fire to Lai!"

    print("\n2. John Lee fire1↔fire2 (should be COMPATIBLE):")
    result = context_compatible('John Lee', s_fire, s_fire2)
    print(f"   compatible={result.compatible}, overlap={result.overlap:.3f}")
    print(f"   reason: {result.reason}")
    assert result.compatible, "John Lee SHOULD bind fire1 to fire2!"

    print("\n3. Wang Fuk Court fire1↔fire2 (BACKBONE, should be COMPATIBLE):")
    result = context_compatible('Wang Fuk Court', s_fire, s_fire2)
    print(f"   compatible={result.compatible}, overlap={result.overlap:.3f}")
    print(f"   reason: {result.reason}")
    # Note: Wang Fuk Court may have high entropy but is still compatible
    # because its companions overlap within fire context

    print("\n4. Underpowered test (sparse surface):")
    result = context_compatible('John Lee', s_fire, s_sparse)
    print(f"   compatible={result.compatible}, underpowered={result.underpowered}")
    print(f"   reason: {result.reason}")
    assert result.underpowered, "Should flag as underpowered!"

    print("\n" + "=" * 60)
    print("✓ All locked rules validated")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(validate_context_compatibility())
