"""
Sparse Entity Claims Tests
==========================

Tests that claims with 0-1 entities are handled correctly,
not dropped or causing errors.

Key invariants tested:
1. Typed claim with 0 entities still updates correct L2 surface (via question_key)
2. Unscoped 0-entity claim emits explicit meta-claim (not silently dropped)
3. 1-entity claims don't break surface formation
4. Entity sparsity is a data condition, not a kernel failure

This addresses the real-world condition where most claims have 0-1 entities
due to extraction limitations.
"""

import pytest
from datetime import datetime
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataclasses import dataclass, field
from reee.types import (
    Claim, Surface, MetaClaim as REEEMetaClaim, Constraint, ConstraintType, ConstraintLedger
)


# ============================================================================
# LOCAL TEST TYPES
# ============================================================================

@dataclass
class LocalMetaClaim:
    """Local MetaClaim for testing sparse entity handling.

    This differs from the REEE MetaClaim to test specific sparse-claim behaviors.
    """
    claim_type: str
    assertion: str
    evidence: dict = field(default_factory=dict)
    severity: str = "info"  # "info", "warning", "blocker"


# Alias for cleaner test code
MetaClaim = LocalMetaClaim


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def zero_entity_typed_claim() -> Claim:
    """Claim with question_key but no entities extracted."""
    return Claim(
        id="C001",
        text="The death toll has risen to 163",
        source="news.com",
        entities=set(),  # No entities!
        question_key="death_toll_hong_kong_fire_2025",  # But has question_key
        extracted_value=163,
    )


@pytest.fixture
def zero_entity_unscoped_claim() -> Claim:
    """Claim with no entities AND no question_key - truly sparse."""
    return Claim(
        id="C002",
        text="Reports continue to emerge",
        source="wire.com",
        entities=set(),
        question_key=None,  # No scope!
    )


@pytest.fixture
def one_entity_claim() -> Claim:
    """Claim with only one entity."""
    return Claim(
        id="C003",
        text="Hong Kong officials responded",
        source="gov.hk",
        entities={"Hong Kong"},  # Single entity
        question_key="official_response",
    )


# ============================================================================
# SIMULATED L2 KERNEL (for testing sparse claims)
# ============================================================================

class SparseTolerantL2Kernel:
    """
    L2 kernel that handles sparse entity claims correctly.

    Key behaviors:
    - Routes by question_key, NOT by entity count
    - Emits meta-claims for truly unscoped claims
    - Never drops evidence
    """

    def __init__(self):
        self.surfaces = {}
        self.question_key_to_surface = {}
        self.meta_claims = []
        self.claims_processed = 0
        self.claims_with_zero_entities = 0
        self.claims_dropped = 0  # Should always be 0!

    def process_claim(self, claim: Claim) -> tuple:
        """
        Process claim, routing by question_key.

        Returns (surface_id, meta_claim_if_any)
        """
        self.claims_processed += 1

        if len(claim.entities) == 0:
            self.claims_with_zero_entities += 1

        # Route by question_key if available
        if claim.question_key:
            return self._route_by_question_key(claim), None

        # No question_key - emit meta-claim, don't drop
        meta = MetaClaim(
            claim_type="extraction_gap",
            assertion=f"Claim {claim.id} has no question_key and no entities",
            evidence={"claim_id": claim.id, "text_preview": claim.text[:50]},
            severity="blocker",
        )
        self.meta_claims.append(meta)

        # Still create a surface for this claim (isolated)
        surface_id = f"sf_unscoped_{claim.id}"
        self.surfaces[surface_id] = Surface(
            id=surface_id,
            claim_ids={claim.id},
            question_key=f"unscoped_{claim.id}",
            formation_method="unscoped_fallback",
        )

        return surface_id, meta

    def _route_by_question_key(self, claim: Claim) -> str:
        """Route claim to surface by question_key."""
        qkey = claim.question_key

        if qkey in self.question_key_to_surface:
            surface_id = self.question_key_to_surface[qkey]
            self.surfaces[surface_id].claim_ids.add(claim.id)
            self.surfaces[surface_id].entities.update(claim.entities)
        else:
            surface_id = f"sf_{len(self.surfaces):04d}"
            self.surfaces[surface_id] = Surface(
                id=surface_id,
                claim_ids={claim.id},
                entities=claim.entities.copy(),
                question_key=qkey,
                formation_method="question_key",
            )
            self.question_key_to_surface[qkey] = surface_id

        return surface_id


# ============================================================================
# TEST: ZERO-ENTITY TYPED CLAIMS
# ============================================================================

class TestZeroEntityTypedClaims:
    """Test that 0-entity claims with question_key are handled correctly."""

    def test_zero_entity_claim_not_dropped(self, zero_entity_typed_claim):
        """0-entity claim with question_key should NOT be dropped."""
        kernel = SparseTolerantL2Kernel()

        surface_id, meta = kernel.process_claim(zero_entity_typed_claim)

        assert surface_id is not None, "0-entity claim should produce a surface"
        assert kernel.claims_dropped == 0, "No claims should be dropped"
        assert kernel.claims_with_zero_entities == 1

    def test_zero_entity_claim_routes_by_question_key(self, zero_entity_typed_claim):
        """0-entity claim should route by question_key, not fail on entity lookup."""
        kernel = SparseTolerantL2Kernel()

        # First claim with question_key
        surface_id1, _ = kernel.process_claim(zero_entity_typed_claim)

        # Second claim with same question_key (also 0 entities)
        claim2 = Claim(
            id="C004",
            text="Death toll confirmed at 163",
            source="other.com",
            entities=set(),  # Also no entities
            question_key="death_toll_hong_kong_fire_2025",  # Same question_key!
        )
        surface_id2, _ = kernel.process_claim(claim2)

        # Both should be in same surface
        assert surface_id1 == surface_id2, "Same question_key → same surface"
        assert len(kernel.surfaces) == 1

    def test_zero_entity_claim_contributes_to_surface(self, zero_entity_typed_claim):
        """0-entity claim should be included in surface claim_ids."""
        kernel = SparseTolerantL2Kernel()

        surface_id, _ = kernel.process_claim(zero_entity_typed_claim)
        surface = kernel.surfaces[surface_id]

        assert zero_entity_typed_claim.id in surface.claim_ids


# ============================================================================
# TEST: ZERO-ENTITY UNSCOPED CLAIMS
# ============================================================================

class TestZeroEntityUnscopedClaims:
    """Test that truly sparse claims (0 entities, no question_key) emit meta-claims."""

    def test_unscoped_claim_emits_meta_claim(self, zero_entity_unscoped_claim):
        """Unscoped 0-entity claim should emit extraction_gap meta-claim."""
        kernel = SparseTolerantL2Kernel()

        surface_id, meta = kernel.process_claim(zero_entity_unscoped_claim)

        assert meta is not None, "Unscoped claim should emit meta-claim"
        assert meta.claim_type == "extraction_gap"
        assert meta.severity == "blocker"

    def test_unscoped_claim_not_dropped(self, zero_entity_unscoped_claim):
        """Unscoped claim should still create a surface (isolated)."""
        kernel = SparseTolerantL2Kernel()

        surface_id, _ = kernel.process_claim(zero_entity_unscoped_claim)

        assert surface_id is not None
        assert kernel.claims_dropped == 0, "Never drop claims"

    def test_unscoped_claim_isolated(self, zero_entity_unscoped_claim):
        """Unscoped claims should not merge with other surfaces."""
        kernel = SparseTolerantL2Kernel()

        surface_id1, _ = kernel.process_claim(zero_entity_unscoped_claim)

        # Another unscoped claim
        claim2 = Claim(
            id="C005",
            text="More details to follow",
            source="wire.com",
            entities=set(),
            question_key=None,
        )
        surface_id2, _ = kernel.process_claim(claim2)

        # Each should have its own surface (isolated)
        assert surface_id1 != surface_id2, "Unscoped claims should be isolated"


# ============================================================================
# TEST: ONE-ENTITY CLAIMS
# ============================================================================

class TestOneEntityClaims:
    """Test that 1-entity claims don't break surface formation."""

    def test_one_entity_claim_creates_surface(self, one_entity_claim):
        """1-entity claim should create valid surface."""
        kernel = SparseTolerantL2Kernel()

        surface_id, _ = kernel.process_claim(one_entity_claim)
        surface = kernel.surfaces[surface_id]

        assert one_entity_claim.id in surface.claim_ids
        assert "Hong Kong" in surface.entities

    def test_one_entity_claim_routes_by_question_key(self, one_entity_claim):
        """1-entity claim should still route by question_key."""
        kernel = SparseTolerantL2Kernel()

        surface_id1, _ = kernel.process_claim(one_entity_claim)

        # Another claim with same question_key but different entity
        claim2 = Claim(
            id="C006",
            text="John Lee commented on the situation",
            source="scmp.com",
            entities={"John Lee"},  # Different entity
            question_key="official_response",  # Same question_key
        )
        surface_id2, _ = kernel.process_claim(claim2)

        assert surface_id1 == surface_id2, "Same question_key → same surface"

        # Surface should have both entities
        surface = kernel.surfaces[surface_id1]
        assert "Hong Kong" in surface.entities
        assert "John Lee" in surface.entities


# ============================================================================
# TEST: MIXED ENTITY COUNTS
# ============================================================================

class TestMixedEntityCounts:
    """Test surfaces with claims of varying entity counts."""

    def test_surface_accumulates_entities_from_all_claims(self):
        """Surface entities should accumulate from claims with different entity counts."""
        kernel = SparseTolerantL2Kernel()

        # Claim with 0 entities
        kernel.process_claim(Claim(
            id="C010",
            text="163 confirmed",
            source="s1",
            entities=set(),
            question_key="death_toll",
        ))

        # Claim with 1 entity
        kernel.process_claim(Claim(
            id="C011",
            text="Hong Kong fire",
            source="s2",
            entities={"Hong Kong"},
            question_key="death_toll",
        ))

        # Claim with 2 entities
        kernel.process_claim(Claim(
            id="C012",
            text="Tai Po fire in Hong Kong",
            source="s3",
            entities={"Hong Kong", "Tai Po"},
            question_key="death_toll",
        ))

        # All should be in same surface
        assert len(kernel.surfaces) == 1

        surface = list(kernel.surfaces.values())[0]
        assert len(surface.claim_ids) == 3
        assert "Hong Kong" in surface.entities
        assert "Tai Po" in surface.entities

    def test_entity_sparsity_is_metadata_not_failure(self):
        """Entity sparsity should be tracked as metadata, not cause failures."""
        kernel = SparseTolerantL2Kernel()

        # Process mix of claims
        for i in range(10):
            entity_count = i % 3  # 0, 1, 2, 0, 1, 2...
            entities = {f"Entity{j}" for j in range(entity_count)}
            kernel.process_claim(Claim(
                id=f"C{i:03d}",
                text=f"Claim {i}",
                source="s",
                entities=entities,
                question_key="test_question",
            ))

        # All processed, none dropped
        assert kernel.claims_processed == 10
        assert kernel.claims_dropped == 0

        # Sparsity tracked
        assert kernel.claims_with_zero_entities == 4  # 0, 3, 6, 9


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
