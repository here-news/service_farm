"""
Surface Identity Tests
======================

Tests that L2 Surface identity is based on question_key,
NOT on entity overlap or embedding similarity.

Key invariants tested:
1. Same question_key → same surface
2. Different question_key → different surfaces (even with entity overlap)
3. Conflicts stay inside one surface
4. Motifs are L3 evidence, not L2 identity

This is Upgrade 2 from the kernel consolidation plan:
"L2 = question_key surfaces (not motifs)"
"""

import pytest
from datetime import datetime
from typing import Set, Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reee.types import (
    Surface, Claim, Relation, ConstraintLedger
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def death_toll_claims() -> list:
    """Claims about death toll - same question_key, should be same surface."""
    return [
        Claim(
            id="C001",
            text="At least 160 people killed in Hong Kong fire",
            source="scmp.com",
            entities={"Hong Kong", "Tai Po", "Fire"},
            anchor_entities={"Hong Kong Fire 2025"},
            question_key="death_toll_hong_kong_fire_2025",
            extracted_value=160,
        ),
        Claim(
            id="C002",
            text="Death toll rises to 163 in Tai Po apartment fire",
            source="rthk.hk",
            entities={"Hong Kong", "Tai Po", "Fire", "Queen Elizabeth Hospital"},
            anchor_entities={"Hong Kong Fire 2025"},
            question_key="death_toll_hong_kong_fire_2025",  # Same question_key!
            extracted_value=163,
            has_update_language=True,
        ),
        Claim(
            id="C003",
            text="Fire Services confirms 163 casualties at Tai Po",
            source="news.gov.hk",
            entities={"Hong Kong", "Tai Po", "Fire Services"},
            anchor_entities={"Hong Kong Fire 2025"},
            question_key="death_toll_hong_kong_fire_2025",  # Same question_key!
            extracted_value=163,
        ),
    ]


@pytest.fixture
def fire_cause_claims() -> list:
    """Claims about fire cause - different question_key from death toll."""
    return [
        Claim(
            id="C010",
            text="Investigators suspect electrical fault caused Tai Po fire",
            source="scmp.com",
            entities={"Hong Kong", "Tai Po", "Fire"},  # Same entities as death toll!
            anchor_entities={"Hong Kong Fire 2025"},
            question_key="fire_cause_tai_po_2025",  # Different question_key!
            extracted_value="electrical_fault",
        ),
        Claim(
            id="C011",
            text="Arson ruled out in Tai Po fire investigation",
            source="rthk.hk",
            entities={"Hong Kong", "Tai Po", "Fire"},
            anchor_entities={"Hong Kong Fire 2025"},
            question_key="fire_cause_tai_po_2025",  # Same as C010
            extracted_value="not_arson",
        ),
    ]


@pytest.fixture
def conflicting_claims() -> list:
    """Claims with same question_key but conflicting values."""
    return [
        Claim(
            id="C020",
            text="Early reports suggest 100 dead in fire",
            source="tabloid.com",
            entities={"Hong Kong Fire"},
            question_key="death_toll_hong_kong_fire_2025",
            extracted_value=100,
        ),
        Claim(
            id="C021",
            text="Hospital confirms 163 deaths from fire",
            source="hospital.gov.hk",
            entities={"Hong Kong Fire", "Queen Elizabeth Hospital"},
            question_key="death_toll_hong_kong_fire_2025",  # Same question_key!
            extracted_value=163,  # Conflicts with C020!
        ),
    ]


# ============================================================================
# SIMULATED L2 KERNEL (for testing invariants)
# ============================================================================

class L2SurfaceKernel:
    """
    Simplified L2 kernel for testing surface identity invariants.

    Routes claims to surfaces by question_key.
    """

    def __init__(self):
        self.surfaces: Dict[str, Surface] = {}
        self.question_key_to_surface: Dict[str, str] = {}
        self.claim_to_surface: Dict[str, str] = {}
        self.internal_relations: Dict[str, list] = {}  # surface_id -> [(c1, c2, relation)]

    def process_claim(self, claim: Claim) -> str:
        """
        Route claim to surface by question_key.
        Returns surface_id.
        """
        qkey = claim.question_key or f"unknown_{claim.id}"

        if qkey in self.question_key_to_surface:
            # Existing surface for this question_key
            surface_id = self.question_key_to_surface[qkey]
            surface = self.surfaces[surface_id]

            # Detect relation to existing claims
            for existing_claim_id in surface.claim_ids:
                relation = self._detect_relation(claim, existing_claim_id)
                if relation:
                    if surface_id not in self.internal_relations:
                        self.internal_relations[surface_id] = []
                    self.internal_relations[surface_id].append(
                        (existing_claim_id, claim.id, relation)
                    )

            surface.claim_ids.add(claim.id)
            surface.entities.update(claim.entities)
        else:
            # Create new surface for this question_key
            surface_id = f"sf_{len(self.surfaces):04d}"
            surface = Surface(
                id=surface_id,
                claim_ids={claim.id},
                entities=claim.entities.copy(),
                anchor_entities=claim.anchor_entities.copy() if claim.anchor_entities else set(),
                question_key=qkey,
                formation_method="question_key",
            )
            self.surfaces[surface_id] = surface
            self.question_key_to_surface[qkey] = surface_id

        self.claim_to_surface[claim.id] = surface_id
        return surface_id

    def _detect_relation(self, new_claim: Claim, existing_claim_id: str) -> Relation:
        """Detect relation between claims (simplified)."""
        # In real system, this would use typed values
        if new_claim.has_update_language:
            return Relation.SUPERSEDES
        if new_claim.extracted_value:
            return Relation.REFINES
        return Relation.CONFIRMS

    def get_surface_for_question_key(self, qkey: str) -> Surface:
        """Get surface by question_key."""
        surface_id = self.question_key_to_surface.get(qkey)
        return self.surfaces.get(surface_id) if surface_id else None

    def get_conflicts(self, surface_id: str) -> list:
        """Get all CONFLICTS relations in a surface."""
        return [
            (c1, c2, r) for c1, c2, r in self.internal_relations.get(surface_id, [])
            if r == Relation.CONFLICTS
        ]


# ============================================================================
# TEST: SAME QUESTION_KEY → SAME SURFACE
# ============================================================================

class TestSameQuestionKeySameSurface:
    """Test that claims with same question_key go to same surface."""

    def test_death_toll_claims_same_surface(self, death_toll_claims):
        """All death toll claims should be in one surface."""
        kernel = L2SurfaceKernel()

        surface_ids = set()
        for claim in death_toll_claims:
            sid = kernel.process_claim(claim)
            surface_ids.add(sid)

        assert len(surface_ids) == 1, \
            f"Same question_key should produce 1 surface, got {len(surface_ids)}"

    def test_surface_has_all_claims(self, death_toll_claims):
        """Surface should contain all claims with that question_key."""
        kernel = L2SurfaceKernel()

        for claim in death_toll_claims:
            kernel.process_claim(claim)

        surface = kernel.get_surface_for_question_key("death_toll_hong_kong_fire_2025")
        assert surface is not None
        assert len(surface.claim_ids) == 3
        assert all(c.id in surface.claim_ids for c in death_toll_claims)

    def test_surface_question_key_set(self, death_toll_claims):
        """Surface should have question_key property set."""
        kernel = L2SurfaceKernel()

        for claim in death_toll_claims:
            kernel.process_claim(claim)

        surface = kernel.get_surface_for_question_key("death_toll_hong_kong_fire_2025")
        assert surface.question_key == "death_toll_hong_kong_fire_2025"

    def test_surface_formation_method(self, death_toll_claims):
        """Surface should have formation_method = question_key."""
        kernel = L2SurfaceKernel()

        for claim in death_toll_claims:
            kernel.process_claim(claim)

        surface = kernel.get_surface_for_question_key("death_toll_hong_kong_fire_2025")
        assert surface.formation_method == "question_key"


# ============================================================================
# TEST: DIFFERENT QUESTION_KEY → DIFFERENT SURFACES
# ============================================================================

class TestDifferentQuestionKeyDifferentSurfaces:
    """Test that different question_keys produce different surfaces."""

    def test_death_toll_vs_cause_different_surfaces(
        self, death_toll_claims, fire_cause_claims
    ):
        """Death toll and fire cause should be separate surfaces."""
        kernel = L2SurfaceKernel()

        for claim in death_toll_claims + fire_cause_claims:
            kernel.process_claim(claim)

        assert len(kernel.surfaces) == 2, \
            f"Different question_keys should produce 2 surfaces, got {len(kernel.surfaces)}"

    def test_entity_overlap_does_not_merge(
        self, death_toll_claims, fire_cause_claims
    ):
        """Entity overlap should NOT cause surface merge."""
        kernel = L2SurfaceKernel()

        for claim in death_toll_claims + fire_cause_claims:
            kernel.process_claim(claim)

        death_surface = kernel.get_surface_for_question_key("death_toll_hong_kong_fire_2025")
        cause_surface = kernel.get_surface_for_question_key("fire_cause_tai_po_2025")

        # Both surfaces have overlapping entities
        shared_entities = death_surface.entities & cause_surface.entities
        assert len(shared_entities) > 0, "Test setup: surfaces should share entities"

        # But they're still separate surfaces
        assert death_surface.id != cause_surface.id

    def test_same_anchor_different_question_key(
        self, death_toll_claims, fire_cause_claims
    ):
        """Same anchor entities but different question_key → separate surfaces."""
        kernel = L2SurfaceKernel()

        for claim in death_toll_claims + fire_cause_claims:
            kernel.process_claim(claim)

        death_surface = kernel.get_surface_for_question_key("death_toll_hong_kong_fire_2025")
        cause_surface = kernel.get_surface_for_question_key("fire_cause_tai_po_2025")

        # Both have "Hong Kong Fire 2025" as anchor
        assert "Hong Kong Fire 2025" in death_surface.anchor_entities
        assert "Hong Kong Fire 2025" in cause_surface.anchor_entities

        # But they're still separate surfaces
        assert death_surface.id != cause_surface.id


# ============================================================================
# TEST: CONFLICTS STAY INSIDE SURFACE
# ============================================================================

class TestConflictsInsideSurface:
    """Test that conflicting claims remain in the same surface."""

    def test_conflicting_values_same_surface(self, conflicting_claims):
        """Conflicting values with same question_key → same surface."""
        kernel = L2SurfaceKernel()

        surface_ids = set()
        for claim in conflicting_claims:
            sid = kernel.process_claim(claim)
            surface_ids.add(sid)

        assert len(surface_ids) == 1, \
            "Conflicting claims should be in same surface (same question_key)"

    def test_conflict_does_not_split_surface(self, conflicting_claims):
        """A conflict should NOT cause surface split."""
        kernel = L2SurfaceKernel()

        # Process first claim
        kernel.process_claim(conflicting_claims[0])

        # Process conflicting claim
        kernel.process_claim(conflicting_claims[1])

        # Still only 1 surface
        assert len(kernel.surfaces) == 1

    def test_surface_entities_accumulate(self, conflicting_claims):
        """Surface entities should include all claims' entities."""
        kernel = L2SurfaceKernel()

        for claim in conflicting_claims:
            kernel.process_claim(claim)

        surface = list(kernel.surfaces.values())[0]

        # Should have entities from both claims
        assert "Hong Kong Fire" in surface.entities
        assert "Queen Elizabeth Hospital" in surface.entities


# ============================================================================
# TEST: MOTIFS ARE L3 EVIDENCE, NOT L2 IDENTITY
# ============================================================================

class TestMotifsAreL3Evidence:
    """Test that motifs don't define L2 surface identity."""

    def test_motif_ids_do_not_affect_surface_routing(self):
        """Claims should route by question_key, not motif membership."""
        kernel = L2SurfaceKernel()

        # Two claims with different motifs but same question_key
        claim1 = Claim(
            id="C100",
            text="Test claim 1",
            source="source1",
            entities={"Entity A", "Entity B"},
            question_key="same_question",
        )
        claim2 = Claim(
            id="C101",
            text="Test claim 2",
            source="source2",
            entities={"Entity C", "Entity D"},  # Different entities/motif!
            question_key="same_question",  # Same question_key!
        )

        sid1 = kernel.process_claim(claim1)
        sid2 = kernel.process_claim(claim2)

        assert sid1 == sid2, "Same question_key → same surface regardless of motif"

    def test_surface_motif_ids_are_evidence(self):
        """Surface.motif_ids is for L3 linking, not L2 identity."""
        surface = Surface(
            id="sf_test",
            claim_ids={"c1", "c2"},
            question_key="test_question",
            formation_method="question_key",
            motif_ids={"mtf_001", "mtf_002"},  # Motifs are metadata
        )

        # Motif IDs are stored but don't define identity
        assert surface.question_key == "test_question"
        assert len(surface.motif_ids) == 2

        # Two surfaces with same question_key would merge even with different motifs


# ============================================================================
# TEST: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases for surface identity."""

    def test_missing_question_key_uses_claim_id(self):
        """Claims without question_key get unique surfaces."""
        kernel = L2SurfaceKernel()

        claim1 = Claim(id="C200", text="Claim 1", source="s1", entities=set())
        claim2 = Claim(id="C201", text="Claim 2", source="s2", entities=set())

        sid1 = kernel.process_claim(claim1)
        sid2 = kernel.process_claim(claim2)

        # Different claims without question_key → different surfaces
        assert sid1 != sid2

    def test_empty_entities_still_routes_by_question_key(self):
        """Claims with empty entities still route by question_key."""
        kernel = L2SurfaceKernel()

        claim1 = Claim(
            id="C300",
            text="Claim with no entities",
            source="s1",
            entities=set(),
            question_key="sparse_question",
        )
        claim2 = Claim(
            id="C301",
            text="Another claim with no entities",
            source="s2",
            entities=set(),
            question_key="sparse_question",
        )

        sid1 = kernel.process_claim(claim1)
        sid2 = kernel.process_claim(claim2)

        assert sid1 == sid2, "Same question_key → same surface even with no entities"

    def test_question_key_is_case_sensitive(self):
        """Question keys are case-sensitive."""
        kernel = L2SurfaceKernel()

        claim1 = Claim(
            id="C400",
            text="Claim 1",
            source="s1",
            entities=set(),
            question_key="Death_Toll",
        )
        claim2 = Claim(
            id="C401",
            text="Claim 2",
            source="s2",
            entities=set(),
            question_key="death_toll",  # Different case!
        )

        sid1 = kernel.process_claim(claim1)
        sid2 = kernel.process_claim(claim2)

        assert sid1 != sid2, "Question keys are case-sensitive"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
