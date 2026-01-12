"""
Tests for Membrane Compiler

These tests verify the deterministic membrane rules:
1. 'Hong Kong + Tai Po does not open spine' - broad locations don't form spine
2. 'Wang Fuk Court as referent_facility opens spine' - specific place forms spine
3. 'Person-only overlap without witness downgrades to metabolic'
4. 'Low confidence forces DEFER'
"""

import pytest
from reee.compiler.membrane import (
    Action,
    EdgeType,
    ReferentRole,
    Referent,
    IncidentArtifact,
    MembraneDecision,
    compile_pair,
    assert_invariants,
    CONFIDENCE_THRESHOLD_MERGE,
    CONFIDENCE_THRESHOLD_PERIPHERY,
    TIME_WITNESS_WINDOW_SECONDS,
)


# =============================================================================
# Test Fixtures
# =============================================================================

def make_artifact(
    incident_id: str,
    referents: list[tuple[str, ReferentRole]] = None,
    contexts: set[str] = None,
    time_start: float = None,
    confidence: float = 0.9,
) -> IncidentArtifact:
    """Helper to create test artifacts."""
    refs = frozenset(
        Referent(entity_id=eid, role=role)
        for eid, role in (referents or [])
    )
    return IncidentArtifact(
        incident_id=incident_id,
        referents=refs,
        contexts=frozenset(contexts or set()),
        time_start=time_start,
        confidence=confidence,
    )


# =============================================================================
# Test 1: Broad locations don't form spine (context-only)
# =============================================================================

class TestBroadLocationsNoSpine:
    """Hong Kong + Tai Po as context does not open spine."""

    def test_shared_context_only_no_spine(self):
        """Two incidents sharing only context entities → no spine edge."""
        art1 = make_artifact(
            "incident_1",
            referents=[],  # No referents
            contexts={"Hong Kong", "Tai Po"},
        )
        art2 = make_artifact(
            "incident_2",
            referents=[],  # No referents
            contexts={"Hong Kong", "Tai Po"},
        )

        decision = compile_pair(art1, art2)

        # Should be CONTEXT_FOR edge (metabolic), not spine
        assert decision.edge_type == EdgeType.CONTEXT_FOR
        assert decision.action == Action.PERIPHERY
        assert not decision.shared_referents
        assert_invariants(decision)

    def test_no_overlap_at_all(self):
        """Two incidents with no shared entities → UNRELATED."""
        art1 = make_artifact(
            "incident_1",
            referents=[("person_A", ReferentRole.PERSON)],
            contexts={"Hong Kong"},
        )
        art2 = make_artifact(
            "incident_2",
            referents=[("person_B", ReferentRole.PERSON)],
            contexts={"Beijing"},
        )

        decision = compile_pair(art1, art2)

        assert decision.edge_type == EdgeType.UNRELATED
        assert decision.action == Action.DEFER
        assert not decision.shared_referents
        assert_invariants(decision)


# =============================================================================
# Test 2: Specific place (FACILITY) opens spine
# =============================================================================

class TestFacilityOpensSpine:
    """Wang Fuk Court as referent PLACE opens spine edge."""

    def test_shared_place_opens_spine(self):
        """Two incidents sharing a PLACE referent → spine edge."""
        art1 = make_artifact(
            "incident_1",
            referents=[("Wang Fuk Court", ReferentRole.PLACE)],
            confidence=0.9,
        )
        art2 = make_artifact(
            "incident_2",
            referents=[("Wang Fuk Court", ReferentRole.PLACE)],
            confidence=0.9,
        )

        decision = compile_pair(art1, art2)

        # Should be spine edge with MERGE action
        assert decision.edge_type in (EdgeType.SAME_HAPPENING, EdgeType.UPDATE_TO)
        assert decision.action == Action.MERGE
        assert len(decision.shared_referents) == 1
        assert any(r.entity_id == "Wang Fuk Court" for r in decision.shared_referents)
        assert_invariants(decision)

    def test_shared_object_opens_spine(self):
        """Two incidents sharing an OBJECT referent → spine edge."""
        art1 = make_artifact(
            "incident_1",
            referents=[("Flight MH370", ReferentRole.OBJECT)],
            confidence=0.8,
        )
        art2 = make_artifact(
            "incident_2",
            referents=[("Flight MH370", ReferentRole.OBJECT)],
            confidence=0.8,
        )

        decision = compile_pair(art1, art2)

        assert decision.edge_type in (EdgeType.SAME_HAPPENING, EdgeType.UPDATE_TO)
        assert decision.action == Action.MERGE
        assert_invariants(decision)

    def test_temporal_classification_same_happening(self):
        """Incidents within 24h → SAME_HAPPENING."""
        now = 1700000000.0
        art1 = make_artifact(
            "incident_1",
            referents=[("Building X", ReferentRole.PLACE)],
            time_start=now,
        )
        art2 = make_artifact(
            "incident_2",
            referents=[("Building X", ReferentRole.PLACE)],
            time_start=now + 3600,  # 1 hour later
        )

        decision = compile_pair(art1, art2)

        assert decision.edge_type == EdgeType.SAME_HAPPENING
        assert_invariants(decision)

    def test_temporal_classification_update_to(self):
        """Incidents > 24h apart → UPDATE_TO."""
        now = 1700000000.0
        art1 = make_artifact(
            "incident_1",
            referents=[("Building X", ReferentRole.PLACE)],
            time_start=now,
        )
        art2 = make_artifact(
            "incident_2",
            referents=[("Building X", ReferentRole.PLACE)],
            time_start=now + 48 * 3600,  # 48 hours later
        )

        decision = compile_pair(art1, art2)

        assert decision.edge_type == EdgeType.UPDATE_TO
        assert_invariants(decision)


# =============================================================================
# Test 3: Person-only overlap without witness → metabolic
# =============================================================================

class TestPersonOnlyWitness:
    """Person-only overlap requires time witness for spine."""

    def test_person_only_no_witness_metabolic(self):
        """Person overlap without time witness → PHASE_OF (metabolic)."""
        art1 = make_artifact(
            "incident_1",
            referents=[("Donald Trump", ReferentRole.PERSON)],
            time_start=None,  # No time
        )
        art2 = make_artifact(
            "incident_2",
            referents=[("Donald Trump", ReferentRole.PERSON)],
            time_start=None,  # No time
        )

        decision = compile_pair(art1, art2)

        # Should be metabolic (PHASE_OF), not spine
        assert decision.edge_type == EdgeType.PHASE_OF
        assert decision.action == Action.PERIPHERY
        assert decision.shared_referents  # Still has shared referents
        assert_invariants(decision)

    def test_person_only_far_apart_metabolic(self):
        """Person overlap with times > 7 days apart → metabolic."""
        now = 1700000000.0
        art1 = make_artifact(
            "incident_1",
            referents=[("Donald Trump", ReferentRole.PERSON)],
            time_start=now,
        )
        art2 = make_artifact(
            "incident_2",
            referents=[("Donald Trump", ReferentRole.PERSON)],
            time_start=now + 30 * 24 * 3600,  # 30 days later
        )

        decision = compile_pair(art1, art2)

        assert decision.edge_type == EdgeType.PHASE_OF
        assert decision.action == Action.PERIPHERY
        assert_invariants(decision)

    def test_person_with_time_witness_spine(self):
        """Person overlap within 7 days → spine edge."""
        now = 1700000000.0
        art1 = make_artifact(
            "incident_1",
            referents=[("Donald Trump", ReferentRole.PERSON)],
            time_start=now,
        )
        art2 = make_artifact(
            "incident_2",
            referents=[("Donald Trump", ReferentRole.PERSON)],
            time_start=now + 3 * 24 * 3600,  # 3 days later
        )

        decision = compile_pair(art1, art2)

        # With time witness, person CAN form spine
        assert decision.edge_type in (EdgeType.SAME_HAPPENING, EdgeType.UPDATE_TO)
        assert decision.action == Action.MERGE
        assert_invariants(decision)

    def test_person_plus_place_spine_regardless(self):
        """Person + PLACE overlap → spine (place takes precedence)."""
        art1 = make_artifact(
            "incident_1",
            referents=[
                ("Donald Trump", ReferentRole.PERSON),
                ("White House", ReferentRole.PLACE),
            ],
            time_start=None,  # No time
        )
        art2 = make_artifact(
            "incident_2",
            referents=[
                ("Donald Trump", ReferentRole.PERSON),
                ("White House", ReferentRole.PLACE),
            ],
            time_start=None,  # No time
        )

        decision = compile_pair(art1, art2)

        # Place overlap overrides person-only rule
        assert decision.edge_type in (EdgeType.SAME_HAPPENING, EdgeType.UPDATE_TO)
        assert decision.action == Action.MERGE
        assert_invariants(decision)


# =============================================================================
# Test 4: Low confidence forces DEFER
# =============================================================================

class TestLowConfidenceDefer:
    """Low confidence artifacts force DEFER action."""

    def test_low_confidence_defer(self):
        """Confidence below threshold → DEFER."""
        art1 = make_artifact(
            "incident_1",
            referents=[("Wang Fuk Court", ReferentRole.PLACE)],
            confidence=0.3,  # Below CONFIDENCE_THRESHOLD_PERIPHERY (0.4)
        )
        art2 = make_artifact(
            "incident_2",
            referents=[("Wang Fuk Court", ReferentRole.PLACE)],
            confidence=0.9,
        )

        decision = compile_pair(art1, art2)

        assert decision.action == Action.DEFER
        assert decision.edge_type == EdgeType.UNRELATED
        assert decision.confidence < CONFIDENCE_THRESHOLD_PERIPHERY
        assert_invariants(decision)

    def test_medium_confidence_periphery(self):
        """Confidence between thresholds → PERIPHERY (not MERGE)."""
        art1 = make_artifact(
            "incident_1",
            referents=[("Wang Fuk Court", ReferentRole.PLACE)],
            confidence=0.5,  # Between 0.4 and 0.7
        )
        art2 = make_artifact(
            "incident_2",
            referents=[("Wang Fuk Court", ReferentRole.PLACE)],
            confidence=0.5,
        )

        decision = compile_pair(art1, art2)

        # Should be spine edge type but PERIPHERY action
        assert decision.edge_type in (EdgeType.SAME_HAPPENING, EdgeType.UPDATE_TO)
        assert decision.action == Action.PERIPHERY
        assert_invariants(decision)

    def test_high_confidence_merge(self):
        """Confidence above merge threshold → MERGE."""
        art1 = make_artifact(
            "incident_1",
            referents=[("Wang Fuk Court", ReferentRole.PLACE)],
            confidence=0.9,
        )
        art2 = make_artifact(
            "incident_2",
            referents=[("Wang Fuk Court", ReferentRole.PLACE)],
            confidence=0.8,
        )

        decision = compile_pair(art1, art2)

        assert decision.action == Action.MERGE
        assert decision.confidence >= CONFIDENCE_THRESHOLD_MERGE
        assert_invariants(decision)


# =============================================================================
# Invariant Tests
# =============================================================================

class TestInvariants:
    """Test that invariants catch violations."""

    def test_invariant_spine_requires_referents(self):
        """Spine edge without shared referents should fail invariant."""
        bad_decision = MembraneDecision(
            edge_type=EdgeType.SAME_HAPPENING,
            action=Action.MERGE,
            confidence=0.9,
            reason="Test",
            shared_referents=frozenset(),  # Empty!
        )

        with pytest.raises(AssertionError, match="shared referents"):
            assert_invariants(bad_decision)

    def test_invariant_metabolic_no_merge(self):
        """Metabolic edge with MERGE action should fail invariant."""
        bad_decision = MembraneDecision(
            edge_type=EdgeType.CAUSES,
            action=Action.MERGE,  # Invalid for metabolic
            confidence=0.9,
            reason="Test",
            shared_referents=frozenset([Referent("x", ReferentRole.PLACE)]),
        )

        with pytest.raises(AssertionError, match="cannot MERGE"):
            assert_invariants(bad_decision)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_exactly_at_merge_threshold(self):
        """Confidence exactly at MERGE threshold → MERGE."""
        art1 = make_artifact(
            "incident_1",
            referents=[("X", ReferentRole.PLACE)],
            confidence=CONFIDENCE_THRESHOLD_MERGE,  # Exactly 0.7
        )
        art2 = make_artifact(
            "incident_2",
            referents=[("X", ReferentRole.PLACE)],
            confidence=CONFIDENCE_THRESHOLD_MERGE,
        )

        decision = compile_pair(art1, art2)
        assert decision.action == Action.MERGE

    def test_exactly_at_periphery_threshold(self):
        """Confidence exactly at PERIPHERY threshold → PERIPHERY."""
        art1 = make_artifact(
            "incident_1",
            referents=[("X", ReferentRole.PLACE)],
            confidence=CONFIDENCE_THRESHOLD_PERIPHERY,  # Exactly 0.4
        )
        art2 = make_artifact(
            "incident_2",
            referents=[("X", ReferentRole.PLACE)],
            confidence=CONFIDENCE_THRESHOLD_PERIPHERY,
        )

        decision = compile_pair(art1, art2)
        assert decision.action == Action.PERIPHERY

    def test_time_witness_exactly_at_boundary(self):
        """Time difference exactly at 7-day boundary."""
        now = 1700000000.0
        art1 = make_artifact(
            "incident_1",
            referents=[("Person", ReferentRole.PERSON)],
            time_start=now,
        )
        art2 = make_artifact(
            "incident_2",
            referents=[("Person", ReferentRole.PERSON)],
            time_start=now + TIME_WITNESS_WINDOW_SECONDS,  # Exactly 7 days
        )

        decision = compile_pair(art1, art2)
        # At boundary should still count as witness
        assert decision.action == Action.MERGE

    def test_mixed_referent_roles(self):
        """Multiple referent types, only some overlap."""
        art1 = make_artifact(
            "incident_1",
            referents=[
                ("Building X", ReferentRole.PLACE),
                ("Person A", ReferentRole.PERSON),
                ("Vehicle Y", ReferentRole.OBJECT),
            ],
        )
        art2 = make_artifact(
            "incident_2",
            referents=[
                ("Building X", ReferentRole.PLACE),  # Overlaps
                ("Person B", ReferentRole.PERSON),   # Different
            ],
        )

        decision = compile_pair(art1, art2)

        # PLACE overlap → spine
        assert decision.action == Action.MERGE
        assert len(decision.shared_referents) == 1
        assert any(r.entity_id == "Building X" for r in decision.shared_referents)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
