"""
Tests for runtime invariant guards.

These tests verify that:
1. Spine edges are rejected without MembraneDecision provenance
2. Edge types must match their action classification
3. SpineEdgeGuard correctly validates batches
"""

import pytest
from reee.compiler.membrane import Action, EdgeType, MembraneDecision, Referent, ReferentRole
from reee.compiler.guards import (
    validate_edge_for_persistence,
    validate_decision_for_persistence,
    SpineEdgeGuard,
    is_spine_edge,
    is_metabolic_edge,
    SPINE_EDGE_TYPES,
    METABOLIC_EDGE_TYPES,
    UnauthorizedSpineEdgeError,
    MissingProvenanceError,
    ActionEdgeMismatchError,
    SpineEdgeViolationError,
)


# Helper to create a test referent
def make_referent(entity_id: str, role: ReferentRole = ReferentRole.PLACE) -> Referent:
    return Referent(entity_id=entity_id, role=role)


# Helper to create test decisions
def make_decision(
    action: Action,
    edge_type: EdgeType = None,
    confidence: float = 0.9,
    reason: str = "test",
    shared_referents: frozenset = None,
) -> MembraneDecision:
    if shared_referents is None:
        shared_referents = frozenset()
    return MembraneDecision(
        edge_type=edge_type,
        action=action,
        confidence=confidence,
        reason=reason,
        shared_referents=shared_referents,
    )


# =============================================================================
# Edge Classification Tests
# =============================================================================

def test_spine_edge_classification():
    """Verify SAME_HAPPENING and UPDATE_TO are spine edges."""
    assert is_spine_edge(EdgeType.SAME_HAPPENING)
    assert is_spine_edge(EdgeType.UPDATE_TO)
    assert not is_spine_edge(EdgeType.CONTEXT_FOR)


def test_metabolic_edge_classification():
    """Verify CONTEXT_FOR is metabolic edge."""
    assert is_metabolic_edge(EdgeType.CONTEXT_FOR)
    assert not is_metabolic_edge(EdgeType.SAME_HAPPENING)
    assert not is_metabolic_edge(EdgeType.UPDATE_TO)


def test_edge_type_sets_are_disjoint():
    """Spine and metabolic edge sets must not overlap."""
    overlap = SPINE_EDGE_TYPES & METABOLIC_EDGE_TYPES
    assert len(overlap) == 0, f"Overlap found: {overlap}"


# =============================================================================
# validate_edge_for_persistence Tests
# =============================================================================

def test_spine_edge_requires_decision():
    """Spine edge without decision should fail."""
    with pytest.raises(MissingProvenanceError):
        validate_edge_for_persistence(
            edge_type=EdgeType.SAME_HAPPENING,
            decision=None,
            incident_a="in_abc",
            incident_b="in_def",
        )


def test_spine_edge_requires_merge_action():
    """Spine edge with non-MERGE decision should fail."""
    decision = make_decision(
        action=Action.PERIPHERY,
        edge_type=EdgeType.CONTEXT_FOR,
        confidence=0.8,
    )
    with pytest.raises(UnauthorizedSpineEdgeError):
        validate_edge_for_persistence(
            edge_type=EdgeType.SAME_HAPPENING,
            decision=decision,
        )


def test_spine_edge_type_must_match_decision():
    """Edge type must match the decision's edge_type."""
    decision = make_decision(
        action=Action.MERGE,
        edge_type=EdgeType.UPDATE_TO,  # Decision says UPDATE_TO
        confidence=0.9,
    )
    with pytest.raises(ActionEdgeMismatchError):
        validate_edge_for_persistence(
            edge_type=EdgeType.SAME_HAPPENING,  # But we're persisting SAME_HAPPENING
            decision=decision,
        )


def test_valid_spine_edge_passes():
    """Valid spine edge with proper decision should pass."""
    ref = make_referent("location_abc", ReferentRole.PLACE)
    decision = make_decision(
        action=Action.MERGE,
        edge_type=EdgeType.SAME_HAPPENING,
        confidence=0.9,
        reason="matching referent: location_abc",
        shared_referents=frozenset([ref]),
    )
    # Should not raise
    validate_edge_for_persistence(
        edge_type=EdgeType.SAME_HAPPENING,
        decision=decision,
        incident_a="in_abc",
        incident_b="in_def",
    )


def test_metabolic_edge_without_decision_passes():
    """Metabolic edges don't require decisions."""
    # Should not raise
    validate_edge_for_persistence(
        edge_type=EdgeType.CONTEXT_FOR,
        decision=None,
    )


def test_metabolic_edge_with_merge_decision_fails():
    """Metabolic edge cannot come from MERGE action."""
    decision = make_decision(
        action=Action.MERGE,
        edge_type=EdgeType.SAME_HAPPENING,
        confidence=0.9,
    )
    with pytest.raises(ActionEdgeMismatchError):
        validate_edge_for_persistence(
            edge_type=EdgeType.CONTEXT_FOR,
            decision=decision,
        )


# =============================================================================
# validate_decision_for_persistence Tests
# =============================================================================

def test_merge_decision_must_have_spine_edge_type():
    """MERGE decisions must specify spine edge types."""
    decision = make_decision(
        action=Action.MERGE,
        edge_type=EdgeType.CONTEXT_FOR,  # Wrong - metabolic type
        confidence=0.9,
    )
    with pytest.raises(ActionEdgeMismatchError):
        validate_decision_for_persistence(decision)


def test_merge_decision_confidence_bounds():
    """MERGE confidence must be in [0, 1]."""
    decision = make_decision(
        action=Action.MERGE,
        edge_type=EdgeType.SAME_HAPPENING,
        confidence=1.5,  # Invalid
    )
    with pytest.raises(SpineEdgeViolationError):
        validate_decision_for_persistence(decision)


def test_periphery_decision_cannot_have_spine_edge():
    """PERIPHERY decisions cannot have spine edge types."""
    decision = make_decision(
        action=Action.PERIPHERY,
        edge_type=EdgeType.SAME_HAPPENING,  # Wrong - spine type
        confidence=0.8,
    )
    with pytest.raises(ActionEdgeMismatchError):
        validate_decision_for_persistence(decision)


def test_valid_decisions_pass():
    """Valid decisions should pass validation."""
    # Valid MERGE
    merge_decision = make_decision(
        action=Action.MERGE,
        edge_type=EdgeType.SAME_HAPPENING,
        confidence=0.9,
    )
    validate_decision_for_persistence(merge_decision)

    # Valid PERIPHERY
    periphery_decision = make_decision(
        action=Action.PERIPHERY,
        edge_type=EdgeType.CONTEXT_FOR,
        confidence=0.7,
    )
    validate_decision_for_persistence(periphery_decision)

    # Valid DEFER
    defer_decision = make_decision(
        action=Action.DEFER,
        edge_type=None,
        confidence=0.6,
    )
    validate_decision_for_persistence(defer_decision)


# =============================================================================
# SpineEdgeGuard Tests
# =============================================================================

def test_guard_validates_all_edges():
    """Guard should validate all registered edges."""
    guard = SpineEdgeGuard(strict=True)

    # Register valid edge
    valid_decision = make_decision(
        action=Action.MERGE,
        edge_type=EdgeType.SAME_HAPPENING,
        confidence=0.9,
    )
    guard.register_edge(
        incident_a="in_abc",
        incident_b="in_def",
        edge_type=EdgeType.SAME_HAPPENING,
        decision=valid_decision,
    )

    # Should pass
    assert guard.validate_all() is True


def test_guard_catches_violations_strict():
    """Guard in strict mode should raise on violation."""
    guard = SpineEdgeGuard(strict=True)

    # Register invalid edge (no decision for spine edge)
    guard.register_edge(
        incident_a="in_abc",
        incident_b="in_def",
        edge_type=EdgeType.SAME_HAPPENING,
        decision=None,  # Missing!
    )

    with pytest.raises(MissingProvenanceError):
        guard.validate_all()


def test_guard_collects_violations_non_strict():
    """Guard in non-strict mode should collect violations."""
    guard = SpineEdgeGuard(strict=False)

    # Register multiple invalid edges
    guard.register_edge(
        incident_a="in_abc",
        incident_b="in_def",
        edge_type=EdgeType.SAME_HAPPENING,
        decision=None,
    )
    guard.register_edge(
        incident_a="in_ghi",
        incident_b="in_jkl",
        edge_type=EdgeType.UPDATE_TO,
        decision=None,
    )

    # Should not raise, but return False
    assert guard.validate_all() is False
    assert len(guard.get_violations()) == 2


def test_guard_context_manager():
    """Guard should work as context manager."""
    with SpineEdgeGuard() as guard:
        valid_decision = make_decision(
            action=Action.MERGE,
            edge_type=EdgeType.SAME_HAPPENING,
            confidence=0.9,
        )
        guard.register_edge(
            incident_a="in_abc",
            incident_b="in_def",
            edge_type=EdgeType.SAME_HAPPENING,
            decision=valid_decision,
        )
        guard.validate_all()  # Should not raise


def test_guard_empty_batch():
    """Guard should pass with no edges registered."""
    guard = SpineEdgeGuard()
    assert guard.validate_all() is True
    assert len(guard.get_violations()) == 0


# =============================================================================
# Integration Tests
# =============================================================================

def test_full_workflow():
    """Test typical workflow: compile → validate → (persist)."""
    # Simulate compilation result
    ref = make_referent("wang_fuk_court", ReferentRole.PLACE)
    decisions = [
        make_decision(
            action=Action.MERGE,
            edge_type=EdgeType.SAME_HAPPENING,
            confidence=0.95,
            reason="shared location witness",
            shared_referents=frozenset([ref]),
        ),
        make_decision(
            action=Action.PERIPHERY,
            edge_type=EdgeType.CONTEXT_FOR,
            confidence=0.7,
            reason="shared context only",
        ),
    ]

    # Validate before persisting
    with SpineEdgeGuard() as guard:
        # Register spine edge
        guard.register_edge(
            incident_a="in_001",
            incident_b="in_002",
            edge_type=EdgeType.SAME_HAPPENING,
            decision=decisions[0],
        )

        # Metabolic edge doesn't need registration (no decision required)
        # but if we have one, validate it too
        guard.register_edge(
            incident_a="in_003",
            incident_b="in_004",
            edge_type=EdgeType.CONTEXT_FOR,
            decision=decisions[1],
        )

        # All valid
        assert guard.validate_all() is True


# =============================================================================
# Metabolic Edge Isolation Tests
# =============================================================================

from reee.compiler.guards import (
    MetabolicMembershipViolationError,
    assert_metabolic_edges_not_in_membership,
    validate_case_spine_only,
)
from reee.compiler import CompiledEdge


def make_compiled_edge(
    incident_a: str,
    incident_b: str,
    action: Action,
    edge_type: EdgeType,
    confidence: float = 0.8,
) -> CompiledEdge:
    """Helper to create CompiledEdge for testing."""
    decision = make_decision(
        action=action,
        edge_type=edge_type,
        confidence=confidence,
    )
    return CompiledEdge(
        incident_a=incident_a,
        incident_b=incident_b,
        decision=decision,
        artifact_hash_a=f"hash_{incident_a}",
        artifact_hash_b=f"hash_{incident_b}",
    )


def test_validate_case_spine_only_with_valid_spine():
    """Case with spine edges should pass validation."""
    spine_edges = [
        make_compiled_edge("in_001", "in_002", Action.MERGE, EdgeType.SAME_HAPPENING),
        make_compiled_edge("in_002", "in_003", Action.MERGE, EdgeType.SAME_HAPPENING),
    ]
    case_ids = {"in_001", "in_002", "in_003"}

    # Should pass
    assert validate_case_spine_only(case_ids, spine_edges) is True


def test_validate_case_spine_only_single_incident():
    """Single incident case needs no spine edges."""
    case_ids = {"in_001"}

    # Single incident is valid with no spine edges
    assert validate_case_spine_only(case_ids, []) is True


def test_validate_case_spine_only_empty_case():
    """Empty case should pass."""
    assert validate_case_spine_only(set(), []) is True


def test_validate_case_spine_only_fails_multi_incident_no_spine():
    """Multiple incidents without spine edges should fail."""
    case_ids = {"in_001", "in_002", "in_003"}

    with pytest.raises(MetabolicMembershipViolationError):
        validate_case_spine_only(case_ids, [])


def test_validate_case_spine_only_fails_disconnected():
    """Spine edges that don't connect all incidents should fail."""
    # in_001 and in_002 are connected, but in_003 is isolated
    spine_edges = [
        make_compiled_edge("in_001", "in_002", Action.MERGE, EdgeType.SAME_HAPPENING),
    ]
    case_ids = {"in_001", "in_002", "in_003"}

    with pytest.raises(MetabolicMembershipViolationError):
        validate_case_spine_only(case_ids, spine_edges)


def test_assert_metabolic_not_in_membership_passes():
    """Metabolic edges that don't affect membership should pass."""
    spine_edges = [
        make_compiled_edge("in_001", "in_002", Action.MERGE, EdgeType.SAME_HAPPENING),
    ]
    metabolic_edges = [
        make_compiled_edge("in_002", "in_003", Action.PERIPHERY, EdgeType.CONTEXT_FOR),
    ]
    case_ids = {"in_001", "in_002"}  # Only includes spine-connected incidents

    # Should not raise
    assert_metabolic_edges_not_in_membership(spine_edges, metabolic_edges, case_ids)


def test_assert_metabolic_not_in_membership_fails_when_metabolic_bridges():
    """Metabolic edge bridging non-spine-connected incidents should fail."""
    spine_edges = [
        make_compiled_edge("in_001", "in_002", Action.MERGE, EdgeType.SAME_HAPPENING),
    ]
    metabolic_edges = [
        # This metabolic edge connects in_003 to in_002, but in_003 is not in spine
        make_compiled_edge("in_003", "in_002", Action.PERIPHERY, EdgeType.CONTEXT_FOR),
    ]
    # If in_003 is in the case but not spine-connected, it suggests metabolic was used
    case_ids = {"in_001", "in_002", "in_003"}

    with pytest.raises(MetabolicMembershipViolationError):
        assert_metabolic_edges_not_in_membership(spine_edges, metabolic_edges, case_ids)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
