"""
Tests for L4 Authority: Weaver owns structure, canonical worker owns presentation.

Key invariant (release criterion):
    "canonical worker must be able to delete and rerun without changing
    any case membership—only presentation changes."

This means:
1. Only weaver can create/modify CONTAINS relationships
2. Canonical worker can only modify presentation fields
3. verify_case_membership_unchanged should pass after canonical worker runs
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from reee.compiler.l4_authority import (
    CaseChangedEvent,
    CasePresentationUpdate,
    persist_case_structure,
    persist_metabolic_edges,
    emit_case_changed_event,
    persist_case_presentation,
    get_case_for_enrichment,
    verify_case_membership_unchanged,
)
from reee.compiler.membrane import Action, EdgeType, MembraneDecision, Referent, ReferentRole
from reee.compiler.weaver_compiler import Case as CompilerCase, CompilationResult, CompiledEdge


# =============================================================================
# CaseChangedEvent Tests
# =============================================================================

def test_case_changed_event_to_dict():
    """Event should serialize correctly."""
    event = CaseChangedEvent(
        event_type="case_formed",
        timestamp="2024-01-01T00:00:00Z",
        case_id="case_abc123",
        kernel_signature="case_abc123",
        incident_ids=["in_aaa", "in_bbb", "in_ccc"],
        core_entities=["Wang Fuk Court"],
        is_new=True,
    )

    d = event.to_dict()
    assert d["type"] == "case_formed"
    assert d["case_id"] == "case_abc123"
    assert len(d["incident_ids"]) == 3
    assert d["is_new"] is True


def test_case_changed_event_to_json():
    """Event should serialize to valid JSON."""
    event = CaseChangedEvent(
        event_type="case_updated",
        timestamp="2024-01-01T00:00:00Z",
        case_id="case_xyz789",
        kernel_signature="case_xyz789",
        incident_ids=["in_111"],
        core_entities=[],
        is_new=False,
    )

    json_str = event.to_json()
    assert '"type": "case_updated"' in json_str
    assert '"is_new": false' in json_str


# =============================================================================
# CasePresentationUpdate Tests
# =============================================================================

def test_presentation_update_fields():
    """Presentation update should only have presentation fields."""
    update = CasePresentationUpdate(
        case_id="case_abc",
        title="Hong Kong Fire",
        description="Fire at Wang Fuk Court",
        case_type="breaking",
        surface_count=10,
        source_count=5,
        claim_count=25,
    )

    assert update.case_id == "case_abc"
    assert update.title == "Hong Kong Fire"
    assert update.surface_count == 10

    # Should NOT have incident_ids or kernel_signature
    assert not hasattr(update, "incident_ids")


# =============================================================================
# persist_case_structure Tests (Weaver Authority)
# =============================================================================

@pytest.mark.asyncio
async def test_persist_case_structure_creates_contains_relationships():
    """Weaver should create CONTAINS relationships."""
    # Mock Neo4j
    mock_neo4j = AsyncMock()
    mock_neo4j._execute_read.return_value = []  # Case doesn't exist
    mock_neo4j._execute_write.return_value = None

    # Create test case (Case has: case_id, incident_ids (FrozenSet), spine_edges, metabolic_edges)
    case = CompilerCase(
        case_id="case_test",
        incident_ids=frozenset({"in_001", "in_002"}),
        spine_edges=[],
        metabolic_edges=[],
    )

    result = CompilationResult(
        cases=[case],
        all_edges=[],
        deferred=[],
        inquiries=[],
        stats={"compiler_version": "1.0.0"},
    )

    is_new = await persist_case_structure(mock_neo4j, case, result)

    # Should have called write
    assert mock_neo4j._execute_write.called

    # Check the query includes CONTAINS
    call_args = mock_neo4j._execute_write.call_args_list
    assert len(call_args) >= 1

    # First call should be the main case persist
    query = call_args[0][0][0]  # First positional arg is the query
    assert "CONTAINS" in query
    assert "incident_ids" in query


@pytest.mark.asyncio
async def test_persist_case_structure_returns_is_new():
    """Should return True for new cases, False for existing."""
    mock_neo4j = AsyncMock()

    case = CompilerCase(
        case_id="case_test",
        incident_ids=frozenset({"in_001"}),
        spine_edges=[],
        metabolic_edges=[],
    )
    result = CompilationResult(cases=[case], all_edges=[], deferred=[], inquiries=[], stats={})

    # Test new case
    mock_neo4j._execute_read.return_value = []
    is_new = await persist_case_structure(mock_neo4j, case, result)
    assert is_new is True

    # Test existing case
    mock_neo4j._execute_read.return_value = [{"id": "case_test"}]
    is_new = await persist_case_structure(mock_neo4j, case, result)
    assert is_new is False


# =============================================================================
# persist_case_presentation Tests (Canonical Worker Authority)
# =============================================================================

@pytest.mark.asyncio
async def test_persist_case_presentation_does_not_touch_contains():
    """Canonical worker MUST NOT modify CONTAINS relationships."""
    mock_neo4j = AsyncMock()
    mock_neo4j._execute_write.return_value = None

    update = CasePresentationUpdate(
        case_id="case_abc",
        title="Test Title",
        description="Test description",
        case_type="breaking",
    )

    await persist_case_presentation(mock_neo4j, update)

    # Check the query
    call_args = mock_neo4j._execute_write.call_args
    query = call_args[0][0]

    # CRITICAL: Query must NOT modify CONTAINS
    assert "DELETE" not in query.upper()
    assert "-[r:CONTAINS]->" not in query
    assert "UNWIND" not in query  # Used for creating relationships
    assert "incident_ids" not in call_args[0][1]  # Not in params either


@pytest.mark.asyncio
async def test_persist_case_presentation_updates_only_presentation_fields():
    """Should only update title, description, case_type, etc."""
    mock_neo4j = AsyncMock()

    update = CasePresentationUpdate(
        case_id="case_xyz",
        title="Updated Title",
        description="Updated desc",
        case_type="developing",
        surface_count=15,
        source_count=8,
        claim_count=42,
    )

    await persist_case_presentation(mock_neo4j, update)

    # Check params
    params = mock_neo4j._execute_write.call_args[0][1]
    assert params["title"] == "Updated Title"
    assert params["description"] == "Updated desc"
    assert params["case_type"] == "developing"
    assert params["surface_count"] == 15


# =============================================================================
# verify_case_membership_unchanged Tests
# =============================================================================

@pytest.mark.asyncio
async def test_verify_membership_unchanged_passes_when_identical():
    """Should return True when membership hasn't changed."""
    mock_neo4j = AsyncMock()
    mock_neo4j._execute_read.return_value = [
        {"incident_ids": ["in_001", "in_002", "in_003"]}
    ]

    expected = {"in_001", "in_002", "in_003"}
    result = await verify_case_membership_unchanged(mock_neo4j, "case_test", expected)

    assert result is True


@pytest.mark.asyncio
async def test_verify_membership_unchanged_fails_when_different():
    """Should return False when membership changed."""
    mock_neo4j = AsyncMock()
    mock_neo4j._execute_read.return_value = [
        {"incident_ids": ["in_001", "in_002", "in_NEW"]}  # in_NEW is unexpected
    ]

    expected = {"in_001", "in_002", "in_003"}
    result = await verify_case_membership_unchanged(mock_neo4j, "case_test", expected)

    assert result is False


@pytest.mark.asyncio
async def test_verify_membership_unchanged_fails_when_case_missing():
    """Should return False when case doesn't exist."""
    mock_neo4j = AsyncMock()
    mock_neo4j._execute_read.return_value = []  # No case found

    result = await verify_case_membership_unchanged(
        mock_neo4j, "case_nonexistent", {"in_001"}
    )

    assert result is False


# =============================================================================
# emit_case_changed_event Tests
# =============================================================================

@pytest.mark.asyncio
async def test_emit_case_changed_event_publishes():
    """Should publish event to Redis."""
    mock_redis = AsyncMock()

    case = CompilerCase(
        case_id="case_emit",
        incident_ids=frozenset({"in_a", "in_b"}),
        spine_edges=[],
        metabolic_edges=[],
    )

    await emit_case_changed_event(mock_redis, case, is_new=True)

    assert mock_redis.publish.called
    call_args = mock_redis.publish.call_args
    channel = call_args[0][0]
    message = call_args[0][1]

    assert channel == "weaver_events"
    assert "case_formed" in message


@pytest.mark.asyncio
async def test_emit_case_changed_event_handles_no_redis():
    """Should not fail when Redis is None."""
    case = CompilerCase(
        case_id="case_no_redis",
        incident_ids=frozenset(),
        spine_edges=[],
        metabolic_edges=[],
    )

    # Should not raise
    await emit_case_changed_event(None, case, is_new=True)


# =============================================================================
# Integration-like Tests
# =============================================================================

@pytest.mark.asyncio
async def test_l4_authority_workflow():
    """
    Test the full L4 authority workflow:
    1. Weaver creates case with membership
    2. Canonical worker enriches presentation
    3. Membership remains unchanged
    """
    mock_neo4j = AsyncMock()

    # Step 1: Weaver creates case
    case = CompilerCase(
        case_id="case_workflow",
        incident_ids=frozenset({"in_001", "in_002", "in_003"}),
        spine_edges=[],
        metabolic_edges=[],
    )
    result = CompilationResult(cases=[case], all_edges=[], deferred=[], inquiries=[], stats={})

    mock_neo4j._execute_read.return_value = []  # New case
    await persist_case_structure(mock_neo4j, case, result)

    # Step 2: Canonical worker enriches
    update = CasePresentationUpdate(
        case_id="case_workflow",
        title="Wang Fuk Court Fire",
        description="Multiple casualties reported",
        case_type="breaking",
    )
    await persist_case_presentation(mock_neo4j, update)

    # Step 3: Verify membership unchanged
    mock_neo4j._execute_read.return_value = [
        {"incident_ids": ["in_001", "in_002", "in_003"]}
    ]

    unchanged = await verify_case_membership_unchanged(
        mock_neo4j, "case_workflow", {"in_001", "in_002", "in_003"}
    )

    assert unchanged is True


# =============================================================================
# persist_metabolic_edges Tests
# =============================================================================

@pytest.mark.asyncio
async def test_persist_metabolic_edges_marks_is_metabolic():
    """Metabolic edges must be marked as non-structural."""
    mock_neo4j = AsyncMock()

    # Create a metabolic edge with CONTEXT_FOR decision
    metabolic_decision = MembraneDecision(
        edge_type=EdgeType.CONTEXT_FOR,
        action=Action.PERIPHERY,
        confidence=0.6,
        reason="shared context only",
        shared_referents=frozenset(),
    )

    metabolic_edge = CompiledEdge(
        incident_a="in_001",
        incident_b="in_002",
        decision=metabolic_decision,
        artifact_hash_a="hash_001",
        artifact_hash_b="hash_002",
    )

    case = CompilerCase(
        case_id="case_metabolic",
        incident_ids=frozenset({"in_001", "in_002"}),
        spine_edges=[],
        metabolic_edges=[metabolic_edge],
    )

    count = await persist_metabolic_edges(mock_neo4j, case, compiler_version="1.0.0")

    assert count == 1
    assert mock_neo4j._execute_write.called

    # Check properties include is_metabolic=True
    call_args = mock_neo4j._execute_write.call_args
    params = call_args[0][1]
    props = params["properties"]

    assert props["is_metabolic"] is True
    assert props["compiler_authorized"] is False  # Metabolic edges don't need auth


@pytest.mark.asyncio
async def test_persist_metabolic_edges_does_not_use_spine_types():
    """Metabolic edges should use CONTEXT_FOR not SAME_HAPPENING."""
    mock_neo4j = AsyncMock()

    metabolic_decision = MembraneDecision(
        edge_type=EdgeType.CONTEXT_FOR,
        action=Action.PERIPHERY,
        confidence=0.5,
        reason="peripheral",
        shared_referents=frozenset(),
    )

    metabolic_edge = CompiledEdge(
        incident_a="in_a",
        incident_b="in_b",
        decision=metabolic_decision,
        artifact_hash_a="hash_a",
        artifact_hash_b="hash_b",
    )

    case = CompilerCase(
        case_id="case_ctx",
        incident_ids=frozenset({"in_a", "in_b"}),
        spine_edges=[],
        metabolic_edges=[metabolic_edge],
    )

    await persist_metabolic_edges(mock_neo4j, case)

    # Check query uses CONTEXT_FOR
    query = mock_neo4j._execute_write.call_args[0][0]
    assert "CONTEXT_FOR" in query
    assert "SAME_HAPPENING" not in query
    assert "UPDATE_TO" not in query


@pytest.mark.asyncio
async def test_persist_metabolic_edges_returns_count():
    """Should return correct count of persisted edges."""
    mock_neo4j = AsyncMock()

    # Create 3 metabolic edges
    edges = []
    for i in range(3):
        decision = MembraneDecision(
            edge_type=EdgeType.CONTEXT_FOR,
            action=Action.PERIPHERY,
            confidence=0.5,
            reason=f"context_{i}",
            shared_referents=frozenset(),
        )
        edges.append(CompiledEdge(
            incident_a=f"in_{i}",
            incident_b=f"in_{i+1}",
            decision=decision,
            artifact_hash_a=f"hash_{i}",
            artifact_hash_b=f"hash_{i+1}",
        ))

    case = CompilerCase(
        case_id="case_multi",
        incident_ids=frozenset({f"in_{i}" for i in range(4)}),
        spine_edges=[],
        metabolic_edges=edges,
    )

    count = await persist_metabolic_edges(mock_neo4j, case)

    assert count == 3
    assert mock_neo4j._execute_write.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
