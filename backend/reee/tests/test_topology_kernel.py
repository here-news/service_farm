"""
Tests for TopologyKernel and Phase 2 modules.

These tests verify:
- Surface key computation
- Incident routing semantics
- TopologyKernel end-to-end processing
- Weaver semantics alignment (MIN_SHARED_ANCHORS, etc.)
"""

import pytest
from datetime import datetime, timedelta
from typing import List

from reee.contracts.evidence import ClaimEvidence
from reee.contracts.state import (
    SurfaceKey,
    SurfaceState,
    IncidentState,
    PartitionSnapshot,
)
from reee.contracts.signals import SignalType

from reee.topo import (
    # Surface update
    compute_surface_key,
    apply_claim_to_surface,
    SurfaceKeyParams,
    # Incident routing
    find_candidates,
    decide_route,
    RoutingParams,
    RouteOutcome,
    # Topology kernel
    TopologyKernel,
    KernelParams,
)


class TestComputeSurfaceKey:
    """Tests for compute_surface_key."""

    def test_explicit_key_used(self):
        """Explicit question_key should be used when provided."""
        evidence = ClaimEvidence(
            claim_id="c1",
            text="Some claim about death toll",
            source_id="bbc.com",
            entities=frozenset({"Hong Kong", "John Lee"}),
            anchors=frozenset({"Hong Kong", "John Lee"}),
            question_key="fire_death_count",
            question_key_confidence=0.9,
        )

        result = compute_surface_key(evidence)

        assert result.question_key == "fire_death_count"
        assert "FALLBACK_EXPLICIT" in result.trace.rules_fired

    def test_pattern_fallback(self):
        """Pattern should be used when no explicit key."""
        evidence = ClaimEvidence(
            claim_id="c1",
            text="15 people killed in the fire",
            source_id="bbc.com",
            entities=frozenset({"Hong Kong"}),
            anchors=frozenset({"Hong Kong"}),
        )

        result = compute_surface_key(evidence)

        assert "death_count" in result.question_key
        assert "FALLBACK_PATTERN" in result.trace.rules_fired

    def test_singleton_fallback_when_no_semantic(self):
        """Singleton key when no pattern or explicit key - semantic-first design."""
        evidence = ClaimEvidence(
            claim_id="c1",
            text="John spoke at the conference",
            source_id="news.com",
            entities=frozenset({"John Lee", "Tech Conference"}),
            anchors=frozenset({"John Lee"}),
        )

        result = compute_surface_key(evidence)

        # With semantic-first design, no ENTITY fallback - goes straight to SINGLETON
        assert result.question_key.startswith("singleton_")
        assert "FALLBACK_SINGLETON" in result.trace.rules_fired

    def test_scope_from_anchors(self):
        """Scope should be derived from anchors."""
        evidence = ClaimEvidence(
            claim_id="c1",
            text="Fire in Hong Kong",
            source_id="bbc.com",
            entities=frozenset({"Hong Kong", "Tai Po"}),
            anchors=frozenset({"Hong Kong", "Tai Po"}),
            question_key="fire_status",
        )

        result = compute_surface_key(evidence)

        assert "hongkong" in result.scope_id
        assert "taipo" in result.scope_id

    def test_trace_includes_confidence(self):
        """Trace should include confidence metrics."""
        evidence = ClaimEvidence(
            claim_id="c1",
            text="Test claim",
            source_id="test.com",
            entities=frozenset({"A"}),
            entity_confidence=0.7,
        )

        result = compute_surface_key(evidence)

        assert result.trace.features.extraction_confidence == 0.7


class TestApplyClaimToSurface:
    """Tests for apply_claim_to_surface."""

    def test_create_new_surface(self):
        """Should create new surface when none exists."""
        evidence = ClaimEvidence(
            claim_id="c1",
            text="Test claim",
            source_id="bbc.com",
            entities=frozenset({"A", "B"}),
            anchors=frozenset({"A"}),
            time=datetime(2024, 1, 15, 12, 0),
        )
        key = SurfaceKey(scope_id="scope_a", question_key="test_key")

        surface, is_new = apply_claim_to_surface(None, evidence, key)

        assert is_new
        assert "c1" in surface.claim_ids
        assert "A" in surface.entities
        assert "B" in surface.entities
        assert "A" in surface.anchor_entities
        assert "bbc.com" in surface.sources

    def test_update_existing_surface(self):
        """Should update existing surface."""
        key = SurfaceKey(scope_id="scope_a", question_key="test_key")
        existing = SurfaceState(
            key=key,
            claim_ids=frozenset({"c1"}),
            entities=frozenset({"A"}),
            anchor_entities=frozenset({"A"}),
            sources=frozenset({"bbc.com"}),
            time_start=datetime(2024, 1, 15),
        )

        evidence = ClaimEvidence(
            claim_id="c2",
            text="Another claim",
            source_id="cnn.com",
            entities=frozenset({"A", "C"}),
            anchors=frozenset({"A"}),
            time=datetime(2024, 1, 16),
        )

        surface, is_new = apply_claim_to_surface(existing, evidence, key)

        assert not is_new
        assert "c1" in surface.claim_ids
        assert "c2" in surface.claim_ids
        assert "C" in surface.entities
        assert "cnn.com" in surface.sources
        assert surface.time_end == datetime(2024, 1, 16)


class TestFindCandidates:
    """Tests for find_candidates."""

    @pytest.fixture
    def sample_snapshot(self) -> PartitionSnapshot:
        """Create a snapshot with candidate incidents."""
        incidents = [
            IncidentState(
                id="inc_1",
                signature="inc_sig_1",
                surface_ids=frozenset({"sf_1"}),
                anchor_entities=frozenset({"Hong Kong", "John Lee"}),
                companion_entities=frozenset({"Carrie Lam", "Tai Po"}),
                time_start=datetime(2024, 1, 15),
            ),
            IncidentState(
                id="inc_2",
                signature="inc_sig_2",
                surface_ids=frozenset({"sf_2"}),
                anchor_entities=frozenset({"California", "Gavin Newsom"}),
                companion_entities=frozenset({"Los Angeles"}),
                time_start=datetime(2024, 1, 15),
            ),
        ]
        return PartitionSnapshot(
            scope_id="scope_test",
            surfaces=[],
            incidents=incidents,
        )

    def test_finds_matching_anchors(self, sample_snapshot):
        """Should find incidents with matching anchors."""
        key = SurfaceKey(scope_id="scope_hongkong_johnlee", question_key="test")
        surface = SurfaceState(
            key=key,
            entities=frozenset({"Hong Kong", "John Lee", "New Entity"}),
            anchor_entities=frozenset({"Hong Kong", "John Lee"}),
        )

        candidates = find_candidates(
            surface=surface,
            snapshot=sample_snapshot,
            params=RoutingParams(min_shared_anchors=2),
        )

        assert len(candidates) == 1
        assert candidates[0].incident_signature == "inc_sig_1"
        assert candidates[0].anchor_count == 2

    def test_no_match_insufficient_anchors(self, sample_snapshot):
        """Should not match with insufficient anchors."""
        key = SurfaceKey(scope_id="scope_hongkong", question_key="test")
        surface = SurfaceState(
            key=key,
            entities=frozenset({"Hong Kong"}),
            anchor_entities=frozenset({"Hong Kong"}),  # Only 1 anchor
        )

        candidates = find_candidates(
            surface=surface,
            snapshot=sample_snapshot,
            params=RoutingParams(min_shared_anchors=2),
        )

        assert len(candidates) == 0

    def test_companion_overlap_computed(self, sample_snapshot):
        """Should compute companion overlap."""
        key = SurfaceKey(scope_id="scope_hongkong_johnlee", question_key="test")
        surface = SurfaceState(
            key=key,
            entities=frozenset({"Hong Kong", "John Lee", "Tai Po", "Extra Entity"}),
            anchor_entities=frozenset({"Hong Kong", "John Lee"}),
        )

        candidates = find_candidates(
            surface=surface,
            snapshot=sample_snapshot,
            params=RoutingParams(),
        )

        assert len(candidates) == 1
        # Companions: surface has {Tai Po, Extra Entity}, incident has {Carrie Lam, Tai Po}
        # Intersection: {Tai Po}, Union: {Tai Po, Extra Entity, Carrie Lam}
        # Overlap: 1/3 ≈ 0.33
        assert candidates[0].companion_overlap > 0.3


class TestDecideRoute:
    """Tests for decide_route."""

    def test_join_compatible_incident(self):
        """Should join incident with compatible companions."""
        key = SurfaceKey(scope_id="scope_a_b", question_key="test")
        surface = SurfaceState(
            key=key,
            entities=frozenset({"A", "B", "C", "D"}),
            anchor_entities=frozenset({"A", "B"}),
        )

        # High companion overlap
        candidates = [
            pytest.importorskip("reee.kernel.incident_routing").CandidateScore(
                incident_id="inc_1",
                incident_signature="inc_sig_1",
                shared_anchors=frozenset({"A", "B"}),
                anchor_count=2,
                companion_overlap=0.5,  # Above threshold
                is_underpowered=False,
            )
        ]

        result = decide_route(
            surface=surface,
            candidates=candidates,
            params=RoutingParams(companion_overlap_threshold=0.15),
        )

        assert result.outcome == RouteOutcome.JOINED_EXISTING
        assert result.incident_signature == "inc_sig_1"
        assert not result.is_new

    def test_bridge_blocked_signal(self):
        """Should emit BRIDGE_BLOCKED for incompatible companions."""
        from reee.topo.incident_routing import CandidateScore

        key = SurfaceKey(scope_id="scope_a_b", question_key="test")
        surface = SurfaceState(
            key=key,
            entities=frozenset({"A", "B", "X", "Y"}),
            anchor_entities=frozenset({"A", "B"}),
        )

        # Low companion overlap
        candidates = [
            CandidateScore(
                incident_id="inc_1",
                incident_signature="inc_sig_1",
                shared_anchors=frozenset({"A", "B"}),
                anchor_count=2,
                companion_overlap=0.05,  # Below threshold
                is_underpowered=False,
            )
        ]

        result = decide_route(
            surface=surface,
            candidates=candidates,
            params=RoutingParams(companion_overlap_threshold=0.15),
        )

        assert result.outcome == RouteOutcome.CREATED_NEW
        assert result.is_new
        assert len(result.signals) == 1
        assert result.signals[0].signal_type == SignalType.BRIDGE_BLOCKED

    def test_underpowered_allows_join(self):
        """Underpowered companions should allow join (benefit of doubt)."""
        from reee.topo.incident_routing import CandidateScore

        key = SurfaceKey(scope_id="scope_a_b", question_key="test")
        surface = SurfaceState(
            key=key,
            entities=frozenset({"A", "B"}),  # No companions
            anchor_entities=frozenset({"A", "B"}),
        )

        candidates = [
            CandidateScore(
                incident_id="inc_1",
                incident_signature="inc_sig_1",
                shared_anchors=frozenset({"A", "B"}),
                anchor_count=2,
                companion_overlap=0.5,  # Default underpowered score
                is_underpowered=True,
            )
        ]

        result = decide_route(
            surface=surface,
            candidates=candidates,
            params=RoutingParams(),
        )

        assert result.outcome == RouteOutcome.JOINED_EXISTING
        assert "COMPANION_UNDERPOWERED_ALLOW" in result.trace.rules_fired

    def test_create_new_when_no_candidates(self):
        """Should create new incident when no candidates."""
        key = SurfaceKey(scope_id="scope_a_b", question_key="test")
        surface = SurfaceState(
            key=key,
            entities=frozenset({"A", "B"}),
            anchor_entities=frozenset({"A", "B"}),
            time_start=datetime(2024, 1, 15),
        )

        result = decide_route(
            surface=surface,
            candidates=[],
            params=RoutingParams(),
        )

        assert result.outcome == RouteOutcome.CREATED_NEW
        assert result.is_new
        assert result.incident_signature.startswith("inc_")


class TestTopologyKernel:
    """Tests for TopologyKernel end-to-end."""

    def test_process_single_evidence(self):
        """Should process single evidence and return delta."""
        kernel = TopologyKernel()

        evidence = ClaimEvidence(
            claim_id="c1",
            text="15 people killed in the Hong Kong fire",
            source_id="bbc.com",
            entities=frozenset({"Hong Kong", "Tai Po"}),
            anchors=frozenset({"Hong Kong", "Tai Po"}),
            time=datetime(2024, 1, 15, 12, 0),
        )

        snapshot = PartitionSnapshot(
            scope_id="scope_hongkong_taipo",
            surfaces=[],
            incidents=[],
        )

        delta = kernel.process_evidence(snapshot, evidence)

        # Should have created a new surface
        assert len(delta.surface_upserts) == 1
        assert "c1" in delta.surface_upserts[0].claim_ids

        # Should have created a new incident
        assert len(delta.incident_upserts) == 1

        # Should have traces
        assert len(delta.decision_traces) == 2  # surface_key + incident_membership

        # Should have links
        assert len(delta.links) >= 2  # claim→surface, surface→incident

    def test_process_evidence_joins_incident(self):
        """Should join existing incident when compatible."""
        kernel = TopologyKernel()

        # Existing incident
        existing_incident = IncidentState(
            id="inc_existing",
            signature="inc_sig_existing",
            surface_ids=frozenset({"sf_1"}),
            anchor_entities=frozenset({"Hong Kong", "Tai Po"}),
            companion_entities=frozenset({"John Lee"}),
            time_start=datetime(2024, 1, 14),
        )

        snapshot = PartitionSnapshot(
            scope_id="scope_hongkong_taipo",
            surfaces=[],
            incidents=[existing_incident],
        )

        evidence = ClaimEvidence(
            claim_id="c2",
            text="Death toll rises in Hong Kong fire",
            source_id="cnn.com",
            entities=frozenset({"Hong Kong", "Tai Po", "John Lee", "Carrie Lam"}),
            anchors=frozenset({"Hong Kong", "Tai Po"}),
            time=datetime(2024, 1, 15),
        )

        delta = kernel.process_evidence(snapshot, evidence)

        # Should update existing incident
        assert len(delta.incident_upserts) == 1
        # Check if incident was joined (not new)
        routing_trace = [t for t in delta.decision_traces if t.decision_type == "incident_membership"][0]
        assert routing_trace.outcome == "joined_existing"

    def test_process_batch(self):
        """Should process batch of evidence."""
        kernel = TopologyKernel()

        evidence_list = [
            ClaimEvidence(
                claim_id="c1",
                text="Fire started in Tai Po",
                source_id="bbc.com",
                entities=frozenset({"Hong Kong", "Tai Po"}),
                anchors=frozenset({"Hong Kong", "Tai Po"}),
                question_key="fire_status",
                time=datetime(2024, 1, 15, 10, 0),
            ),
            ClaimEvidence(
                claim_id="c2",
                text="10 dead in Tai Po fire",
                source_id="cnn.com",
                entities=frozenset({"Hong Kong", "Tai Po"}),
                anchors=frozenset({"Hong Kong", "Tai Po"}),
                question_key="fire_death_count",
                time=datetime(2024, 1, 15, 12, 0),
            ),
            ClaimEvidence(
                claim_id="c3",
                text="Fire contained in Tai Po",
                source_id="reuters.com",
                entities=frozenset({"Hong Kong", "Tai Po"}),
                anchors=frozenset({"Hong Kong", "Tai Po"}),
                question_key="fire_status",
                time=datetime(2024, 1, 15, 14, 0),
            ),
        ]

        snapshot = PartitionSnapshot(
            scope_id="scope_hongkong_taipo",
            surfaces=[],
            incidents=[],
        )

        delta = kernel.process_batch(snapshot, evidence_list)

        # Should have 2 surfaces (fire_status and fire_death_count)
        assert len(delta.surface_upserts) >= 2

        # Should have 1 incident (all share anchors)
        assert len(delta.incident_upserts) >= 1

        # Should have traces for all evidence
        assert len(delta.decision_traces) >= 6  # 3 * (surface_key + incident_membership)


class TestWeaverSemanticsAlignment:
    """Tests verifying alignment with current weaver semantics."""

    def test_min_shared_anchors_default(self):
        """Default MIN_SHARED_ANCHORS should be 2."""
        params = RoutingParams()
        assert params.min_shared_anchors == 2

    def test_companion_overlap_threshold_default(self):
        """Default COMPANION_OVERLAP_THRESHOLD should be 0.15."""
        params = RoutingParams()
        assert params.companion_overlap_threshold == 0.15

    def test_bridge_blocked_evidence_structure(self):
        """BRIDGE_BLOCKED signal should have expected evidence structure."""
        from reee.topo.incident_routing import CandidateScore

        key = SurfaceKey(scope_id="scope_a_b", question_key="test")
        surface = SurfaceState(
            key=key,
            entities=frozenset({"A", "B", "X", "Y"}),
            anchor_entities=frozenset({"A", "B"}),
        )

        candidates = [
            CandidateScore(
                incident_id="inc_1",
                incident_signature="inc_sig_1",
                shared_anchors=frozenset({"A", "B"}),
                anchor_count=2,
                companion_overlap=0.05,
                is_underpowered=False,
            )
        ]

        result = decide_route(surface, candidates, RoutingParams())

        signal = result.signals[0]
        assert "blocking_entity" in signal.evidence
        assert "shared_anchors" in signal.evidence
        assert "surface_companions" in signal.evidence
        assert "companion_overlap" in signal.evidence
        assert "threshold" in signal.evidence
