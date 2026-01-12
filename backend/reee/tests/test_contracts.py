"""
Tests for REEE Contracts.

These tests verify:
- Immutability of frozen dataclasses
- Correct serialization
- Evidence hash stability
- Signal/inquiry mapping
"""

import pytest
from datetime import datetime

from reee.contracts import (
    ClaimEvidence,
    TypedObservation,
    SurfaceKey,
    SurfaceState,
    IncidentState,
    PartitionSnapshot,
    TopologyDelta,
    Link,
    DecisionTrace,
    BeliefUpdateTrace,
    FeatureVector,
    SignalType,
    EpistemicSignal,
    InquiryType,
    InquirySeed,
)
from reee.contracts.signals import signal_to_inquiry, Severity
from reee.contracts.state import compute_incident_signature


class TestClaimEvidence:
    """Tests for ClaimEvidence contract."""

    def test_immutability(self):
        """ClaimEvidence should be immutable."""
        evidence = ClaimEvidence(
            claim_id="c1",
            text="Test claim",
            source_id="bbc.com",
        )
        with pytest.raises(AttributeError):
            evidence.text = "Modified"

    def test_evidence_hash_stability(self):
        """Same evidence should produce same hash."""
        evidence1 = ClaimEvidence(
            claim_id="c1",
            text="Test claim",
            source_id="bbc.com",
            entities=frozenset({"Entity A", "Entity B"}),
        )
        evidence2 = ClaimEvidence(
            claim_id="c1",
            text="Test claim",
            source_id="bbc.com",
            entities=frozenset({"Entity B", "Entity A"}),  # Different order
        )
        assert evidence1.evidence_hash == evidence2.evidence_hash

    def test_evidence_hash_changes(self):
        """Different evidence should produce different hash."""
        evidence1 = ClaimEvidence(
            claim_id="c1",
            text="Test claim",
            source_id="bbc.com",
        )
        evidence2 = ClaimEvidence(
            claim_id="c1",
            text="Different claim",
            source_id="bbc.com",
        )
        assert evidence1.evidence_hash != evidence2.evidence_hash

    def test_with_enrichment(self):
        """with_enrichment should create new instance with added data."""
        evidence = ClaimEvidence(
            claim_id="c1",
            text="Test claim",
            source_id="bbc.com",
        )
        enriched = evidence.with_enrichment(
            entities=frozenset({"Entity A"}),
            entity_confidence=0.9,
        )

        # Original unchanged
        assert evidence.entities == frozenset()
        assert evidence.entity_confidence == 0.5

        # Enriched has new values
        assert enriched.entities == frozenset({"Entity A"})
        assert enriched.entity_confidence == 0.9

    def test_companions_property(self):
        """companions should return entities minus anchors."""
        evidence = ClaimEvidence(
            claim_id="c1",
            text="Test",
            source_id="test",
            entities=frozenset({"A", "B", "C"}),
            anchors=frozenset({"A"}),
        )
        assert evidence.companions == frozenset({"B", "C"})


class TestSurfaceKey:
    """Tests for SurfaceKey contract."""

    def test_signature_deterministic(self):
        """Signature should be deterministic."""
        key1 = SurfaceKey(scope_id="scope_a", question_key="death_count")
        key2 = SurfaceKey(scope_id="scope_a", question_key="death_count")
        assert key1.signature == key2.signature

    def test_signature_differs_by_scope(self):
        """Different scope should produce different signature."""
        key1 = SurfaceKey(scope_id="scope_a", question_key="death_count")
        key2 = SurfaceKey(scope_id="scope_b", question_key="death_count")
        assert key1.signature != key2.signature

    def test_signature_differs_by_question(self):
        """Different question_key should produce different signature."""
        key1 = SurfaceKey(scope_id="scope_a", question_key="death_count")
        key2 = SurfaceKey(scope_id="scope_a", question_key="injury_count")
        assert key1.signature != key2.signature

    def test_signature_format(self):
        """Signature should have correct format."""
        key = SurfaceKey(scope_id="scope_test", question_key="test_key")
        assert key.signature.startswith("sf_")
        assert len(key.signature) == 15  # "sf_" + 12 chars


class TestIncidentSignature:
    """Tests for incident signature computation."""

    def test_signature_deterministic(self):
        """Signature should be deterministic."""
        anchors = frozenset({"Entity A", "Entity B"})
        time1 = datetime(2024, 1, 15, 12, 0, 0)
        time2 = datetime(2024, 1, 15, 12, 0, 0)

        sig1 = compute_incident_signature(anchors, time1)
        sig2 = compute_incident_signature(anchors, time2)
        assert sig1 == sig2

    def test_signature_same_week(self):
        """Dates in same ISO week should produce same signature."""
        anchors = frozenset({"Entity A"})
        # Monday and Friday of same week
        time1 = datetime(2024, 1, 15, 12, 0, 0)  # Monday
        time2 = datetime(2024, 1, 19, 12, 0, 0)  # Friday

        sig1 = compute_incident_signature(anchors, time1)
        sig2 = compute_incident_signature(anchors, time2)
        assert sig1 == sig2

    def test_signature_different_week(self):
        """Dates in different ISO weeks should produce different signature."""
        anchors = frozenset({"Entity A"})
        time1 = datetime(2024, 1, 15, 12, 0, 0)  # Week 3
        time2 = datetime(2024, 1, 22, 12, 0, 0)  # Week 4

        sig1 = compute_incident_signature(anchors, time1)
        sig2 = compute_incident_signature(anchors, time2)
        assert sig1 != sig2

    def test_signature_unknown_time(self):
        """None time should produce 'unknown' component."""
        anchors = frozenset({"Entity A"})
        sig = compute_incident_signature(anchors, None)
        assert sig.startswith("inc_")

    def test_signature_format(self):
        """Signature should have correct format."""
        anchors = frozenset({"Entity A"})
        sig = compute_incident_signature(anchors, datetime.now())
        assert sig.startswith("inc_")
        assert len(sig) == 16  # "inc_" + 12 chars


class TestSurfaceState:
    """Tests for SurfaceState."""

    def test_with_claim(self):
        """with_claim should add claim and update state."""
        key = SurfaceKey(scope_id="scope_a", question_key="test")
        surface = SurfaceState(key=key)

        updated = surface.with_claim(
            claim_id="c1",
            entities=frozenset({"A", "B"}),
            anchors=frozenset({"A"}),
            source_id="bbc.com",
            claim_time=datetime(2024, 1, 15, 12, 0, 0),
        )

        assert "c1" in updated.claim_ids
        assert "A" in updated.entities
        assert "B" in updated.entities
        assert "A" in updated.anchor_entities
        assert "bbc.com" in updated.sources
        assert updated.time_start == datetime(2024, 1, 15, 12, 0, 0)

    def test_with_claim_expands_time_bounds(self):
        """Multiple claims should expand time bounds."""
        key = SurfaceKey(scope_id="scope_a", question_key="test")
        surface = SurfaceState(
            key=key,
            time_start=datetime(2024, 1, 15),
            time_end=datetime(2024, 1, 15),
        )

        updated = surface.with_claim(
            claim_id="c2",
            entities=frozenset(),
            anchors=frozenset(),
            source_id=None,
            claim_time=datetime(2024, 1, 20),
        )

        assert updated.time_start == datetime(2024, 1, 15)
        assert updated.time_end == datetime(2024, 1, 20)


class TestSignals:
    """Tests for signal contracts."""

    def test_signal_to_inquiry_mapping(self):
        """Signals should map to appropriate inquiries."""
        signal = EpistemicSignal(
            id="sig_1",
            signal_type=SignalType.CONFLICT,
            subject_id="surface_1",
            subject_type="surface",
            severity=Severity.WARNING,
            evidence={"values": [10, 20]},
            resolution_hint="Find authoritative source",
            timestamp=datetime.now(),
        )

        inquiry = signal_to_inquiry(signal)

        assert inquiry is not None
        assert inquiry.inquiry_type == InquiryType.RESOLVE_VALUE
        assert inquiry.source_signal_id == "sig_1"

    def test_signal_without_inquiry_mapping(self):
        """Some signals don't need inquiries."""
        signal = EpistemicSignal(
            id="sig_1",
            signal_type=SignalType.SCOPE_UNDERPOWERED,
            subject_id="claim_1",
            subject_type="claim",
            severity=Severity.INFO,
            evidence={},
            resolution_hint=None,
            timestamp=datetime.now(),
        )

        inquiry = signal_to_inquiry(signal)
        assert inquiry is None


class TestTopologyDelta:
    """Tests for TopologyDelta."""

    def test_has_changes(self):
        """has_changes should detect structural changes."""
        empty_delta = TopologyDelta()
        assert not empty_delta.has_changes

        delta_with_surface = TopologyDelta(
            surface_upserts=[SurfaceState(key=SurfaceKey("s", "q"))]
        )
        assert delta_with_surface.has_changes

    def test_merge(self):
        """Deltas should merge correctly."""
        delta1 = TopologyDelta(
            surface_upserts=[SurfaceState(key=SurfaceKey("s1", "q"))],
            links=[Link("a", "REL", "b")],
        )
        delta2 = TopologyDelta(
            surface_upserts=[SurfaceState(key=SurfaceKey("s2", "q"))],
            links=[Link("c", "REL", "d")],
        )

        merged = delta1.merge(delta2)

        assert len(merged.surface_upserts) == 2
        assert len(merged.links) == 2

    def test_to_summary(self):
        """to_summary should provide useful stats."""
        delta = TopologyDelta(
            surface_upserts=[SurfaceState(key=SurfaceKey("s", "q"))],
            signals=[
                EpistemicSignal(
                    id="sig_1",
                    signal_type=SignalType.CONFLICT,
                    subject_id="s",
                    subject_type="surface",
                    severity=Severity.WARNING,
                    evidence={},
                    resolution_hint=None,
                    timestamp=datetime.now(),
                )
            ],
        )

        summary = delta.to_summary()

        assert summary["surfaces_upserted"] == 1
        assert summary["signals"] == 1
        assert "conflict" in summary["signal_types"]
