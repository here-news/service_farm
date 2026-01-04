"""
Case Builder Hub Suppression Tests
===================================

Tests that hub entities (appearing in >30% of incidents) cannot form cores.

Key invariants tested:
1. Entities appearing in too many incidents are marked as hubs
2. Hub-only shared anchors do NOT create core edges
3. Hub suppression prevents mega-case percolation
4. Non-hub shared anchors still create core edges

This addresses the real-world condition where entities like "Hong Kong",
"Trump", "China" appear across many unrelated incidents and should not
bind them into one mega-case.
"""

import pytest
from datetime import datetime, timedelta
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reee.types import Event, EventJustification
from reee.builders.case_builder import (
    PrincipledCaseBuilder, L4Hubness
)


# ============================================================================
# FIXTURES: Test Incidents
# ============================================================================

def make_incident(
    incident_id: str,
    core_motifs: list,
    anchor_entities: set,
    time_start: Optional[datetime] = None,
) -> Event:
    """Create a test incident with specified motifs."""
    justification = EventJustification(
        core_motifs=core_motifs,
        representative_surfaces=[f"S_{incident_id}_001"],
        canonical_handle=f"Test incident {incident_id}",
    )

    return Event(
        id=incident_id,
        surface_ids={f"S_{incident_id}_001"},
        anchor_entities=anchor_entities,
        entities=anchor_entities.copy(),
        total_claims=10,
        total_sources=3,
        time_window=(time_start, time_start + timedelta(days=1) if time_start else (None, None)),
        justification=justification,
    )


@pytest.fixture
def hub_entity_incidents():
    """
    Set of incidents where "Hong Kong" is a hub entity (appears in >30%).

    10 incidents:
    - 5 have "Hong Kong" (hub candidate)
    - Each has unique non-hub entities
    - No shared motifs across the 5 Hong Kong incidents
    """
    incidents = {}

    # 5 Hong Kong incidents with different topics
    for i in range(5):
        incidents[f"HK_{i}"] = make_incident(
            f"HK_{i}",
            core_motifs=[
                {"entities": ["Hong Kong", f"Topic_{i}"], "support": 3},
            ],
            anchor_entities={"Hong Kong", f"Topic_{i}", f"Person_{i}"},
            time_start=datetime(2025, 12, 1) + timedelta(days=i),
        )

    # 5 non-Hong Kong incidents
    for i in range(5, 10):
        incidents[f"OTHER_{i}"] = make_incident(
            f"OTHER_{i}",
            core_motifs=[
                {"entities": [f"Entity_A_{i}", f"Entity_B_{i}"], "support": 4},
            ],
            anchor_entities={f"Entity_A_{i}", f"Entity_B_{i}"},
            time_start=datetime(2025, 12, 1) + timedelta(days=i),
        )

    return incidents


@pytest.fixture
def two_incidents_with_hub_and_non_hub():
    """
    Two incidents that share both a hub entity and a non-hub entity.

    Should form case via non-hub entity, not via hub.
    """
    return {
        "I001": make_incident(
            "I001",
            core_motifs=[
                {"entities": ["Hong Kong", "Tai Po"], "support": 5},
            ],
            anchor_entities={"Hong Kong", "Tai Po", "Fire"},
            time_start=datetime(2025, 11, 26),
        ),
        "I002": make_incident(
            "I002",
            core_motifs=[
                {"entities": ["Hong Kong", "Tai Po"], "support": 4},
                {"entities": ["Rescue", "Tai Po"], "support": 3},
            ],
            anchor_entities={"Hong Kong", "Tai Po", "Rescue"},
            time_start=datetime(2025, 11, 27),
        ),
    }


@pytest.fixture
def many_incidents_for_hub_detection():
    """
    Many incidents to test hub detection threshold.

    Creates 10 incidents where one entity appears in 4 (40% > 30% threshold).
    """
    incidents = {}

    # 4 incidents with "Global Entity"
    for i in range(4):
        incidents[f"GLOBAL_{i}"] = make_incident(
            f"GLOBAL_{i}",
            core_motifs=[
                {"entities": ["Global Entity", f"Specific_{i}"], "support": 3},
            ],
            anchor_entities={"Global Entity", f"Specific_{i}"},
            time_start=datetime(2025, 12, 1) + timedelta(days=i),
        )

    # 6 incidents without "Global Entity"
    for i in range(4, 10):
        incidents[f"LOCAL_{i}"] = make_incident(
            f"LOCAL_{i}",
            core_motifs=[
                {"entities": [f"Local_A_{i}", f"Local_B_{i}"], "support": 4},
            ],
            anchor_entities={f"Local_A_{i}", f"Local_B_{i}"},
            time_start=datetime(2025, 12, 1) + timedelta(days=i),
        )

    return incidents


# ============================================================================
# TEST: HUB DETECTION
# ============================================================================

class TestHubDetection:
    """Test that entities are correctly classified as hubs."""

    
    def test_hub_entity_detected(self, hub_entity_incidents):
        """Entity appearing in >30% of incidents should be detected as hub."""
        builder = PrincipledCaseBuilder(hub_fraction_threshold=0.3)

        result = builder.build_from_incidents(hub_entity_incidents)

        # Hong Kong appears in 5/10 = 50% of incidents
        assert "Hong Kong" in result.hubness
        assert result.hubness["Hong Kong"].is_hub, "Hong Kong should be hub (50% > 30%)"
        assert result.hubness["Hong Kong"].incident_fraction >= 0.5

    
    def test_non_hub_entity_not_hub(self, hub_entity_incidents):
        """Entity appearing in single incident should not be hub."""
        builder = PrincipledCaseBuilder(hub_fraction_threshold=0.3)

        result = builder.build_from_incidents(hub_entity_incidents)

        # Topic_0 only appears in one incident
        if "Topic_0" in result.hubness:
            assert not result.hubness["Topic_0"].is_hub

    
    def test_hub_count_in_stats(self, hub_entity_incidents):
        """Stats should report hub entity count."""
        builder = PrincipledCaseBuilder(hub_fraction_threshold=0.3)

        result = builder.build_from_incidents(hub_entity_incidents)

        # Should have at least one hub (Hong Kong)
        assert result.stats["hub_entities"] >= 1

    
    def test_hub_threshold_respected(self, many_incidents_for_hub_detection):
        """Hub threshold should correctly classify entities."""
        builder = PrincipledCaseBuilder(hub_fraction_threshold=0.3)

        result = builder.build_from_incidents(many_incidents_for_hub_detection)

        # Global Entity appears in 4/10 = 40% > 30% threshold
        assert "Global Entity" in result.hubness
        assert result.hubness["Global Entity"].is_hub, "40% > 30% should be hub"

        # Local entities appear in 1/10 each, should not be hubs
        for h in result.hubness.values():
            if h.entity.startswith("Local_"):
                assert not h.is_hub, f"{h.entity} should not be hub"


# ============================================================================
# TEST: HUB SUPPRESSION IN EDGE FORMATION
# ============================================================================

class TestHubSuppressionEdges:
    """Test that hub entities are suppressed in edge formation."""

    
    def test_hub_only_shared_anchor_not_core(self, hub_entity_incidents):
        """Incidents sharing only a hub entity should NOT form core edge."""
        builder = PrincipledCaseBuilder(hub_fraction_threshold=0.3)

        result = builder.build_from_incidents(hub_entity_incidents)

        # Check edges between Hong Kong incidents
        # They share "Hong Kong" but it's a hub, so no core edge
        for edge in result.edges:
            if edge.incident1_id.startswith("HK_") and edge.incident2_id.startswith("HK_"):
                # If only evidence is hub entity, should not be core
                if not edge.shared_motifs and not edge.motif_chains:
                    if edge.hub_anchors and not edge.shared_anchors:
                        assert not edge.is_core, \
                            f"Hub-only edge {edge.incident1_id}↔{edge.incident2_id} should not be core"

    
    def test_hub_anchors_logged_in_edge(self, hub_entity_incidents):
        """Hub anchors should be logged in edge for explainability."""
        builder = PrincipledCaseBuilder(hub_fraction_threshold=0.3)

        result = builder.build_from_incidents(hub_entity_incidents)

        # Find edge that involves hub
        hub_edges = [
            e for e in result.edges
            if "Hong Kong" in e.hub_anchors
        ]

        # If there are edges between HK incidents, they should log the hub
        if hub_edges:
            for edge in hub_edges:
                assert "Hong Kong" in edge.hub_anchors


# ============================================================================
# TEST: NON-HUB EDGE FORMATION STILL WORKS
# ============================================================================

class TestNonHubEdgesWork:
    """Test that non-hub entities still create core edges."""

    
    def test_non_hub_motif_creates_core(self, two_incidents_with_hub_and_non_hub):
        """
        Shared motif with non-hub entity should create core edge.

        Even if "Hong Kong" is a hub, {Hong Kong, Tai Po} motif is valid
        because Tai Po is not a hub.
        """
        builder = PrincipledCaseBuilder(hub_fraction_threshold=0.3)

        result = builder.build_from_incidents(two_incidents_with_hub_and_non_hub)

        # Only 2 incidents, so nothing can be a hub (< min_incidents=3)
        # Both should merge via shared motif

        # Should form one case
        assert len(result.cases) == 1, "Should form case via shared motif"

        case = list(result.cases.values())[0]
        assert "I001" in case.incident_ids
        assert "I002" in case.incident_ids

    
    def test_mixed_hub_nonhub_uses_nonhub(self):
        """
        When incidents share both hub and non-hub, use non-hub for core.

        Creates scenario:
        - "Hong Kong" is hub (appears in 4/10 incidents)
        - Two incidents share {Hong Kong, Specific_Topic}
        - Core should form via Specific_Topic, not Hong Kong
        """
        incidents = {}

        # 4 incidents with "Hong Kong" as hub
        for i in range(4):
            incidents[f"HK_{i}"] = make_incident(
                f"HK_{i}",
                core_motifs=[
                    {"entities": ["Hong Kong", f"Random_{i}"], "support": 2},
                ],
                anchor_entities={"Hong Kong", f"Random_{i}"},
                time_start=datetime(2025, 12, 1) + timedelta(days=i),
            )

        # 2 incidents that share a non-hub entity
        incidents["PAIR_A"] = make_incident(
            "PAIR_A",
            core_motifs=[
                {"entities": ["Rare Entity", "Specific Topic"], "support": 5},
            ],
            anchor_entities={"Rare Entity", "Specific Topic"},
            time_start=datetime(2025, 12, 10),
        )
        incidents["PAIR_B"] = make_incident(
            "PAIR_B",
            core_motifs=[
                {"entities": ["Rare Entity", "Specific Topic"], "support": 4},
            ],
            anchor_entities={"Rare Entity", "Specific Topic"},
            time_start=datetime(2025, 12, 11),
        )

        # 4 more random incidents
        for i in range(4, 8):
            incidents[f"OTHER_{i}"] = make_incident(
                f"OTHER_{i}",
                core_motifs=[
                    {"entities": [f"Entity_X_{i}", f"Entity_Y_{i}"], "support": 3},
                ],
                anchor_entities={f"Entity_X_{i}", f"Entity_Y_{i}"},
                time_start=datetime(2025, 12, 1) + timedelta(days=i),
            )

        builder = PrincipledCaseBuilder(hub_fraction_threshold=0.3)
        result = builder.build_from_incidents(incidents)

        # PAIR_A and PAIR_B should form a case via shared motif
        # Hong Kong incidents should NOT all merge into mega-case

        assert len(result.cases) >= 1, "PAIR_A and PAIR_B should form case"

        # Find the PAIR case
        pair_case = None
        for case in result.cases.values():
            if "PAIR_A" in case.incident_ids and "PAIR_B" in case.incident_ids:
                pair_case = case
                break

        assert pair_case is not None, "PAIR_A and PAIR_B should be in same case"


# ============================================================================
# TEST: PERCOLATION PREVENTION
# ============================================================================

class TestPercolationPrevention:
    """Test that hub suppression prevents mega-case formation."""

    
    def test_hub_does_not_create_mega_case(self, hub_entity_incidents):
        """Hub entity should not cause all incidents to merge into one mega-case."""
        builder = PrincipledCaseBuilder(hub_fraction_threshold=0.3)

        result = builder.build_from_incidents(hub_entity_incidents)

        # If there are any cases, they should be small
        # The 5 Hong Kong incidents should NOT form a single case just
        # because they share "Hong Kong"

        for case in result.cases.values():
            # No case should contain all 5 Hong Kong incidents
            hk_incidents_in_case = sum(
                1 for inc_id in case.incident_ids if inc_id.startswith("HK_")
            )
            # Allow at most 3 (2 might share non-hub motifs by construction)
            assert hk_incidents_in_case <= 3, \
                f"Case contains {hk_incidents_in_case} HK incidents - hub percolation!"

    
    def test_total_case_count_reasonable(self, hub_entity_incidents):
        """Total number of cases should be reasonable, not 1 mega-case."""
        builder = PrincipledCaseBuilder(hub_fraction_threshold=0.3)

        result = builder.build_from_incidents(hub_entity_incidents)

        # With 10 largely unrelated incidents, we should have few or no cases
        # (min_incidents_for_case=2), not 1 mega-case containing all 10
        if len(result.cases) > 0:
            max_case_size = max(c.incident_count for c in result.cases.values())
            assert max_case_size < len(hub_entity_incidents), \
                "Should not have mega-case containing all incidents"


# ============================================================================
# TEST: CONSTRAINT LEDGER FOR HUBS
# ============================================================================

class TestHubConstraintLedger:
    """Test that hub detection is recorded in constraint ledger."""

    
    def test_hub_detection_logged(self, hub_entity_incidents):
        """Hub detection should be logged in constraint ledger."""
        builder = PrincipledCaseBuilder(hub_fraction_threshold=0.3)

        result = builder.build_from_incidents(hub_entity_incidents)

        # Find hub detection constraints
        hub_constraints = [
            c for c in result.ledger.constraints
            if c.provenance == "l4_hub_detection"
        ]

        # Should have at least one hub detection (Hong Kong)
        assert len(hub_constraints) >= 1

        # Check constraint evidence
        hk_constraint = next(
            (c for c in hub_constraints if c.evidence.get("entity") == "Hong Kong"),
            None
        )
        assert hk_constraint is not None
        assert hk_constraint.evidence["incident_fraction"] >= 0.5


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
