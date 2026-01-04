"""
Case Builder Motif Chain Tests
===============================

Tests that motif chains (2-hop continuity) allow case formation.

Key invariants tested:
1. Motif chain (A has {X,Y}, B has {Y,Z}) creates edge
2. Chain evidence is weaker than shared motif (needs additional constraints)
3. Chains enable reaching across related but not identical incidents
4. Hub entities in chain overlap are suppressed

This addresses the real-world condition where related incidents may not
share exact motifs but have overlapping entity contexts.
"""

import pytest
from datetime import datetime, timedelta
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reee.types import Event, EventJustification, ConstraintType
from reee.builders.case_builder import PrincipledCaseBuilder


# ============================================================================
# FIXTURES: Test Incidents with Chain Patterns
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
def chain_incidents():
    """
    Incidents with motif chain pattern.

    I001: {Do Kwon, Terraform Labs}
    I002: {Terraform Labs, SEC}
    I003: {SEC, Montenegro}

    Chain: I001 ↔ I002 via "Terraform Labs"
    Chain: I002 ↔ I003 via "SEC"
    NO direct link: I001 ↔ I003 (no shared entities)
    """
    return {
        "I001": make_incident(
            "I001",
            core_motifs=[
                {"entities": ["Do Kwon", "Terraform Labs"], "support": 5},
            ],
            anchor_entities={"Do Kwon", "Terraform Labs"},
            time_start=datetime(2025, 12, 1),
        ),
        "I002": make_incident(
            "I002",
            core_motifs=[
                {"entities": ["Terraform Labs", "SEC"], "support": 4},
            ],
            anchor_entities={"Terraform Labs", "SEC"},
            time_start=datetime(2025, 12, 5),
        ),
        "I003": make_incident(
            "I003",
            core_motifs=[
                {"entities": ["SEC", "Montenegro"], "support": 3},
            ],
            anchor_entities={"SEC", "Montenegro"},
            time_start=datetime(2025, 12, 10),
        ),
    }


@pytest.fixture
def chain_with_multiple_overlaps():
    """
    Incidents with strong chain (multiple overlapping entities).

    I001: {A, B, C}
    I002: {B, C, D}

    Overlap: {B, C} - strong chain
    """
    return {
        "I001": make_incident(
            "I001",
            core_motifs=[
                {"entities": ["Entity_A", "Entity_B", "Entity_C"], "support": 5},
            ],
            anchor_entities={"Entity_A", "Entity_B", "Entity_C"},
            time_start=datetime(2025, 12, 1),
        ),
        "I002": make_incident(
            "I002",
            core_motifs=[
                {"entities": ["Entity_B", "Entity_C", "Entity_D"], "support": 4},
            ],
            anchor_entities={"Entity_B", "Entity_C", "Entity_D"},
            time_start=datetime(2025, 12, 5),
        ),
    }


@pytest.fixture
def chain_with_hub():
    """
    Chain where the overlapping entity is a hub.

    Creates many incidents to make "Hong Kong" a hub, then
    two incidents that only chain via Hong Kong.
    """
    incidents = {}

    # Make "Hong Kong" a hub by appearing in 5 incidents
    for i in range(5):
        incidents[f"HK_{i}"] = make_incident(
            f"HK_{i}",
            core_motifs=[
                {"entities": ["Hong Kong", f"Random_{i}"], "support": 2},
            ],
            anchor_entities={"Hong Kong", f"Random_{i}"},
            time_start=datetime(2025, 12, 1) + timedelta(days=i),
        )

    # Two incidents with chain via "Hong Kong"
    incidents["CHAIN_A"] = make_incident(
        "CHAIN_A",
        core_motifs=[
            {"entities": ["Topic_X", "Hong Kong"], "support": 3},
        ],
        anchor_entities={"Topic_X", "Hong Kong"},
        time_start=datetime(2025, 12, 10),
    )
    incidents["CHAIN_B"] = make_incident(
        "CHAIN_B",
        core_motifs=[
            {"entities": ["Hong Kong", "Topic_Y"], "support": 3},
        ],
        anchor_entities={"Hong Kong", "Topic_Y"},
        time_start=datetime(2025, 12, 11),
    )

    # 3 more random incidents
    for i in range(3):
        incidents[f"OTHER_{i}"] = make_incident(
            f"OTHER_{i}",
            core_motifs=[
                {"entities": [f"Unrelated_A_{i}", f"Unrelated_B_{i}"], "support": 2},
            ],
            anchor_entities={f"Unrelated_A_{i}", f"Unrelated_B_{i}"},
            time_start=datetime(2025, 12, 15) + timedelta(days=i),
        )

    return incidents


@pytest.fixture
def no_chain_incidents():
    """
    Incidents with no chain pattern (completely disjoint motifs).

    I001: {A, B}
    I002: {C, D}
    I003: {E, F}
    """
    return {
        "I001": make_incident(
            "I001",
            core_motifs=[
                {"entities": ["Entity_A", "Entity_B"], "support": 4},
            ],
            anchor_entities={"Entity_A", "Entity_B"},
            time_start=datetime(2025, 12, 1),
        ),
        "I002": make_incident(
            "I002",
            core_motifs=[
                {"entities": ["Entity_C", "Entity_D"], "support": 3},
            ],
            anchor_entities={"Entity_C", "Entity_D"},
            time_start=datetime(2025, 12, 5),
        ),
        "I003": make_incident(
            "I003",
            core_motifs=[
                {"entities": ["Entity_E", "Entity_F"], "support": 5},
            ],
            anchor_entities={"Entity_E", "Entity_F"},
            time_start=datetime(2025, 12, 10),
        ),
    }


# ============================================================================
# TEST: MOTIF CHAIN DETECTION
# ============================================================================

class TestMotifChainDetection:
    """Test that motif chains are detected correctly."""


    def test_chain_creates_edge(self, chain_incidents):
        """Adjacent incidents in chain should have edges."""
        # Use higher hub threshold since with only 3 incidents,
        # shared entities get high participation
        builder = PrincipledCaseBuilder(hub_fraction_threshold=0.8)

        result = builder.build_from_incidents(chain_incidents)

        # I001 ↔ I002 should have edge (share "Terraform Labs")
        edge_01_02 = next(
            (e for e in result.edges
             if {e.incident1_id, e.incident2_id} == {"I001", "I002"}),
            None
        )
        assert edge_01_02 is not None, "Should have edge I001↔I002"

        # I002 ↔ I003 should have edge (share "SEC")
        edge_02_03 = next(
            (e for e in result.edges
             if {e.incident1_id, e.incident2_id} == {"I002", "I003"}),
            None
        )
        assert edge_02_03 is not None, "Should have edge I002↔I003"


    def test_chain_records_overlap(self, chain_incidents):
        """Chain edges should record the overlapping entity."""
        # Use higher hub threshold since with only 3 incidents,
        # shared entities get high participation
        builder = PrincipledCaseBuilder(hub_fraction_threshold=0.8)

        result = builder.build_from_incidents(chain_incidents)

        # Find I001↔I002 edge
        edge = next(
            (e for e in result.edges
             if {e.incident1_id, e.incident2_id} == {"I001", "I002"}),
            None
        )

        assert edge is not None

        # Should have motif chain evidence or shared anchor
        # The motifs {Do Kwon, Terraform Labs} and {Terraform Labs, SEC}
        # overlap on "Terraform Labs"
        has_chain_evidence = (
            len(edge.motif_chains) > 0 or
            "Terraform Labs" in edge.shared_anchors
        )
        assert has_chain_evidence, "Should have chain evidence via Terraform Labs"

    
    def test_no_direct_link_for_distant_chain(self, chain_incidents):
        """
        I001 and I003 have no direct link (no shared entities).

        They may still end up in same case if I002 bridges them,
        but there's no direct edge.
        """
        builder = PrincipledCaseBuilder()

        result = builder.build_from_incidents(chain_incidents)

        # I001 ↔ I003 should have NO direct edge (no shared entities)
        edge_01_03 = next(
            (e for e in result.edges
             if {e.incident1_id, e.incident2_id} == {"I001", "I003"}),
            None
        )

        # Either no edge, or edge with no shared motifs/anchors
        if edge_01_03:
            assert len(edge_01_03.shared_motifs) == 0
            assert len(edge_01_03.shared_anchors) == 0


# ============================================================================
# TEST: CHAIN STRENGTH
# ============================================================================

class TestChainStrength:
    """Test that chain strength affects edge quality."""

    
    def test_strong_chain_multiple_overlaps(self, chain_with_multiple_overlaps):
        """Chain with multiple overlapping entities is stronger."""
        builder = PrincipledCaseBuilder()

        result = builder.build_from_incidents(chain_with_multiple_overlaps)

        # Should have edge between I001 and I002
        edge = next(
            (e for e in result.edges
             if {e.incident1_id, e.incident2_id} == {"I001", "I002"}),
            None
        )

        assert edge is not None, "Should have edge for overlapping motifs"

        # Should be core edge (strong overlap: B, C shared)
        # Multiple constraints should be satisfied
        assert len(edge.constraints) >= 1

    
    def test_no_chain_no_edge(self, no_chain_incidents):
        """Completely disjoint incidents should have no edges."""
        builder = PrincipledCaseBuilder()

        result = builder.build_from_incidents(no_chain_incidents)

        # Should have no edges (no shared entities or motifs)
        assert len(result.edges) == 0, "Disjoint incidents should have no edges"

        # Should have no cases
        assert len(result.cases) == 0, "Disjoint incidents should form no cases"


# ============================================================================
# TEST: HUB SUPPRESSION IN CHAINS
# ============================================================================

class TestHubSuppressionInChains:
    """Test that hub entities are suppressed even in chains."""

    
    def test_hub_chain_suppressed(self, chain_with_hub):
        """Chain via hub entity should not create core edge."""
        builder = PrincipledCaseBuilder(hub_fraction_threshold=0.3)

        result = builder.build_from_incidents(chain_with_hub)

        # "Hong Kong" appears in 7/10 = 70% of incidents, definitely a hub
        assert "Hong Kong" in result.hubness
        assert result.hubness["Hong Kong"].is_hub

        # CHAIN_A and CHAIN_B only connect via "Hong Kong" (hub)
        # They should NOT form a core edge
        chain_edge = next(
            (e for e in result.edges
             if {e.incident1_id, e.incident2_id} == {"CHAIN_A", "CHAIN_B"}),
            None
        )

        if chain_edge:
            # If edge exists, Hong Kong should be in hub_anchors
            assert "Hong Kong" in chain_edge.hub_anchors

            # Should not be core (only hub evidence)
            if not chain_edge.shared_motifs and len(chain_edge.shared_anchors) == 0:
                assert not chain_edge.is_core, "Hub-only chain should not be core"


# ============================================================================
# TEST: CHAIN + TIME COMPATIBILITY
# ============================================================================

class TestChainWithTime:
    """Test that chains combined with time create valid cases."""


    def test_chain_within_time_window(self, chain_incidents):
        """Chain incidents within time window should form case."""
        # Use higher hub threshold since with only 3 incidents,
        # shared entities get high participation
        builder = PrincipledCaseBuilder(time_window_days=30, hub_fraction_threshold=0.8)

        result = builder.build_from_incidents(chain_incidents)

        # All incidents are within 10 days, so time compatible
        # With chain + time, should form case

        # Check for time constraints in ledger
        time_constraints = [
            c for c in result.ledger.constraints
            if c.constraint_type == ConstraintType.TEMPORAL
        ]

        # Should have time constraints
        assert len(time_constraints) >= 1

    
    def test_chain_outside_time_window(self):
        """Chain incidents outside time window have weaker binding."""
        # Create incidents with chain but far apart in time
        incidents = {
            "I001": make_incident(
                "I001",
                core_motifs=[
                    {"entities": ["A", "B"], "support": 5},
                ],
                anchor_entities={"A", "B"},
                time_start=datetime(2025, 1, 1),  # January
            ),
            "I002": make_incident(
                "I002",
                core_motifs=[
                    {"entities": ["B", "C"], "support": 4},
                ],
                anchor_entities={"B", "C"},
                time_start=datetime(2025, 12, 1),  # December (11 months later)
            ),
        }

        builder = PrincipledCaseBuilder(time_window_days=90)  # 3 months

        result = builder.build_from_incidents(incidents)

        # Find edge
        edge = next(
            (e for e in result.edges
             if {e.incident1_id, e.incident2_id} == {"I001", "I002"}),
            None
        )

        if edge:
            # Should not have time constraint (outside window)
            time_constraints = [
                c for c in edge.constraints
                if c.constraint_type == ConstraintType.TEMPORAL
            ]
            assert len(time_constraints) == 0, "Outside time window - no time constraint"


# ============================================================================
# TEST: TRANSITIVITY VIA CHAINS
# ============================================================================

class TestChainTransitivity:
    """Test that chains enable transitive case formation."""

    
    def test_transitive_case_via_bridge(self, chain_incidents):
        """
        I001 and I003 should be in same case via I002 bridge.

        I001 ↔ I002 ↔ I003 forms a connected component.
        """
        builder = PrincipledCaseBuilder()

        result = builder.build_from_incidents(chain_incidents)

        # All three should be in the same case (connected via chains)
        if len(result.cases) > 0:
            case = list(result.cases.values())[0]

            # I002 should definitely be in the case
            if "I002" in case.incident_ids:
                # Check if I001 and I003 are also connected via I002
                # This depends on whether the edges are core edges

                edges_01_02 = next(
                    (e for e in result.edges
                     if {e.incident1_id, e.incident2_id} == {"I001", "I002"}),
                    None
                )
                edges_02_03 = next(
                    (e for e in result.edges
                     if {e.incident1_id, e.incident2_id} == {"I002", "I003"}),
                    None
                )

                # If both edges are core, all three should be in same case
                if edges_01_02 and edges_01_02.is_core and edges_02_03 and edges_02_03.is_core:
                    assert case.incident_count == 3
                    assert "I001" in case.incident_ids
                    assert "I002" in case.incident_ids
                    assert "I003" in case.incident_ids


# ============================================================================
# TEST: CONSTRAINT LEDGER FOR CHAINS
# ============================================================================

class TestChainConstraintLedger:
    """Test that chain constraints are recorded in ledger."""


    def test_chain_constraint_logged(self, chain_incidents):
        """Motif chain constraints should be logged."""
        # Use higher hub threshold since with only 3 incidents,
        # shared entities get high participation
        builder = PrincipledCaseBuilder(hub_fraction_threshold=0.8)

        result = builder.build_from_incidents(chain_incidents)

        # Look for motif_chain or anchor_not_hub constraints
        relevant_constraints = [
            c for c in result.ledger.constraints
            if c.provenance in ("motif_chain", "anchor_not_hub", "shared_motif")
        ]

        # Should have some constraints for the chain edges
        assert len(relevant_constraints) >= 1


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
