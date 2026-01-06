"""
Case Builder Motif Recurrence Tests
====================================

Tests that shared motifs (k≥2 entities) across incidents → same case core.

Key invariants tested:
1. Shared motif (same k-set in both incidents) → core edge
2. Motif recurrence creates cases (not anchor intersection)
3. Multiple shared motifs strengthen the edge
4. Motifs from justification.core_motifs are consumed correctly

This addresses the real-world condition where anchor overlap is not
a valid sufficient statistic for L4 formation.
"""

import pytest
from datetime import datetime, timedelta
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reee.types import (
    Event, Surface, EventJustification, MembershipLevel, SurfaceMembership
)
from reee.builders.case_builder import (
    PrincipledCaseBuilder, CaseBuilderResult, MotifProfile, CaseEdge
)


# ============================================================================
# FIXTURES: Test Incidents
# ============================================================================

def make_incident(
    incident_id: str,
    core_motifs: list,
    anchor_entities: set,
    time_start: Optional[datetime] = None,
    total_claims: int = 10,
) -> Event:
    """Create a test incident with specified motifs."""
    justification = EventJustification(
        core_motifs=core_motifs,
        representative_surfaces=[f"S_{incident_id}_001"],
        canonical_handle=f"Test incident {incident_id}",
    )

    return Event(
        id=incident_id,
        surface_ids={f"S_{incident_id}_001", f"S_{incident_id}_002"},
        anchor_entities=anchor_entities,
        entities=anchor_entities.copy(),
        total_claims=total_claims,
        total_sources=3,
        time_window=(time_start, time_start + timedelta(days=1) if time_start else (None, None)),
        justification=justification,
    )


@pytest.fixture
def incident_do_kwon_1():
    """First Do Kwon incident with {Do Kwon, Terraform Labs} motif."""
    return make_incident(
        "I001",
        core_motifs=[
            {"entities": ["Do Kwon", "Terraform Labs"], "support": 5},
            {"entities": ["SEC", "Do Kwon"], "support": 3},
        ],
        anchor_entities={"Do Kwon", "Terraform Labs", "SEC"},
        time_start=datetime(2025, 12, 1),
    )


@pytest.fixture
def incident_do_kwon_2():
    """Second Do Kwon incident with same {Do Kwon, Terraform Labs} motif."""
    return make_incident(
        "I002",
        core_motifs=[
            {"entities": ["Do Kwon", "Terraform Labs"], "support": 4},
            {"entities": ["Montenegro", "Do Kwon"], "support": 2},
        ],
        anchor_entities={"Do Kwon", "Terraform Labs", "Montenegro"},
        time_start=datetime(2025, 12, 5),
    )


@pytest.fixture
def incident_do_kwon_3():
    """Third Do Kwon incident with same {Do Kwon, Terraform Labs} motif."""
    return make_incident(
        "I003",
        core_motifs=[
            {"entities": ["Do Kwon", "Terraform Labs"], "support": 6},
            {"entities": ["Judge Engelmayer", "Do Kwon"], "support": 3},
        ],
        anchor_entities={"Do Kwon", "Terraform Labs", "Judge Engelmayer"},
        time_start=datetime(2025, 12, 10),
    )


@pytest.fixture
def incident_hong_kong_fire():
    """Hong Kong fire incident - different motifs."""
    return make_incident(
        "I004",
        core_motifs=[
            {"entities": ["Hong Kong", "Tai Po"], "support": 8},
            {"entities": ["Fire", "Tai Po"], "support": 5},
        ],
        anchor_entities={"Hong Kong", "Tai Po", "Fire"},
        time_start=datetime(2025, 11, 26),
    )


@pytest.fixture
def incident_unrelated():
    """Completely unrelated incident."""
    return make_incident(
        "I005",
        core_motifs=[
            {"entities": ["Apple", "iPhone"], "support": 4},
        ],
        anchor_entities={"Apple", "iPhone", "Tim Cook"},
        time_start=datetime(2025, 12, 15),
    )


# ============================================================================
# TEST: SHARED MOTIF → SAME CASE
# ============================================================================

class TestSharedMotifCoreEdge:
    """Test that shared motifs create core edges."""

    def test_shared_motif_creates_core_edge(
        self, incident_do_kwon_1, incident_do_kwon_2
    ):
        """Two incidents sharing a motif should be connected by a core edge."""
        builder = PrincipledCaseBuilder()

        incidents = {
            incident_do_kwon_1.id: incident_do_kwon_1,
            incident_do_kwon_2.id: incident_do_kwon_2,
        }

        result = builder.build_from_incidents(incidents)

        # Should have edges between the two incidents
        assert len(result.edges) >= 1

        # Find edge between I001 and I002
        edge = next(
            (e for e in result.edges
             if {e.incident1_id, e.incident2_id} == {"I001", "I002"}),
            None
        )
        assert edge is not None, "Should have edge between I001 and I002"

        # Edge should be core (shared motif)
        assert edge.is_core, "Shared motif should create core edge"

        # Should have shared motif evidence
        assert len(edge.shared_motifs) >= 1
        assert frozenset(["Do Kwon", "Terraform Labs"]) in edge.shared_motifs

    def test_shared_motif_creates_case(
        self, incident_do_kwon_1, incident_do_kwon_2
    ):
        """Two incidents sharing a motif should form a case."""
        builder = PrincipledCaseBuilder()

        incidents = {
            incident_do_kwon_1.id: incident_do_kwon_1,
            incident_do_kwon_2.id: incident_do_kwon_2,
        }

        result = builder.build_from_incidents(incidents)

        # Should form exactly one case
        assert len(result.cases) == 1

        case = list(result.cases.values())[0]

        # Case should contain both incidents
        assert incident_do_kwon_1.id in case.incident_ids
        assert incident_do_kwon_2.id in case.incident_ids

    def test_three_incidents_one_case(
        self, incident_do_kwon_1, incident_do_kwon_2, incident_do_kwon_3
    ):
        """Three incidents sharing a motif should form one case."""
        # With 3 incidents all sharing {Do Kwon, Terraform Labs},
        # both entities have 100% participation which exceeds any threshold.
        # So we need to effectively disable hub detection for this test.
        builder = PrincipledCaseBuilder(hub_fraction_threshold=1.1)  # >100% = no hubs

        incidents = {
            incident_do_kwon_1.id: incident_do_kwon_1,
            incident_do_kwon_2.id: incident_do_kwon_2,
            incident_do_kwon_3.id: incident_do_kwon_3,
        }

        result = builder.build_from_incidents(incidents)

        # Should form exactly one case
        assert len(result.cases) == 1

        case = list(result.cases.values())[0]

        # Case should contain all three incidents
        assert case.incident_count == 3
        assert "I001" in case.incident_ids
        assert "I002" in case.incident_ids
        assert "I003" in case.incident_ids


# ============================================================================
# TEST: NO SHARED MOTIF → SEPARATE CASES
# ============================================================================

class TestNoSharedMotifSeparateCases:
    """Test that incidents without shared motifs stay separate."""

    def test_different_motifs_no_merge(
        self, incident_do_kwon_1, incident_hong_kong_fire
    ):
        """Incidents with different motifs should not merge."""
        builder = PrincipledCaseBuilder()

        incidents = {
            incident_do_kwon_1.id: incident_do_kwon_1,
            incident_hong_kong_fire.id: incident_hong_kong_fire,
        }

        result = builder.build_from_incidents(incidents)

        # Should NOT form cases (each incident is singleton)
        # min_incidents_for_case=2, so no cases should form
        assert len(result.cases) == 0, "Different motifs should not form case"

    def test_unrelated_incidents_separate(
        self, incident_do_kwon_1, incident_do_kwon_2, incident_unrelated
    ):
        """Unrelated incident should not join Do Kwon case."""
        # Use higher hub threshold since with only 3 incidents,
        # shared entities get high participation
        builder = PrincipledCaseBuilder(hub_fraction_threshold=0.7)

        incidents = {
            incident_do_kwon_1.id: incident_do_kwon_1,
            incident_do_kwon_2.id: incident_do_kwon_2,
            incident_unrelated.id: incident_unrelated,
        }

        result = builder.build_from_incidents(incidents)

        # Should form exactly one case (Do Kwon)
        assert len(result.cases) == 1

        case = list(result.cases.values())[0]

        # Case should NOT contain unrelated incident
        assert incident_unrelated.id not in case.incident_ids


# ============================================================================
# TEST: MOTIF PROFILE EXTRACTION
# ============================================================================

class TestMotifProfileExtraction:
    """Test that motif profiles are correctly extracted from justifications."""

    def test_profiles_extracted_from_justification(
        self, incident_do_kwon_1, incident_do_kwon_2
    ):
        """Motif profiles should be extracted from justification.core_motifs."""
        builder = PrincipledCaseBuilder()

        incidents = {
            incident_do_kwon_1.id: incident_do_kwon_1,
            incident_do_kwon_2.id: incident_do_kwon_2,
        }

        result = builder.build_from_incidents(incidents)

        # Should have profiles for both incidents
        assert "I001" in result.motif_profiles
        assert "I002" in result.motif_profiles

        profile1 = result.motif_profiles["I001"]
        profile2 = result.motif_profiles["I002"]

        # Profiles should have motifs
        assert len(profile1.motifs) >= 1
        assert len(profile2.motifs) >= 1

        # Check motif content
        assert frozenset(["Do Kwon", "Terraform Labs"]) in profile1.motifs
        assert frozenset(["Do Kwon", "Terraform Labs"]) in profile2.motifs

    def test_anchor_pair_motifs_extracted_even_without_justification(self):
        """Incident without justification should not fabricate motifs.

        L4 CaseCores are built from constrained motifs (incident.justification.core_motifs),
        not raw anchor pairs. Without a motif profile, the incident is underpowered
        for CaseCore merges and should rely on EntityCase/Topic views instead.
        """
        builder = PrincipledCaseBuilder()

        # Create incident without justification
        incident = Event(
            id="I_NO_JUST",
            surface_ids={"S001"},
            anchor_entities={"Entity1", "Entity2", "Entity3"},
            entities={"Entity1", "Entity2", "Entity3"},
            justification=None,  # No justification - but anchors are enough
        )

        incidents = {"I_NO_JUST": incident}

        result = builder.build_from_incidents(incidents)

        # Should have profile WITH anchor pair motifs
        assert "I_NO_JUST" in result.motif_profiles

        profile = result.motif_profiles["I_NO_JUST"]
        assert len(profile.motifs) == 0, "No justification.core_motifs → no constrained motifs"

        # Stats should count incidents WITH motifs (anchor pairs count)
        assert result.stats["incidents_with_motifs"] == 0


# ============================================================================
# TEST: CONSTRAINT LEDGER
# ============================================================================

class TestConstraintLedger:
    """Test that constraint ledger records evidence correctly."""

    def test_ledger_records_shared_motif(
        self, incident_do_kwon_1, incident_do_kwon_2
    ):
        """Constraint ledger should record shared_motif constraints."""
        builder = PrincipledCaseBuilder()

        incidents = {
            incident_do_kwon_1.id: incident_do_kwon_1,
            incident_do_kwon_2.id: incident_do_kwon_2,
        }

        result = builder.build_from_incidents(incidents)

        # Check ledger has constraints
        assert len(result.ledger.constraints) > 0

        # Find shared_motif constraint
        motif_constraints = [
            c for c in result.ledger.constraints
            if c.provenance == "shared_motif"
        ]

        assert len(motif_constraints) >= 1, "Should have shared_motif constraint"

        # Check constraint content
        mc = motif_constraints[0]
        assert "motif" in mc.evidence
        assert "Do Kwon" in mc.evidence["motif"] or "Terraform Labs" in mc.evidence["motif"]

    def test_ledger_records_case_formation(
        self, incident_do_kwon_1, incident_do_kwon_2
    ):
        """Constraint ledger should record case formation."""
        builder = PrincipledCaseBuilder()

        incidents = {
            incident_do_kwon_1.id: incident_do_kwon_1,
            incident_do_kwon_2.id: incident_do_kwon_2,
        }

        result = builder.build_from_incidents(incidents)

        # Find case_formation constraint
        formation_constraints = [
            c for c in result.ledger.constraints
            if c.provenance == "case_formation"
        ]

        assert len(formation_constraints) >= 1, "Should have case_formation constraint"


# ============================================================================
# TEST: STATS
# ============================================================================

class TestStats:
    """Test that stats are computed correctly."""

    def test_stats_computed(
        self, incident_do_kwon_1, incident_do_kwon_2, incident_do_kwon_3
    ):
        """Stats should reflect the building process."""
        # With 3 incidents all sharing same entities,
        # disable hub detection to allow case formation
        builder = PrincipledCaseBuilder(hub_fraction_threshold=1.1)

        incidents = {
            incident_do_kwon_1.id: incident_do_kwon_1,
            incident_do_kwon_2.id: incident_do_kwon_2,
            incident_do_kwon_3.id: incident_do_kwon_3,
        }

        result = builder.build_from_incidents(incidents)

        assert result.stats["incidents"] == 3
        assert result.stats["profiles_extracted"] == 3
        assert result.stats["unique_motifs"] >= 1
        assert result.stats["cases_formed"] == 1
        assert result.stats["core_edges"] >= 1


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
