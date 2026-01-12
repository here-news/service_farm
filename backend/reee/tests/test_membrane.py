"""
Membrane Contract Tests: Adversarial Scenarios
===============================================

Tests for the membrane decision table, with emphasis on adversarial patterns
from real production leaks (WFC postmortem).

Test categories:
1. Contract tests (basic decision table)
2. WFC leak replay (real incidents that leaked)
3. Co-incident hub patterns (hub entity in multiple stories)
4. Scope pollution (constraint kind spoofing)
5. Semantic-only trap (anti-trap rule)

These tests validate the membrane contract prevents contamination.
"""

import pytest
from typing import Set, Dict, Any, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reee.membrane import (
    Membership, CoreReason, LinkType,
    FocalSet, MembershipDecision,
    classify_incident_membership,
    can_promote_to_multi_spine,
    semantic_cannot_force_core,
    is_structural_witness,
    is_semantic_only,
    WITNESS_KINDS, SEMANTIC_KINDS,
    compute_core_leak_rate,
    compute_witness_scarcity,
)


# =============================================================================
# FIXTURES: Hub entity sets
# =============================================================================

@pytest.fixture
def hk_hubs() -> Set[str]:
    """Hub entities from Hong Kong news corpus."""
    return {
        "Hong Kong",
        "John Lee",          # Chief Executive (appears in everything government-related)
        "China",
        "United States",
    }


@pytest.fixture
def wfc_focal() -> FocalSet:
    """Wang Fuk Court fire focal set."""
    return FocalSet(primary="Wang Fuk Court")


# =============================================================================
# SECTION 1: BASIC CONTRACT TESTS
# =============================================================================

class TestMembraneContract:
    """Basic membrane decision table tests."""

    def test_core_a_single_spine_anchor(self, hk_hubs, wfc_focal):
        """Core-A: spine is anchor in incident."""
        result = classify_incident_membership(
            incident_anchors={"Wang Fuk Court", "Fire Services"},
            focal_set=wfc_focal,
            constraints=[],
            hub_entities=hk_hubs,
        )
        assert result.membership == Membership.CORE
        assert result.core_reason == CoreReason.ANCHOR
        assert result.link_type == LinkType.MEMBER

    def test_core_b_with_two_structural_witnesses(self, hk_hubs, wfc_focal):
        """Core-B: two structural witnesses, one non-time."""
        result = classify_incident_membership(
            incident_anchors={"Tai Po", "Chris Tang"},
            focal_set=wfc_focal,
            constraints=[
                {"id": "c1", "kind": "geo", "source_entity": "Tai Po"},
                {"id": "c2", "kind": "time", "source_entity": ""},
            ],
            hub_entities=hk_hubs,
        )
        assert result.membership == Membership.CORE
        assert result.core_reason == CoreReason.WARRANT
        assert len(result.witnesses) == 2

    def test_periphery_time_only_witness(self, hk_hubs, wfc_focal):
        """Periphery: time-only witness is insufficient."""
        result = classify_incident_membership(
            incident_anchors={"Tai Po", "Chris Tang"},
            focal_set=wfc_focal,
            constraints=[
                {"id": "c1", "kind": "time", "source_entity": ""},
            ],
            hub_entities=hk_hubs,
        )
        assert result.membership == Membership.PERIPHERY
        assert "need" in result.blocked_reason.lower()

    def test_periphery_single_non_time_witness(self, hk_hubs, wfc_focal):
        """Periphery: single non-time witness is insufficient (need >=2)."""
        result = classify_incident_membership(
            incident_anchors={"Tai Po", "Chris Tang"},
            focal_set=wfc_focal,
            constraints=[
                {"id": "c1", "kind": "geo", "source_entity": "Tai Po"},
            ],
            hub_entities=hk_hubs,
        )
        assert result.membership == Membership.PERIPHERY
        assert "1 witness" in result.blocked_reason

    def test_reject_hub_only_anchors(self, hk_hubs, wfc_focal):
        """Reject: incident has only hub anchors."""
        result = classify_incident_membership(
            incident_anchors={"Hong Kong", "John Lee"},
            focal_set=wfc_focal,
            constraints=[],
            hub_entities=hk_hubs,
        )
        assert result.membership == Membership.REJECT
        assert "hub" in result.blocked_reason.lower()


# =============================================================================
# SECTION 2: WFC LEAK REPLAY (Real Production Leaks)
# =============================================================================

class TestWFCLeakReplay:
    """
    Replay real leaked incidents from WFC postmortem.

    These are incidents that were incorrectly classified as CORE
    in the original system. The membrane must reject or periphery them.
    """

    def test_jimmy_lai_trial_rejected(self, hk_hubs, wfc_focal):
        """
        Jimmy Lai trial incident must be REJECT.

        From postmortem: "Jimmy Lai, Apple Daily" incident was merged
        into WFC story only because both share hub "John Lee".
        """
        result = classify_incident_membership(
            incident_anchors={"Jimmy Lai", "Apple Daily", "CCP", "John Lee"},
            focal_set=wfc_focal,
            constraints=[],  # No structural connection to WFC
            hub_entities=hk_hubs,
        )
        # Must not be CORE - this was the smoking gun leak
        assert result.membership != Membership.CORE, \
            "Jimmy Lai trial must not be Core in WFC story"

        # Best: REJECT (unrelated)
        # Acceptable: PERIPHERY (but never persisted)
        assert result.membership in [Membership.REJECT, Membership.PERIPHERY]

    def test_wong_kwok_ngon_is_related_not_core(self, hk_hubs, wfc_focal):
        """
        Wong Kwok-ngon sedition saga is RELATED but not CORE.

        From postmortem: "charged with sedition over fire videos" is
        a separate storyline spawned by WFC fire, not part of it.
        """
        result = classify_incident_membership(
            incident_anchors={"Wong Kwok-ngon", "YouTube", "West Kowloon Courts"},
            focal_set=wfc_focal,
            constraints=[
                # Even if there's time proximity, no geo/event witness for WFC
                {"id": "c1", "kind": "time", "source_entity": ""},
            ],
            hub_entities=hk_hubs,
        )
        # Must not be CORE
        assert result.membership != Membership.CORE, \
            "Wong Kwok-ngon saga is separate storyline, not Core WFC"

    def test_tai_po_blaze_core_b_with_geo_witness(self, hk_hubs, wfc_focal):
        """
        "Tai Po blaze" incident CAN be Core-B with proper witnesses.

        This tests that legitimate Core-B candidates are accepted
        when they have structural evidence (geo containment).
        """
        result = classify_incident_membership(
            incident_anchors={"Tai Po District", "Fire Services"},
            focal_set=wfc_focal,
            constraints=[
                {"id": "c1", "kind": "geo", "source_entity": "Tai Po District"},
                {"id": "c2", "kind": "time", "source_entity": ""},
            ],
            hub_entities=hk_hubs,
        )
        assert result.membership == Membership.CORE
        assert result.core_reason == CoreReason.WARRANT

    def test_john_lee_speech_periphery_without_witness(self, hk_hubs, wfc_focal):
        """
        John Lee speech about fire: PERIPHERY without geo witness.

        Even if John Lee mentions WFC, he's a hub. His speeches don't
        automatically become Core just because he's mentioned.
        """
        result = classify_incident_membership(
            incident_anchors={"John Lee", "Legislative Council"},
            focal_set=wfc_focal,
            constraints=[
                # Only time overlap, no geo/event witness
                {"id": "c1", "kind": "time", "source_entity": ""},
            ],
            hub_entities=hk_hubs,
        )
        assert result.membership == Membership.PERIPHERY

    def test_chris_tang_fire_response_core_b(self, hk_hubs, wfc_focal):
        """
        Chris Tang (Security Secretary) fire response: Core-B possible.

        If there's geo+time+event_type witness, this is valid Core-B.
        """
        result = classify_incident_membership(
            incident_anchors={"Chris Tang", "Tai Po"},
            focal_set=wfc_focal,
            constraints=[
                {"id": "c1", "kind": "geo", "source_entity": "Tai Po"},
                {"id": "c2", "kind": "event_type", "source_entity": "fire_response"},
            ],
            hub_entities=hk_hubs,
        )
        assert result.membership == Membership.CORE
        assert result.core_reason == CoreReason.WARRANT


# =============================================================================
# SECTION 3: CO-INCIDENT HUB PATTERNS
# =============================================================================

class TestCoIncidentHub:
    """
    Test that hub entities appearing in multiple stories
    cannot create cross-story contamination.
    """

    def test_hub_cannot_bridge_stories(self, hk_hubs):
        """Two different stories share hub; hub cannot bridge them."""
        # Story 1: WFC fire
        focal_wfc = FocalSet(primary="Wang Fuk Court")

        # Story 2: Immigration controversy
        focal_immigration = FocalSet(primary="Immigration Department")

        # Incident mentions both stories' spines AND hub
        incident_anchors = {"Immigration Department", "John Lee"}

        # Test against WFC story: should not be Core
        result_wfc = classify_incident_membership(
            incident_anchors=incident_anchors,
            focal_set=focal_wfc,
            constraints=[],
            hub_entities=hk_hubs,
        )
        assert result_wfc.membership != Membership.CORE, \
            "Hub cannot bridge unrelated stories"

        # Test against Immigration story: should be Core-A (spine is anchor)
        result_imm = classify_incident_membership(
            incident_anchors=incident_anchors,
            focal_set=focal_immigration,
            constraints=[],
            hub_entities=hk_hubs,
        )
        assert result_imm.membership == Membership.CORE
        assert result_imm.core_reason == CoreReason.ANCHOR

    def test_hub_witness_ignored(self, hk_hubs, wfc_focal):
        """Witnesses sourced from hub entities are ignored."""
        result = classify_incident_membership(
            incident_anchors={"Tai Po", "Chris Tang"},
            focal_set=wfc_focal,
            constraints=[
                # Hub-sourced witness should be ignored
                {"id": "c1", "kind": "geo", "source_entity": "Hong Kong"},
                # Non-hub witness counts
                {"id": "c2", "kind": "time", "source_entity": ""},
            ],
            hub_entities=hk_hubs,
        )
        # Only 1 valid witness (time), hub-sourced geo is ignored
        assert result.membership == Membership.PERIPHERY
        assert len(result.witnesses) == 1

    def test_multiple_hubs_still_rejected(self, hk_hubs, wfc_focal):
        """Multiple hub anchors don't sum to non-hub."""
        result = classify_incident_membership(
            incident_anchors={"Hong Kong", "John Lee", "China", "United States"},
            focal_set=wfc_focal,
            constraints=[],
            hub_entities=hk_hubs,
        )
        assert result.membership == Membership.REJECT


# =============================================================================
# SECTION 4: SCOPE POLLUTION (Constraint Kind Spoofing)
# =============================================================================

class TestScopePollution:
    """
    Test that semantic constraints cannot masquerade as structural.
    """

    def test_semantic_kinds_not_structural(self):
        """SEMANTIC_KINDS must not be in WITNESS_KINDS."""
        overlap = WITNESS_KINDS & SEMANTIC_KINDS
        assert len(overlap) == 0, f"Overlap detected: {overlap}"

    def test_embedding_constraint_not_witness(self, hk_hubs, wfc_focal):
        """Embedding constraint cannot provide Core-B witness."""
        result = classify_incident_membership(
            incident_anchors={"Tai Po", "Some Entity"},
            focal_set=wfc_focal,
            constraints=[
                {"id": "c1", "kind": "embedding", "source_entity": ""},
                {"id": "c2", "kind": "embedding", "source_entity": ""},
                {"id": "c3", "kind": "embedding", "source_entity": ""},
            ],
            hub_entities=hk_hubs,
        )
        # No structural witnesses, so not Core
        assert result.membership == Membership.PERIPHERY
        assert len(result.witnesses) == 0

    def test_llm_proposal_not_witness(self, hk_hubs, wfc_focal):
        """LLM proposal cannot provide Core-B witness."""
        result = classify_incident_membership(
            incident_anchors={"Tai Po", "Some Entity"},
            focal_set=wfc_focal,
            constraints=[
                {"id": "c1", "kind": "llm_proposal", "source_entity": ""},
                {"id": "c2", "kind": "llm_proposal", "source_entity": ""},
            ],
            hub_entities=hk_hubs,
        )
        assert result.membership == Membership.PERIPHERY

    def test_title_similarity_not_witness(self, hk_hubs, wfc_focal):
        """Title similarity cannot provide Core-B witness."""
        result = classify_incident_membership(
            incident_anchors={"Tai Po", "Some Entity"},
            focal_set=wfc_focal,
            constraints=[
                {"id": "c1", "kind": "title_similarity", "source_entity": ""},
                {"id": "c2", "kind": "title_similarity", "source_entity": ""},
            ],
            hub_entities=hk_hubs,
        )
        assert result.membership == Membership.PERIPHERY

    def test_mixed_semantic_structural_partial(self, hk_hubs, wfc_focal):
        """Mixed constraints: only structural ones count."""
        result = classify_incident_membership(
            incident_anchors={"Tai Po", "Some Entity"},
            focal_set=wfc_focal,
            constraints=[
                {"id": "c1", "kind": "embedding", "source_entity": ""},      # Ignored
                {"id": "c2", "kind": "geo", "source_entity": "Tai Po"},      # Counts
                {"id": "c3", "kind": "llm_proposal", "source_entity": ""},   # Ignored
                {"id": "c4", "kind": "time", "source_entity": ""},           # Counts
            ],
            hub_entities=hk_hubs,
        )
        # 2 structural witnesses (geo + time), geo is non-time
        assert result.membership == Membership.CORE
        assert result.core_reason == CoreReason.WARRANT
        assert len(result.witnesses) == 2


# =============================================================================
# SECTION 5: SEMANTIC-ONLY TRAP (Anti-Trap Rule)
# =============================================================================

class TestSemanticOnlyTrap:
    """
    Test the anti-trap rule: semantic-only evidence cannot force core.
    """

    def test_semantic_cannot_force_core_empty(self):
        """Empty constraints: semantic cannot force core."""
        has_structural, reason = semantic_cannot_force_core([])
        assert not has_structural

    def test_semantic_cannot_force_core_all_semantic(self):
        """All semantic constraints: cannot force core."""
        has_structural, reason = semantic_cannot_force_core([
            {"kind": "embedding"},
            {"kind": "llm_proposal"},
            {"kind": "title_similarity"},
        ])
        assert not has_structural
        assert "semantic-only" in reason.lower()

    def test_semantic_cannot_force_core_has_structural(self):
        """Has structural witness: can potentially be core."""
        has_structural, reason = semantic_cannot_force_core([
            {"kind": "embedding"},
            {"kind": "geo"},
        ])
        assert has_structural
        assert "geo" in reason


# =============================================================================
# SECTION 6: MULTI-SPINE PROMOTION TESTS
# =============================================================================

class TestMultiSpinePromotion:
    """Test multi-spine (dyad/set) promotion gates."""

    def test_hub_cannot_be_co_spine(self, hk_hubs):
        """Hub entities cannot become co-spines."""
        can_promote, reason = can_promote_to_multi_spine(
            primary="Wang Fuk Court",
            candidate_co_spine="Hong Kong",  # Hub
            co_occurrence_count=100,
            total_incidents=200,
            pmi_score=5.0,
            hub_entities=hk_hubs,
        )
        assert not can_promote
        assert "hub" in reason.lower()

    def test_primary_hub_blocked(self, hk_hubs):
        """Primary cannot be a hub."""
        can_promote, reason = can_promote_to_multi_spine(
            primary="John Lee",  # Hub
            candidate_co_spine="Some Entity",
            co_occurrence_count=50,
            total_incidents=100,
            pmi_score=3.0,
            hub_entities=hk_hubs,
        )
        assert not can_promote

    def test_insufficient_co_occurrences(self, hk_hubs):
        """Need >=3 co-occurrences."""
        can_promote, reason = can_promote_to_multi_spine(
            primary="Do Kwon",
            candidate_co_spine="Terraform Labs",
            co_occurrence_count=2,  # Too few
            total_incidents=100,
            pmi_score=5.0,
            hub_entities=hk_hubs,
        )
        assert not can_promote
        assert "co-occurrences" in reason

    def test_low_pmi_blocked(self, hk_hubs):
        """Need PMI >= 2.0."""
        can_promote, reason = can_promote_to_multi_spine(
            primary="Do Kwon",
            candidate_co_spine="Terraform Labs",
            co_occurrence_count=10,
            total_incidents=100,
            pmi_score=1.5,  # Too low
            hub_entities=hk_hubs,
        )
        assert not can_promote
        assert "PMI" in reason

    def test_valid_dyad_promotion(self, hk_hubs):
        """Valid dyad promotion passes all gates."""
        can_promote, reason = can_promote_to_multi_spine(
            primary="Do Kwon",
            candidate_co_spine="Terraform Labs",
            co_occurrence_count=15,
            total_incidents=100,
            pmi_score=4.5,
            hub_entities=hk_hubs,
        )
        assert can_promote
        assert "15" in reason  # co-occurrence count
        assert "4.5" in reason  # PMI


# =============================================================================
# SECTION 7: DYAD FOCAL SET TESTS
# =============================================================================

class TestDyadFocalSet:
    """Test Core-A behavior with dyad focal sets."""

    def test_dyad_requires_both_anchors(self, hk_hubs):
        """Dyad Core-A requires both spines as anchors."""
        focal = FocalSet(
            primary="Do Kwon",
            co_spines={"Terraform Labs"},
            kind="dyad",
        )

        # Only one spine as anchor: not Core-A
        result = classify_incident_membership(
            incident_anchors={"Do Kwon", "SEC"},
            focal_set=focal,
            constraints=[],
            hub_entities=hk_hubs,
        )
        # Doesn't meet dyad Core-A (both spines required)
        # Falls through to other checks
        assert result.membership != Membership.CORE or result.core_reason != CoreReason.ANCHOR

    def test_dyad_both_anchors_core_a(self, hk_hubs):
        """Dyad with both spines as anchors: Core-A."""
        focal = FocalSet(
            primary="Do Kwon",
            co_spines={"Terraform Labs"},
            kind="dyad",
        )

        result = classify_incident_membership(
            incident_anchors={"Do Kwon", "Terraform Labs", "SEC"},
            focal_set=focal,
            constraints=[],
            hub_entities=hk_hubs,
        )
        assert result.membership == Membership.CORE
        assert result.core_reason == CoreReason.ANCHOR


# =============================================================================
# SECTION 8: METRICS TESTS
# =============================================================================

class TestMembraneMetrics:
    """Test membrane health metrics."""

    def test_core_leak_rate_zero(self):
        """Perfect: all core incidents have spine anchor."""
        core_ids = {"i1", "i2", "i3"}
        with_anchor = {"i1", "i2", "i3"}
        rate = compute_core_leak_rate(core_ids, with_anchor)
        assert rate == 0.0

    def test_core_leak_rate_partial(self):
        """Partial: some core incidents lack anchor."""
        core_ids = {"i1", "i2", "i3"}
        with_anchor = {"i1", "i2"}  # i3 leaked
        rate = compute_core_leak_rate(core_ids, with_anchor)
        assert abs(rate - 1/3) < 0.01

    def test_core_leak_rate_empty(self):
        """Empty core: rate is 0."""
        rate = compute_core_leak_rate(set(), set())
        assert rate == 0.0

    def test_witness_scarcity_high(self):
        """High scarcity: many candidates blocked."""
        rate = compute_witness_scarcity(blocked_count=80, candidate_count=100)
        assert rate == 0.8

    def test_witness_scarcity_low(self):
        """Low scarcity: few candidates blocked."""
        rate = compute_witness_scarcity(blocked_count=5, candidate_count=100)
        assert rate == 0.05


# =============================================================================
# SECTION 9: HELPER FUNCTION TESTS
# =============================================================================

class TestHelperFunctions:
    """Test helper functions for witness classification."""

    def test_is_structural_witness(self):
        """Test structural witness kinds."""
        assert is_structural_witness("time")
        assert is_structural_witness("geo")
        assert is_structural_witness("event_type")
        assert is_structural_witness("motif")
        assert is_structural_witness("context")

        assert not is_structural_witness("embedding")
        assert not is_structural_witness("llm_proposal")
        assert not is_structural_witness("title_similarity")
        assert not is_structural_witness("unknown")

    def test_is_semantic_only(self):
        """Test semantic-only kinds."""
        assert is_semantic_only("embedding")
        assert is_semantic_only("llm_proposal")
        assert is_semantic_only("title_similarity")

        assert not is_semantic_only("time")
        assert not is_semantic_only("geo")
        assert not is_semantic_only("unknown")


# =============================================================================
# SECTION 10: FOCAL SET IDENTITY TESTS
# =============================================================================

class TestFocalSetIdentity:
    """Test FocalSet identity key generation."""

    def test_single_spine_identity(self):
        """Single spine identity key."""
        focal = FocalSet(primary="Wang Fuk Court")
        assert focal.identity_key() == "Wang Fuk Court"

    def test_dyad_identity_canonical(self):
        """Dyad identity key is sorted."""
        focal1 = FocalSet(primary="A", co_spines={"B"}, kind="dyad")
        focal2 = FocalSet(primary="B", co_spines={"A"}, kind="dyad")

        # Both should produce same identity key
        assert focal1.identity_key() == focal2.identity_key()
        assert focal1.identity_key() == "A|B"

    def test_all_spines_includes_primary(self):
        """all_spines() includes primary."""
        focal = FocalSet(primary="A", co_spines={"B", "C"}, kind="set")
        assert focal.all_spines() == {"A", "B", "C"}


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
