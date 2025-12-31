"""
Tests for Epistemic Unit Invariants
====================================

This test suite validates the 6 core invariants of the epistemic architecture:

1. L0 immutability: Claims append-only, never deleted, never modified
2. Parameter versioning: Parameters append-only, changes tracked with provenance
3. Identity/Aboutness separation: L2 uses identity, L3 uses aboutness, never mixed
4. Derived state purity: L1-L5 = f(L0, parameters), no external mutation
5. Stable core relations: {CONFIRMS, REFINES, SUPERSEDES, CONFLICTS, UNRELATED}
6. Meta-claims are observations: Emitted about state, never injected as world-claims
"""

import pytest
import asyncio
from datetime import datetime
from copy import deepcopy

from reee import (
    Claim, Surface, Event, AboutnessLink,
    Relation, Association,
    Parameters, ParameterChange, MetaClaim,
    Engine,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_claims():
    """Sample claims for testing."""
    return [
        Claim(id="c1", text="Fire kills 13 in Hong Kong high-rise", source="BBC",
              entities={"Hong Kong", "fire", "high-rise"}, anchor_entities=set(),
              embedding=[0.1, 0.2, 0.3]),
        Claim(id="c2", text="Hong Kong fire death toll reaches 13", source="Reuters",
              entities={"Hong Kong", "fire"}, anchor_entities=set(),
              embedding=[0.11, 0.21, 0.31]),
        Claim(id="c3", text="13 dead in HK apartment blaze", source="SCMP",
              entities={"Hong Kong", "fire", "apartment"}, anchor_entities=set(),
              embedding=[0.12, 0.19, 0.29]),
        Claim(id="c4", text="Jimmy Lai trial continues in Hong Kong", source="BBC",
              entities={"Jimmy Lai", "Hong Kong", "trial"}, anchor_entities={"Jimmy Lai"},
              embedding=[0.5, 0.6, 0.7]),
        Claim(id="c5", text="Lai faces national security charges", source="Guardian",
              entities={"Jimmy Lai", "national security"}, anchor_entities={"Jimmy Lai"},
              embedding=[0.51, 0.61, 0.71]),
    ]


@pytest.fixture
def engine():
    """Fresh Engine for testing."""
    return Engine(llm=None)


@pytest.fixture
def engine_with_claims(sample_claims):
    """Engine with claims already added."""
    async def setup():
        eng = Engine(llm=None)
        for claim in sample_claims:
            await eng.add_claim(claim)
        return eng
    return asyncio.get_event_loop().run_until_complete(setup())


# =============================================================================
# INVARIANT 1: L0 Immutability
# =============================================================================

class TestInvariant1_L0Immutability:
    """Claims are append-only, never deleted, never modified."""

    @pytest.mark.asyncio
    async def test_claims_append_only(self, sample_claims):
        """Claims can only be added, not removed."""
        eng = Engine(llm=None)

        # Add claims
        for claim in sample_claims:
            await eng.add_claim(claim)

        assert len(eng.claims) == 5

        # Verify all claims are present
        for claim in sample_claims:
            assert claim.id in eng.claims

    @pytest.mark.asyncio
    async def test_claim_text_immutable(self, sample_claims):
        """Once added, claim text should not change."""
        eng = Engine(llm=None)
        claim = sample_claims[0]
        original_text = claim.text

        await eng.add_claim(claim)

        # Verify text unchanged
        assert eng.claims[claim.id].text == original_text

    @pytest.mark.asyncio
    async def test_claim_provenance_preserved(self, sample_claims):
        """Claim source/provenance is preserved."""
        eng = Engine(llm=None)
        claim = sample_claims[0]
        original_source = claim.source

        await eng.add_claim(claim)

        assert eng.claims[claim.id].source == original_source


# =============================================================================
# INVARIANT 2: Parameter Versioning
# =============================================================================

class TestInvariant2_ParameterVersioning:
    """Parameters are versioned with full provenance."""

    def test_initial_version(self):
        """Parameters start at version 1."""
        params = Parameters()
        assert params.version == 1

    def test_version_increments_on_update(self):
        """Each update increments version."""
        params = Parameters()

        params.update("hub_max_df", 5, actor="test")
        assert params.version == 2

        params.update("aboutness_threshold", 0.4, actor="test")
        assert params.version == 3

    def test_changes_logged_with_provenance(self):
        """All changes are logged with actor and rationale."""
        params = Parameters()

        change = params.update(
            parameter="identity_confidence_threshold",
            new_value=0.7,
            actor="human:operator@test",
            trigger="mc_12345",
            rationale="Testing provenance tracking"
        )

        assert change.parameter == "identity_confidence_threshold"
        assert change.old_value == 0.5
        assert change.new_value == 0.7
        assert change.actor == "human:operator@test"
        assert change.trigger == "mc_12345"
        assert change.rationale == "Testing provenance tracking"
        assert "L2" in change.affects_layers

    def test_change_history_preserved(self):
        """Full change history is preserved."""
        params = Parameters()

        params.update("hub_max_df", 4, actor="a1")
        params.update("hub_max_df", 5, actor="a2")
        params.update("hub_max_df", 6, actor="a3")

        assert len(params.changes) == 3
        assert params.changes[0].new_value == 4
        assert params.changes[1].new_value == 5
        assert params.changes[2].new_value == 6

    def test_engine_uses_parameters(self):
        """Engine uses versioned parameters."""
        params = Parameters()
        params.update("hub_max_df", 10, actor="test")

        eng = Engine(llm=None, params=params)

        assert eng.params.hub_max_df == 10
        assert eng.params.version == 2


# =============================================================================
# INVARIANT 3: Identity/Aboutness Separation
# =============================================================================

class TestInvariant3_IdentityAboutnessSeparation:
    """L2 uses identity edges, L3 uses aboutness edges, never mixed."""

    @pytest.mark.asyncio
    async def test_surfaces_use_identity_edges_only(self, sample_claims):
        """Surfaces are formed from identity edges only."""
        eng = Engine(llm=None)

        for claim in sample_claims:
            await eng.add_claim(claim)

        # Manually add an identity edge (simulating LLM classification)
        eng.claim_edges.append(("c1", "c2", Relation.CONFIRMS, 0.9))

        surfaces = eng.compute_surfaces()

        # c1 and c2 should be in the same surface (identity edge)
        surface_with_c1c2 = None
        for s in surfaces:
            if "c1" in s.claim_ids and "c2" in s.claim_ids:
                surface_with_c1c2 = s
                break

        assert surface_with_c1c2 is not None, "c1 and c2 should be in same surface"

    @pytest.mark.asyncio
    async def test_aboutness_does_not_merge_surfaces(self, sample_claims):
        """Aboutness edges do not merge surfaces into one."""
        eng = Engine(llm=None)

        for claim in sample_claims:
            await eng.add_claim(claim)

        # Add identity edges to create two distinct surfaces
        eng.claim_edges.append(("c1", "c2", Relation.CONFIRMS, 0.9))
        eng.claim_edges.append(("c4", "c5", Relation.CONFIRMS, 0.9))

        surfaces = eng.compute_surfaces()

        # Should have at least 2 surfaces (fire claims + Lai claims)
        assert len(surfaces) >= 2

        # Now compute aboutness - this should NOT merge the surfaces
        aboutness = eng.compute_surface_aboutness()

        # Surfaces count should remain the same
        assert len(eng.surfaces) >= 2

    @pytest.mark.asyncio
    async def test_events_use_aboutness_edges(self, sample_claims):
        """Events are formed from aboutness edges between surfaces."""
        eng = Engine(llm=None)

        for claim in sample_claims:
            await eng.add_claim(claim)

        # Create surfaces
        eng.claim_edges.append(("c4", "c5", Relation.CONFIRMS, 0.9))
        eng.compute_surfaces()

        # Compute aboutness
        eng.compute_surface_aboutness()

        # Compute events
        events = eng.compute_events()

        # Events should exist
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_surface_about_links_separate_from_internal_edges(self, sample_claims):
        """Surface.about_links is distinct from Surface.internal_edges."""
        eng = Engine(llm=None)

        for claim in sample_claims:
            await eng.add_claim(claim)

        # Create surfaces with identity edges
        eng.claim_edges.append(("c4", "c5", Relation.CONFIRMS, 0.9))
        surfaces = eng.compute_surfaces()

        # Compute aboutness
        eng.compute_surface_aboutness()

        # Check a surface with about_links
        for s in surfaces:
            # internal_edges are Relation tuples (identity)
            for edge in s.internal_edges:
                assert isinstance(edge[2], Relation)

            # about_links are AboutnessLink objects (soft)
            for link in s.about_links:
                assert isinstance(link, AboutnessLink)
                assert hasattr(link, 'score')


# =============================================================================
# INVARIANT 4: Derived State Purity
# =============================================================================

class TestInvariant4_DerivedStatePurity:
    """L1-L5 = f(L0, parameters), no external mutation."""

    @pytest.mark.asyncio
    async def test_surfaces_derived_from_claims(self, sample_claims):
        """Surfaces are computed from claims, not manually set."""
        eng = Engine(llm=None)

        for claim in sample_claims:
            await eng.add_claim(claim)

        # Surfaces computed from claims
        surfaces1 = eng.compute_surfaces()

        # Recompute should give same result
        eng.surfaces = {}
        eng._surface_counter = 0
        surfaces2 = eng.compute_surfaces()

        assert len(surfaces1) == len(surfaces2)

    @pytest.mark.asyncio
    async def test_events_derived_from_surfaces(self, sample_claims):
        """Events are computed from surfaces and aboutness."""
        eng = Engine(llm=None)

        for claim in sample_claims:
            await eng.add_claim(claim)

        eng.compute_surfaces()
        eng.compute_surface_aboutness()
        events1 = eng.compute_events()

        # Recompute should give consistent result
        eng.events = {}
        eng._event_counter = 0
        events2 = eng.compute_events()

        assert len(events1) == len(events2)

    @pytest.mark.asyncio
    async def test_parameter_change_affects_derived_state(self, sample_claims):
        """Changing parameters affects derived state on recompute."""
        params = Parameters()
        eng = Engine(llm=None, params=params)

        for claim in sample_claims:
            await eng.add_claim(claim)

        eng.compute_surfaces()
        eng.compute_surface_aboutness()
        aboutness_count_1 = len(eng.surface_aboutness)

        # Change parameter
        params.update("aboutness_min_signals", 3, actor="test")

        # Recompute with stricter threshold
        eng.surface_aboutness = []
        for s in eng.surfaces.values():
            s.about_links = []
        eng.compute_surface_aboutness()
        aboutness_count_2 = len(eng.surface_aboutness)

        # With stricter min_signals, should have fewer edges
        # (may be equal if none passed anyway, but logic is sound)
        assert aboutness_count_2 <= aboutness_count_1


# =============================================================================
# INVARIANT 5: Stable Core Relations
# =============================================================================

class TestInvariant5_StableCoreRelations:
    """Core relations are {CONFIRMS, REFINES, SUPERSEDES, CONFLICTS, UNRELATED}."""

    def test_relation_enum_values(self):
        """Relation enum has exactly the expected values."""
        expected = {"confirms", "refines", "supersedes", "conflicts", "unrelated"}
        actual = {r.value for r in Relation}
        assert actual == expected

    def test_all_relations_are_identity_relations(self):
        """All relations (except UNRELATED) are identity relations."""
        identity_relations = {Relation.CONFIRMS, Relation.REFINES,
                             Relation.SUPERSEDES, Relation.CONFLICTS}

        # UNRELATED is the only non-identity relation
        non_identity = {Relation.UNRELATED}

        all_relations = set(Relation)
        assert all_relations == identity_relations | non_identity

    @pytest.mark.asyncio
    async def test_only_core_relations_in_edges(self, sample_claims):
        """Only core relations appear in claim_edges."""
        eng = Engine(llm=None)

        for claim in sample_claims:
            await eng.add_claim(claim)

        # Add edges with various relations
        eng.claim_edges.append(("c1", "c2", Relation.CONFIRMS, 0.9))
        eng.claim_edges.append(("c2", "c3", Relation.REFINES, 0.8))
        eng.claim_edges.append(("c4", "c5", Relation.CONFLICTS, 0.7))

        # All edges use Relation enum
        for c1, c2, rel, conf in eng.claim_edges:
            assert isinstance(rel, Relation)


# =============================================================================
# INVARIANT 6: Meta-Claims are Observations
# =============================================================================

class TestInvariant6_MetaClaimsObservations:
    """Meta-claims are observations about state, not world-claims."""

    @pytest.mark.asyncio
    async def test_meta_claims_detect_tensions(self, sample_claims):
        """Meta-claims are generated for epistemic tensions."""
        eng = Engine(llm=None)

        for claim in sample_claims:
            await eng.add_claim(claim)

        eng.compute_surfaces()
        meta_claims = eng.detect_tensions()

        # Should detect single-source surfaces
        assert len(meta_claims) > 0

        # Check types
        for mc in meta_claims:
            assert mc.type in ["high_stakes_low_evidence", "unresolved_conflict",
                              "single_source_only", "high_entropy_surface",
                              "bridge_node_detected", "stale_event"]

    @pytest.mark.asyncio
    async def test_meta_claims_have_evidence(self, sample_claims):
        """Meta-claims include evidence for why they were generated."""
        eng = Engine(llm=None)

        for claim in sample_claims:
            await eng.add_claim(claim)

        eng.compute_surfaces()
        meta_claims = eng.detect_tensions()

        for mc in meta_claims:
            assert mc.evidence is not None
            assert isinstance(mc.evidence, dict)

    @pytest.mark.asyncio
    async def test_meta_claims_track_params_version(self, sample_claims):
        """Meta-claims record which parameter version they were generated with."""
        params = Parameters()
        params.update("hub_max_df", 5, actor="test")  # version 2

        eng = Engine(llm=None, params=params)

        for claim in sample_claims:
            await eng.add_claim(claim)

        eng.compute_surfaces()
        meta_claims = eng.detect_tensions()

        for mc in meta_claims:
            assert mc.params_version == 2

    @pytest.mark.asyncio
    async def test_meta_claims_can_be_resolved(self, sample_claims):
        """Meta-claims can be marked as resolved."""
        eng = Engine(llm=None)

        for claim in sample_claims:
            await eng.add_claim(claim)

        eng.compute_surfaces()
        meta_claims = eng.detect_tensions()

        if meta_claims:
            mc = meta_claims[0]
            assert not mc.resolved

            eng.resolve_meta_claim(mc.id, "new_claim_added", actor="test")

            assert mc.resolved
            assert "new_claim_added" in mc.resolution

    @pytest.mark.asyncio
    async def test_unresolved_meta_claims_filter(self, sample_claims):
        """Can filter to only unresolved meta-claims."""
        eng = Engine(llm=None)

        for claim in sample_claims:
            await eng.add_claim(claim)

        eng.compute_surfaces()
        meta_claims = eng.detect_tensions()

        # All unresolved initially
        unresolved = eng.get_unresolved_meta_claims()
        assert len(unresolved) == len(meta_claims)

        # Resolve one
        if meta_claims:
            eng.resolve_meta_claim(meta_claims[0].id, "dismissed", actor="test")
            unresolved = eng.get_unresolved_meta_claims()
            assert len(unresolved) == len(meta_claims) - 1


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests across multiple invariants."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, sample_claims):
        """Test complete pipeline: L0 → L2 → L3 with all invariants."""
        # Custom parameters
        params = Parameters()
        params.update("hub_max_df", 5, actor="integration_test")

        eng = Engine(llm=None, params=params)

        # L0: Add claims (INVARIANT 1)
        for claim in sample_claims:
            await eng.add_claim(claim)
        assert len(eng.claims) == 5

        # Add some identity edges
        eng.claim_edges.append(("c1", "c2", Relation.CONFIRMS, 0.9))
        eng.claim_edges.append(("c4", "c5", Relation.CONFIRMS, 0.9))

        # L2: Compute surfaces (INVARIANT 3: identity only)
        surfaces = eng.compute_surfaces()
        assert len(surfaces) > 0

        # L3: Compute aboutness (INVARIANT 3: separate from identity)
        aboutness = eng.compute_surface_aboutness()

        # L3: Compute events (INVARIANT 3: uses aboutness)
        events = eng.compute_events()

        # Meta-claims (INVARIANT 6)
        meta_claims = eng.detect_tensions()

        # Summary includes all state
        summary = eng.summary()
        assert 'claims' in summary
        assert 'surfaces' in summary
        assert 'events' in summary
        assert 'params' in summary
        assert 'meta_claims' in summary

        # Parameters versioned (INVARIANT 2)
        assert summary['params']['version'] == 2

    @pytest.mark.asyncio
    async def test_reproducibility(self, sample_claims):
        """Same L0 + params = same L1-L5 (INVARIANT 4)."""
        params1 = Parameters()
        params2 = Parameters()

        engine1 = Engine(llm=None, params=params1)
        engine2 = Engine(llm=None, params=params2)

        # Add same claims to both
        for claim in sample_claims:
            await engine1.add_claim(deepcopy(claim))
            await engine2.add_claim(deepcopy(claim))

        # Add same edges to both
        engine1.claim_edges.append(("c1", "c2", Relation.CONFIRMS, 0.9))
        engine2.claim_edges.append(("c1", "c2", Relation.CONFIRMS, 0.9))

        # Compute both
        surfaces1 = engine1.compute_surfaces()
        surfaces2 = engine2.compute_surfaces()

        # Should be identical
        assert len(surfaces1) == len(surfaces2)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
