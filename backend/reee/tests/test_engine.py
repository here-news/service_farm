"""
Test: Engine Functionality
==========================

Tests the core Engine (EmergenceEngine) functionality.
"""

import pytest
import asyncio

from reee import Engine, Claim, Parameters


@pytest.fixture
def engine():
    """Create a fresh engine with default parameters."""
    return Engine(params=Parameters())


@pytest.fixture
def sample_claims():
    """Sample claims for testing."""
    return [
        Claim(
            id='c1',
            text='Fire broke out in Hong Kong apartment building',
            source='bbc.com',
            entities={'Hong Kong', 'fire', 'apartment'},
            anchor_entities={'Hong Kong'}
        ),
        Claim(
            id='c2',
            text='Hong Kong fire causes mass evacuations',
            source='reuters.com',
            entities={'Hong Kong', 'fire', 'evacuations'},
            anchor_entities={'Hong Kong'}
        ),
        Claim(
            id='c3',
            text='Trump meets with foreign leaders at summit',
            source='cnn.com',
            entities={'Trump', 'foreign leaders', 'summit'},
            anchor_entities={'Trump'}
        ),
        Claim(
            id='c4',
            text='Trump discusses trade policy with allies',
            source='fox.com',
            entities={'Trump', 'trade policy', 'allies'},
            anchor_entities={'Trump'}
        ),
    ]


class TestEngineBasics:
    """Basic engine tests."""

    def test_engine_initialization(self, engine):
        """Test engine initializes correctly."""
        assert engine.params is not None
        assert engine.params.version == 1
        assert len(engine.claims) == 0
        assert len(engine.surfaces) == 0
        assert len(engine.events) == 0

    @pytest.mark.asyncio
    async def test_add_single_claim(self, engine):
        """Test adding a single claim."""
        claim = Claim(
            id='test1',
            text='Test claim text',
            source='test.com',
            entities={'test'},
            anchor_entities={'test'}
        )
        result = await engine.add_claim(claim)
        assert 'claim_id' in result
        assert result['claim_id'] == 'test1'
        assert len(engine.claims) == 1

    @pytest.mark.asyncio
    async def test_add_multiple_claims(self, engine, sample_claims):
        """Test adding multiple claims."""
        for claim in sample_claims:
            await engine.add_claim(claim)
        assert len(engine.claims) == 4


class TestSurfaceComputation:
    """Surface computation tests."""

    @pytest.mark.asyncio
    async def test_compute_surfaces(self, engine, sample_claims):
        """Test surface computation creates surfaces."""
        for claim in sample_claims:
            await engine.add_claim(claim)

        surfaces = engine.compute_surfaces()
        assert len(surfaces) > 0
        # Each claim should be in exactly one surface
        total_claims_in_surfaces = sum(len(s.claim_ids) for s in surfaces)
        assert total_claims_in_surfaces == len(sample_claims)

    @pytest.mark.asyncio
    async def test_surfaces_have_sources(self, engine, sample_claims):
        """Test surfaces track source information."""
        for claim in sample_claims:
            await engine.add_claim(claim)

        surfaces = engine.compute_surfaces()
        for surface in surfaces:
            assert len(surface.sources) > 0


class TestEventComputation:
    """Event computation tests."""

    @pytest.mark.asyncio
    async def test_compute_events(self, engine, sample_claims):
        """Test event computation."""
        for claim in sample_claims:
            await engine.add_claim(claim)

        engine.compute_surfaces()
        engine.compute_surface_aboutness()
        events = engine.compute_events()

        # Should have some events
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_events_group_related_surfaces(self, engine, sample_claims):
        """Test that related surfaces are grouped into events."""
        for claim in sample_claims:
            await engine.add_claim(claim)

        engine.compute_surfaces()
        engine.compute_surface_aboutness()
        events = engine.compute_events()

        # Each surface should be in at least one event
        all_surface_ids = set(engine.surfaces.keys())
        surfaces_in_events = set()
        for event in events:
            surfaces_in_events.update(event.surface_ids)

        assert surfaces_in_events == all_surface_ids


class TestMetaClaims:
    """Meta-claims and tension detection tests."""

    @pytest.mark.asyncio
    async def test_detect_tensions(self, engine, sample_claims):
        """Test tension detection."""
        for claim in sample_claims:
            await engine.add_claim(claim)

        engine.compute_surfaces()
        meta_claims = engine.detect_tensions()

        # Should detect some tensions (at least single-source)
        assert len(meta_claims) > 0

    @pytest.mark.asyncio
    async def test_meta_claim_types(self, engine, sample_claims):
        """Test meta-claim type variety."""
        for claim in sample_claims:
            await engine.add_claim(claim)

        engine.compute_surfaces()
        meta_claims = engine.detect_tensions()

        types = {mc.type for mc in meta_claims}
        assert 'single_source_only' in types  # Each surface has only one source

    @pytest.mark.asyncio
    async def test_resolve_meta_claim(self, engine, sample_claims):
        """Test meta-claim resolution."""
        for claim in sample_claims:
            await engine.add_claim(claim)

        engine.compute_surfaces()
        meta_claims = engine.detect_tensions()

        if meta_claims:
            mc = meta_claims[0]
            resolved = engine.resolve_meta_claim(mc.id, 'manually verified', 'test_user')
            assert resolved is not None
            assert resolved.resolved is True
            assert len(engine.get_unresolved_meta_claims()) == len(meta_claims) - 1


class TestEngineSummary:
    """Engine summary tests."""

    @pytest.mark.asyncio
    async def test_summary(self, engine, sample_claims):
        """Test summary generation."""
        for claim in sample_claims:
            await engine.add_claim(claim)

        engine.compute_surfaces()
        engine.compute_surface_aboutness()
        engine.compute_events()
        engine.detect_tensions()

        summary = engine.summary()

        assert 'claims' in summary
        assert 'surfaces' in summary
        assert 'events' in summary
        assert 'params' in summary
        assert 'meta_claims' in summary

        assert summary['claims'] == 4
        assert summary['surfaces'] > 0
        assert summary['events'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
