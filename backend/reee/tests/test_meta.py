"""
Test: Meta-Claims and Tension Detection
========================================

Tests the meta-claims module for tension detection.
"""

import pytest

from reee import (
    Claim, Surface, Parameters, MetaClaim, Relation
)
from reee.meta import (
    detect_tensions,
    TensionDetector,
    get_unresolved,
    count_by_type,
    resolve_meta_claim
)


@pytest.fixture
def sample_surfaces():
    """Create sample surfaces for testing."""
    return {
        'S001': Surface(
            id='S001',
            claim_ids=['c1', 'c2'],
            sources={'bbc.com'},  # Single source
            entities={'Hong Kong', 'fire'},
            entropy=0.3
        ),
        'S002': Surface(
            id='S002',
            claim_ids=['c3'],
            sources={'cnn.com', 'reuters.com'},  # Multiple sources
            entities={'Trump'},
            entropy=0.9  # High entropy
        ),
    }


@pytest.fixture
def sample_claims():
    """Create sample claims for testing."""
    return {
        'c1': Claim(id='c1', text='Fire in HK', source='bbc.com'),
        'c2': Claim(id='c2', text='HK fire update', source='bbc.com'),
        'c3': Claim(id='c3', text='Trump news', source='cnn.com'),
    }


class TestTensionDetector:
    """TensionDetector class tests."""

    def test_detector_initialization(self, sample_claims, sample_surfaces):
        """Test detector initialization."""
        detector = TensionDetector(
            claims=sample_claims,
            surfaces=sample_surfaces,
            edges=[],
            params=Parameters()
        )
        assert detector is not None

    def test_detect_single_source(self, sample_claims, sample_surfaces):
        """Test detection of single-source surfaces."""
        detector = TensionDetector(
            claims=sample_claims,
            surfaces=sample_surfaces,
            edges=[],
            params=Parameters()
        )
        meta_claims = detector.detect_all()

        single_source = [mc for mc in meta_claims if mc.type == 'single_source_only']
        assert len(single_source) >= 1
        assert any(mc.target_id == 'S001' for mc in single_source)

    def test_detect_high_dispersion(self, sample_claims, sample_surfaces):
        """Test detection of high-dispersion surfaces (geometric, not Jaynes)."""
        params = Parameters(high_entropy_threshold=0.5)
        detector = TensionDetector(
            claims=sample_claims,
            surfaces=sample_surfaces,
            edges=[],
            params=params
        )
        meta_claims = detector.detect_all()

        # Should emit high_dispersion_surface (geometric), not high_entropy_value
        high_dispersion = [mc for mc in meta_claims if mc.type == 'high_dispersion_surface']
        assert len(high_dispersion) >= 1
        assert any(mc.target_id == 'S002' for mc in high_dispersion)

        # Verify evidence uses 'dispersion' not 'entropy'
        for mc in high_dispersion:
            assert 'dispersion' in mc.evidence
            assert 'meaning' in mc.evidence

    def test_detect_conflicts(self, sample_claims, sample_surfaces):
        """Test detection of conflicts."""
        edges = [
            ('c1', 'c2', Relation.CONFLICTS, 0.8)
        ]
        detector = TensionDetector(
            claims=sample_claims,
            surfaces=sample_surfaces,
            edges=edges,
            params=Parameters()
        )
        meta_claims = detector.detect_all()

        conflicts = [mc for mc in meta_claims if mc.type == 'unresolved_conflict']
        assert len(conflicts) >= 1


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_detect_tensions_function(self, sample_claims, sample_surfaces):
        """Test the detect_tensions convenience function."""
        meta_claims = detect_tensions(
            claims=sample_claims,
            surfaces=sample_surfaces,
            edges=[],
            params=Parameters()
        )
        assert isinstance(meta_claims, list)
        assert len(meta_claims) > 0

    def test_get_unresolved(self):
        """Test get_unresolved function."""
        meta_claims = [
            MetaClaim(type='test1', target_id='t1', target_type='surface', resolved=False),
            MetaClaim(type='test2', target_id='t2', target_type='surface', resolved=True),
            MetaClaim(type='test3', target_id='t3', target_type='surface', resolved=False),
        ]
        unresolved = get_unresolved(meta_claims)
        assert len(unresolved) == 2

    def test_count_by_type(self):
        """Test count_by_type function."""
        meta_claims = [
            MetaClaim(type='type_a', target_id='t1', target_type='surface'),
            MetaClaim(type='type_a', target_id='t2', target_type='surface'),
            MetaClaim(type='type_b', target_id='t3', target_type='surface'),
        ]
        counts = count_by_type(meta_claims)
        assert counts['type_a'] == 2
        assert counts['type_b'] == 1

    def test_resolve_meta_claim(self):
        """Test resolve_meta_claim function."""
        meta_claims = [
            MetaClaim(type='test', target_id='t1', target_type='surface'),
        ]
        mc_id = meta_claims[0].id

        resolved = resolve_meta_claim(
            meta_claims=meta_claims,
            meta_claim_id=mc_id,
            resolution='manually verified',
            actor='test_user'
        )

        assert resolved is not None
        assert resolved.resolved is True
        assert 'manually verified' in resolved.resolution
        assert 'test_user' in resolved.resolution

    def test_resolve_nonexistent_meta_claim(self):
        """Test resolving a non-existent meta-claim."""
        meta_claims = [
            MetaClaim(type='test', target_id='t1', target_type='surface'),
        ]

        resolved = resolve_meta_claim(
            meta_claims=meta_claims,
            meta_claim_id='nonexistent_id',
            resolution='test',
            actor='test_user'
        )

        assert resolved is None


class TestMetaClaimEvidence:
    """Test that meta-claims contain proper evidence."""

    def test_single_source_evidence(self, sample_claims, sample_surfaces):
        """Test evidence in single-source meta-claims."""
        meta_claims = detect_tensions(
            claims=sample_claims,
            surfaces=sample_surfaces,
            edges=[],
            params=Parameters()
        )

        single_source = [mc for mc in meta_claims if mc.type == 'single_source_only']
        for mc in single_source:
            assert 'source' in mc.evidence
            assert 'claim_count' in mc.evidence

    def test_high_dispersion_evidence(self, sample_claims, sample_surfaces):
        """Test evidence in high-dispersion meta-claims (geometric, not Jaynes)."""
        params = Parameters(high_entropy_threshold=0.5)
        meta_claims = detect_tensions(
            claims=sample_claims,
            surfaces=sample_surfaces,
            edges=[],
            params=params
        )

        high_dispersion = [mc for mc in meta_claims if mc.type == 'high_dispersion_surface']
        for mc in high_dispersion:
            assert 'dispersion' in mc.evidence  # Renamed from 'entropy'
            assert 'threshold' in mc.evidence
            assert 'meaning' in mc.evidence  # Explains what this measures

    def test_conflict_evidence(self, sample_claims, sample_surfaces):
        """Test evidence in conflict meta-claims."""
        edges = [('c1', 'c2', Relation.CONFLICTS, 0.8)]
        meta_claims = detect_tensions(
            claims=sample_claims,
            surfaces=sample_surfaces,
            edges=edges,
            params=Parameters()
        )

        conflicts = [mc for mc in meta_claims if mc.type == 'unresolved_conflict']
        for mc in conflicts:
            assert 'claim_1' in mc.evidence
            assert 'claim_2' in mc.evidence
            assert 'confidence' in mc.evidence


class TestSemanticInvariants:
    """
    Invariant tests for semantic correctness.

    These tests ensure the naming/semantics don't drift:
    - high_entropy_value = Jaynes H(X|E,S), REQUIRES typed coverage > 0
    - high_dispersion_surface = geometric D(surface), does NOT require typed coverage
    """

    def test_high_entropy_value_never_emitted_without_typed_coverage(self):
        """
        INVARIANT: high_entropy_value must never be emitted when typed_coverage_zero.

        This prevents semantic drift where geometric dispersion gets confused
        with Jaynes entropy over typed variables.
        """
        # Create surfaces with high entropy (geometric dispersion)
        surfaces = {
            'S001': Surface(
                id='S001',
                claim_ids=['c1', 'c2'],
                sources={'bbc.com'},
                entities={'test'},
                entropy=0.9  # High dispersion
            ),
        }

        # NO claims with typed constraints (question_key + extracted_value)
        claims_without_typing = {
            'c1': Claim(id='c1', text='Some text', source='bbc.com'),
            'c2': Claim(id='c2', text='More text', source='bbc.com'),
        }

        detector = TensionDetector(
            claims=claims_without_typing,
            surfaces=surfaces,
            edges=[],
            params=Parameters(high_entropy_threshold=0.5)
        )

        meta_claims = detector.detect_all()

        # INVARIANT: high_entropy_value must NOT appear
        high_entropy = [mc for mc in meta_claims if mc.type == 'high_entropy_value']
        assert len(high_entropy) == 0, (
            "INVARIANT VIOLATION: high_entropy_value emitted when typed coverage is zero. "
            "high_entropy_value is for Jaynes H(X|E,S), not geometric dispersion."
        )

        # high_dispersion_surface SHOULD appear (geometric is independent of typing)
        high_dispersion = [mc for mc in meta_claims if mc.type == 'high_dispersion_surface']
        assert len(high_dispersion) >= 1, (
            "high_dispersion_surface should be emitted for geometric dispersion"
        )

    def test_typed_dependent_metaclaims_gated_by_coverage(self):
        """
        Typed-dependent meta-claims (high_entropy_value, typed_value_conflict,
        coverage_gap) must only be emitted when typed coverage > 0.
        """
        surfaces = {
            'S001': Surface(id='S001', claim_ids=['c1'], sources={'bbc.com'}),
        }

        # Claims WITHOUT typed constraints
        untyped_claims = {
            'c1': Claim(id='c1', text='Text', source='bbc.com'),
        }

        detector = TensionDetector(
            claims=untyped_claims,
            surfaces=surfaces,
            edges=[],
            params=Parameters()
        )

        meta_claims = detector.detect_all()

        # None of these should appear without typed coverage
        typed_dependent_types = {'high_entropy_value', 'typed_value_conflict', 'coverage_gap'}
        for mc in meta_claims:
            assert mc.type not in typed_dependent_types, (
                f"INVARIANT VIOLATION: {mc.type} emitted without typed coverage. "
                f"These meta-claims require claims with question_key + extracted_value."
            )

    def test_dispersion_vs_entropy_naming(self):
        """
        Verify naming convention:
        - high_dispersion_surface uses 'dispersion' in evidence (geometric)
        - high_entropy_value uses 'entropy' in evidence (Jaynes)
        """
        surfaces = {
            'S001': Surface(id='S001', claim_ids=['c1'], sources={'bbc.com'}, entropy=0.9),
        }

        detector = TensionDetector(
            claims={},
            surfaces=surfaces,
            edges=[],
            params=Parameters(high_entropy_threshold=0.5)
        )

        meta_claims = detector.detect_all()
        high_dispersion = [mc for mc in meta_claims if mc.type == 'high_dispersion_surface']

        for mc in high_dispersion:
            # Must use 'dispersion' not 'entropy' to prevent confusion
            assert 'dispersion' in mc.evidence, "high_dispersion_surface should use 'dispersion' key"
            assert 'entropy' not in mc.evidence, "high_dispersion_surface should NOT use 'entropy' key"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
