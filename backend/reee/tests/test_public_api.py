"""
Public API Tests
================

Tests that verify the REEE public API exports are correct and that
deprecated paths are marked appropriately.

This locks down the public interface to prevent accidental breakage.
"""

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# TEST: PUBLIC API EXPORTS
# ============================================================================

class TestPublicAPIExports:
    """Test that public API exports are available and correct."""

    def test_core_types_exported(self):
        """Core data types should be importable from reee."""
        from reee import Claim, Surface, Story, StoryScale

        # Verify they are the correct types
        assert hasattr(Claim, '__dataclass_fields__')
        assert hasattr(Surface, '__dataclass_fields__')

    def test_supporting_types_exported(self):
        """Supporting types should be importable from reee."""
        from reee import (
            Parameters, MetaClaim, Relation, MembershipLevel,
            Constraint, ConstraintType, ConstraintLedger, Motif,
            EventJustification,
        )

        # Verify ConstraintType is an enum
        assert hasattr(ConstraintType, 'STRUCTURAL')
        assert hasattr(ConstraintType, 'SEMANTIC')

    def test_typed_belief_exported(self):
        """Typed belief (Jaynes kernel) should be importable from reee."""
        from reee import (
            ObservationKind, Observation, NoiseModel,
            UniformNoise, CalibratedNoise,
            ValueDomain, CountDomain, CountDomainConfig, CategoricalDomain,
            TypedBeliefState, count_belief, categorical_belief,
        )

        # Verify these are callable/usable
        assert callable(count_belief)
        assert callable(categorical_belief)


# ============================================================================
# TEST: PRINCIPLED PATH EXPORTS
# ============================================================================

class TestPrincipledPathExports:
    """Test that principled path builders are the primary exports."""

    def test_builders_exported(self):
        """Principled builders should be importable from reee."""
        from reee import (
            PrincipledSurfaceBuilder,
            PrincipledEventBuilder,
            MotifConfig,
            SurfaceBuilderResult,
            EventBuilderResult,
            context_compatible,
            ContextResult,
        )

        # Verify these are classes
        assert callable(PrincipledSurfaceBuilder)
        assert callable(PrincipledEventBuilder)
        assert callable(MotifConfig)

    def test_builders_importable_from_submodule(self):
        """Builders should also be importable from reee.builders."""
        from reee.builders import (
            PrincipledSurfaceBuilder,
            PrincipledEventBuilder,
            MotifConfig,
        )

        # Same classes
        from reee import PrincipledSurfaceBuilder as RootBuilder
        assert PrincipledSurfaceBuilder is RootBuilder


# ============================================================================
# TEST: INTERNAL EXPORTS
# ============================================================================

class TestInternalExports:
    """Test that internal types are still accessible but marked as internal."""

    def test_event_type_exported(self):
        """Event (internal L3 type) should be accessible."""
        from reee import Event

        assert hasattr(Event, '__dataclass_fields__')

    def test_views_exported(self):
        """Views should be accessible (internal)."""
        from reee import (
            ViewScale, ViewTrace, ViewResult,
            IncidentViewParams, IncidentEventView, build_incident_events,
            CaseViewParams, CaseView, build_case_clusters,
        )

        assert callable(build_incident_events)
        assert callable(build_case_clusters)

    def test_meta_exported(self):
        """Meta detection should be accessible (internal)."""
        from reee import (
            detect_tensions, TensionDetector,
            get_unresolved, count_by_type, resolve_meta_claim,
        )

        assert callable(detect_tensions)


# ============================================================================
# TEST: DEPRECATED EXPORTS
# ============================================================================

class TestDeprecatedExports:
    """Test that deprecated exports are still accessible but discouraged."""

    def test_aboutness_still_accessible(self):
        """Aboutness (deprecated) should still be importable."""
        from reee import (
            AboutnessScorer,
            compute_aboutness_edges,
            compute_events_from_aboutness,
        )

        # These should work but are deprecated
        assert callable(compute_aboutness_edges)

    def test_legacy_exports_still_accessible(self):
        """Legacy exports should still be importable."""
        from reee import (
            ClaimExtractor, ExtractedClaim,
            ClaimComparator, ComparisonRelation,
            EpistemicKernel, Belief,
        )

        # These should work but are deprecated
        assert callable(ClaimExtractor)


# ============================================================================
# TEST: __all__ CORRECTNESS
# ============================================================================

class TestAllExports:
    """Test that __all__ is correct and complete."""

    def test_all_exports_are_importable(self):
        """Every name in __all__ should be importable."""
        import reee

        for name in reee.__all__:
            assert hasattr(reee, name), f"'{name}' in __all__ but not importable"

    def test_public_api_in_all(self):
        """Public API types should be in __all__."""
        import reee

        public_api = [
            'Claim', 'Surface', 'Story', 'StoryScale',
            'Parameters', 'MetaClaim', 'Relation', 'MembershipLevel',
            'Constraint', 'ConstraintType', 'ConstraintLedger', 'Motif',
            'TypedBeliefState',
        ]

        for name in public_api:
            assert name in reee.__all__, f"Public API '{name}' not in __all__"

    def test_principled_builders_in_all(self):
        """Principled builders should be in __all__."""
        import reee

        builders = [
            'PrincipledSurfaceBuilder',
            'PrincipledEventBuilder',
            'MotifConfig',
            'SurfaceBuilderResult',
            'EventBuilderResult',
        ]

        for name in builders:
            assert name in reee.__all__, f"Builder '{name}' not in __all__"


# ============================================================================
# TEST: API ORGANIZATION
# ============================================================================

class TestAPIOrganization:
    """Test that the API is organized correctly per docstring."""

    def test_docstring_documents_architecture(self):
        """Module docstring should document the principled architecture."""
        import reee

        docstring = reee.__doc__
        assert 'ARCHITECTURE' in docstring or 'Principled' in docstring
        assert 'PrincipledSurfaceBuilder' in docstring or 'Claims' in docstring

    def test_docstring_documents_deprecated(self):
        """Module docstring should document deprecated paths."""
        import reee

        docstring = reee.__doc__
        assert 'DEPRECATED' in docstring or 'deprecated' in docstring
        assert 'AboutnessScorer' in docstring or 'aboutness' in docstring.lower()


# ============================================================================
# TEST: IMPORT PATTERNS
# ============================================================================

class TestImportPatterns:
    """Test recommended import patterns work correctly."""

    def test_recommended_public_import(self):
        """Recommended public API import pattern should work."""
        # This is the recommended pattern for downstream consumers
        from reee import (
            Claim, Surface, Story,
            PrincipledSurfaceBuilder, PrincipledEventBuilder,
            MotifConfig,
            Constraint, ConstraintType, ConstraintLedger,
        )

        # All should be usable
        assert Claim is not None
        assert Surface is not None
        assert Story is not None
        assert PrincipledSurfaceBuilder is not None

    def test_internal_import_pattern(self):
        """Internal import pattern for kernel development should work."""
        # This pattern is for internal kernel development
        from reee import (
            Event,  # Internal L3 type
            IdentityLinker,
            detect_tensions,
        )

        assert Event is not None


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
