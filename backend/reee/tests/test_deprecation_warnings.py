"""
Test Deprecation Warnings
=========================

Verifies that deprecated modules emit proper warnings and will fail
after their removal date.

This test serves as a safety belt - any new code importing deprecated
modules will trigger these warnings, making accidental usage visible.
"""

import pytest
import warnings
from datetime import date


class TestCaseBuilderDeprecation:
    """Tests for case_builder.py deprecation.

    Note: Since Python caches imports, we test the stub's configuration
    directly rather than trying to catch warnings on re-import.
    """

    def test_stub_has_deprecation_date(self):
        """Stub should define deprecation date."""
        from reee.builders import case_builder

        assert hasattr(case_builder, 'DEPRECATION_DATE'), \
            "case_builder stub should define DEPRECATION_DATE"
        assert case_builder.DEPRECATION_DATE.year == 2026, \
            "DEPRECATION_DATE should be in 2026"

    def test_stub_has_removal_date(self):
        """Stub should define removal date."""
        from reee.builders import case_builder

        assert hasattr(case_builder, 'REMOVAL_DATE'), \
            "case_builder stub should define REMOVAL_DATE"
        assert case_builder.REMOVAL_DATE == date(2026, 2, 1), \
            "REMOVAL_DATE should be 2026-02-01"

    def test_stub_docstring_mentions_replacement(self):
        """Stub docstring should mention replacement."""
        from reee.builders import case_builder

        docstring = case_builder.__doc__ or ""
        assert "StoryBuilder" in docstring, \
            "Stub docstring should mention StoryBuilder"
        assert "EntityLens" in docstring, \
            "Stub docstring should mention EntityLens"

    def test_symbols_still_accessible(self):
        """Deprecated symbols should still be accessible during migration."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress for this test

            from reee.builders.case_builder import (
                EntityCase,
                PrincipledCaseBuilder,
                CaseBuilderResult,
                MotifProfile,
                L4Hubness,
                CaseEdge,
            )

            # Verify symbols are actual classes
            assert EntityCase is not None
            assert PrincipledCaseBuilder is not None


class TestToEntityCaseDeprecation:
    """Tests for CompleteStory.to_entity_case() deprecation."""

    def test_to_entity_case_emits_warning(self):
        """to_entity_case() should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from reee.builders.story_builder import CompleteStory, TemporalMode
            from datetime import datetime

            # Create minimal story
            mode = TemporalMode(
                mode_id='test',
                spine='Test',
                time_start=datetime.now(),
                time_end=datetime.now(),
                incident_ids=set(),
                explanation='Test'
            )
            story = CompleteStory(
                story_id='test',
                spine='Test',
                mode=mode,
                core_incident_ids=set(),
                facets={},
                title='Test',
                description='Test',
                time_start=None,
                time_end=None,
                surface_count=0,
                source_count=0,
                claim_count=0,
                expected_facets=set(),
                present_facets=set(),
                coverage_facets=set(),
                missing_facets=set(),
                noise_facets=set(),
                blocked_facets={},
                completeness_score=0.0,
                conflict_inquiries=[],
                gap_inquiries=[],
                quality_inquiries=[],
                membership_weights={},
                explanation='Test',
            )

            # Call deprecated method
            entity_case = story.to_entity_case()

            # Should emit deprecation warning
            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
                and "to_entity_case" in str(warning.message)
            ]

            assert len(deprecation_warnings) >= 1, \
                "to_entity_case() should emit DeprecationWarning"


class TestEntityLensIsNotDeprecated:
    """Verify EntityLens (the replacement) doesn't emit warnings."""

    def test_entity_lens_no_warning(self):
        """EntityLens import should not emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from reee.types import EntityLens

            # Create a lens
            lens = EntityLens.create(
                entity="Test Entity",
                incident_ids={"inc1", "inc2"},
                companion_counts={"Companion": 5},
            )

            # Should NOT have deprecation warnings about EntityLens
            lens_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
                and "EntityLens" in str(warning.message)
            ]

            assert len(lens_warnings) == 0, \
                f"EntityLens should not emit deprecation warnings. Got: {lens_warnings}"


# =============================================================================
# DIRECT RUN SUPPORT
# =============================================================================

if __name__ == "__main__":
    print("Testing deprecation warnings...")
    print("-" * 50)

    # Test case_builder warning
    print("1. Testing case_builder import warning:")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from reee.builders.case_builder import EntityCase
        for warning in w:
            if issubclass(warning.category, DeprecationWarning):
                print(f"   ✓ {warning.message}")

    # Test EntityLens (no warning)
    print("\n2. Testing EntityLens (should be clean):")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from reee.types import EntityLens
        lens = EntityLens.create("Test", {"inc1"}, None)
        lens_warnings = [x for x in w if "EntityLens" in str(x.message)]
        if lens_warnings:
            print(f"   ✗ Unexpected warnings: {lens_warnings}")
        else:
            print(f"   ✓ No deprecation warnings")

    print("-" * 50)
    print("Run with pytest for full validation")
