"""
Test: Module Imports
====================

Verifies all REEE modules can be imported correctly.
"""

import pytest


def test_core_types_import():
    """Test importing core types."""
    from reee import (
        Claim, Surface, Event, Parameters, MetaClaim,
        Relation, Association, MembershipLevel,
        ParameterChange, AboutnessLink, EventSignature, SurfaceMembership
    )
    assert Claim is not None
    assert Surface is not None
    assert Event is not None
    assert Parameters is not None


def test_engine_import():
    """Test importing Engine."""
    from reee import Engine, EmergenceEngine
    assert Engine is not None
    assert EmergenceEngine is Engine  # Backward compatibility alias


def test_identity_import():
    """Test importing identity submodule."""
    from reee.identity import IdentityLinker
    from reee import IdentityLinker as IdentityLinkerTop
    assert IdentityLinker is IdentityLinkerTop


def test_aboutness_import():
    """Test importing aboutness submodule."""
    from reee.aboutness import (
        AboutnessScorer,
        compute_aboutness_edges,
        compute_events_from_aboutness
    )
    from reee import AboutnessScorer as AboutnessScorerTop
    assert AboutnessScorer is AboutnessScorerTop


def test_meta_import():
    """Test importing meta submodule."""
    from reee.meta import (
        detect_tensions,
        TensionDetector,
        get_unresolved,
        count_by_type,
        resolve_meta_claim
    )
    from reee import TensionDetector as TensionDetectorTop
    assert TensionDetector is TensionDetectorTop


def test_interpretation_import():
    """Test importing interpretation functions."""
    from reee import interpret_all, interpret_surface, interpret_event
    assert interpret_all is not None


def test_backward_compatibility():
    """Test backward compatibility via test_eu.core."""
    from test_eu.core import Engine as OldEngine, Claim as OldClaim
    from reee import Engine, Claim
    assert OldEngine is Engine
    assert OldClaim is Claim


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
