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


def test_views_import():
    """Test importing views submodule."""
    from reee.views import (
        # Incident view
        IncidentViewParams,
        IncidentEventView,
        build_incident_events,
        # Case view
        CaseViewParams,
        CaseView,
        build_case_clusters,
        # Hubness
        AnchorHubness,
        HubnessResult,
        analyze_hubness,
        compute_co_anchor_dispersion,
        # Relations
        RelationEdge,
        RelationBackbone,
        build_relation_backbone_from_incidents,
        # Types
        ViewScale,
        ViewTrace,
        ViewResult,
    )
    assert CaseView is not None
    assert IncidentEventView is not None
    assert analyze_hubness is not None


@pytest.mark.skip(reason="test_eu module has been removed - backward compatibility no longer needed")
def test_backward_compatibility():
    """Test backward compatibility via test_eu.core."""
    pass  # test_eu was deleted, keeping test as placeholder


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
