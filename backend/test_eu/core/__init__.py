"""
Epistemic Kernel Core Components

DEPRECATED: This module now re-exports from backend/reee/.
Please use `from reee import ...` instead.
"""

import warnings

# Re-export from reee for backward compatibility
from reee import (
    # Core types
    Claim, Surface, Event, Parameters, MetaClaim,
    Relation, Association, MembershipLevel,
    ParameterChange, AboutnessLink, EventSignature, SurfaceMembership,

    # Engine
    Engine, EmergenceEngine,

    # Identity
    IdentityLinker,

    # Aboutness
    AboutnessScorer, compute_aboutness_edges, compute_events_from_aboutness,

    # Meta
    detect_tensions, TensionDetector, get_unresolved, count_by_type, resolve_meta_claim,

    # Interpretation
    interpret_all, interpret_surface, interpret_event,

    # Legacy
    ClaimExtractor, ExtractedClaim,
    ClaimComparator, ComparisonRelation,
    EpistemicKernel, Belief,
)

__all__ = [
    'Claim', 'Surface', 'Event', 'Parameters', 'MetaClaim',
    'Relation', 'Association', 'MembershipLevel',
    'ParameterChange', 'AboutnessLink', 'EventSignature', 'SurfaceMembership',
    'Engine', 'EmergenceEngine',
    'IdentityLinker',
    'AboutnessScorer', 'compute_aboutness_edges', 'compute_events_from_aboutness',
    'detect_tensions', 'TensionDetector', 'get_unresolved', 'count_by_type', 'resolve_meta_claim',
    'interpret_all', 'interpret_surface', 'interpret_event',
    'ClaimExtractor', 'ExtractedClaim',
    'ClaimComparator', 'ComparisonRelation',
    'EpistemicKernel', 'Belief',
]
