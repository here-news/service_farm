"""
Recursive Epistemic Emergence Engine (REEE)
============================================

A recursive, multi-level system that ingests atomic claims and lets
higher-order structure (facts, surfaces, events, narratives) emerge
from evidence geometry and provenance.

See REEE.md for the full constitution and invariants.
"""

# Core types
from .types import (
    Claim, Surface, Event, Parameters, MetaClaim,
    Relation, Association, MembershipLevel,
    ParameterChange, AboutnessLink, EventSignature, SurfaceMembership
)

# Main engine
from .engine import Engine, EmergenceEngine

# Submodules
from .identity import IdentityLinker
from .aboutness import AboutnessScorer, compute_aboutness_edges, compute_events_from_aboutness
from .meta import detect_tensions, TensionDetector, get_unresolved, count_by_type, resolve_meta_claim
from .interpretation import interpret_all, interpret_surface, interpret_event

# Views (explicit scale projections)
from .views import (
    ViewScale, ViewTrace, ViewResult,
    IncidentViewParams, IncidentEventView, build_incident_events,
    CaseViewParams, CaseView, build_case_clusters,
)

# Legacy components (may be deprecated)
from .extractor import ClaimExtractor, ExtractedClaim
from .comparator import ClaimComparator, Relation as ComparisonRelation
from .kernel import EpistemicKernel, Belief

# Typed Belief (Jaynes kernel)
from .typed_belief import (
    ObservationKind, Observation, NoiseModel, UniformNoise, CalibratedNoise,
    ValueDomain, CountDomain, CountDomainConfig, CategoricalDomain,
    TypedBeliefState, count_belief, categorical_belief
)

__all__ = [
    # Core types
    'Claim',
    'Surface',
    'Event',
    'Parameters',
    'MetaClaim',
    'Relation',
    'Association',
    'MembershipLevel',
    'ParameterChange',
    'AboutnessLink',
    'EventSignature',
    'SurfaceMembership',

    # Engine
    'Engine',
    'EmergenceEngine',

    # Identity
    'IdentityLinker',

    # Aboutness
    'AboutnessScorer',
    'compute_aboutness_edges',
    'compute_events_from_aboutness',

    # Views (explicit scale projections)
    'ViewScale',
    'ViewTrace',
    'ViewResult',
    'IncidentViewParams',
    'IncidentEventView',
    'build_incident_events',
    'CaseViewParams',
    'CaseView',
    'build_case_clusters',

    # Meta
    'detect_tensions',
    'TensionDetector',
    'get_unresolved',
    'count_by_type',
    'resolve_meta_claim',

    # Interpretation
    'interpret_all',
    'interpret_surface',
    'interpret_event',

    # Legacy
    'ClaimExtractor',
    'ExtractedClaim',
    'ClaimComparator',
    'ComparisonRelation',
    'EpistemicKernel',
    'Belief',

    # Typed Belief (Jaynes kernel)
    'ObservationKind',
    'Observation',
    'NoiseModel',
    'UniformNoise',
    'CalibratedNoise',
    'ValueDomain',
    'CountDomain',
    'CountDomainConfig',
    'CategoricalDomain',
    'TypedBeliefState',
    'count_belief',
    'categorical_belief',
]
