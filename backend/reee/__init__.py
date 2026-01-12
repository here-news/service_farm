"""
Recursive Epistemic Emergence Engine (REEE)
============================================

A recursive, multi-level system that ingests atomic claims and lets
higher-order structure (facts, surfaces, events, narratives) emerge
from evidence geometry and provenance.

See REEE.md for the full constitution and invariants.

ARCHITECTURE (Principled Path):
    Claims → PrincipledSurfaceBuilder → Surfaces (L2)
           → PrincipledEventBuilder  → Incidents (L3)
           → StoryBuilder            → Stories (L4)
           → CanonicalWorker         → Titles/Descriptions (presentation)

PUBLIC API (Stable):
- Claim, Surface, Event, Story: Core data types
- PrincipledSurfaceBuilder, PrincipledEventBuilder, StoryBuilder: Pure builders
- membrane.classify_*: Authoritative membership decisions
- TypedBeliefState: Jaynes inference kernel

INTERNAL (Subject to change):
- Views: Explicit scale projections

DEPRECATED (Avoid in new code - will be removed):
- Engine, EmergenceEngine: Use builders directly
- AboutnessScorer: Use PrincipledSurfaceBuilder
- ClaimExtractor, ClaimComparator: Legacy
- EpistemicKernel: Use TypedBeliefState
"""

# =============================================================================
# PUBLIC API - Stable exports for downstream consumers
# =============================================================================

# Core types (PUBLIC)
from .types import (
    # L0-L2: Always stable
    Claim,
    Surface,

    # L3/L4: Story is the public API type
    Story,
    StoryScale,

    # L5: Lens (entity navigation view)
    EntityLens,

    # Supporting types
    Parameters,
    MetaClaim,
    Relation,
    MembershipLevel,

    # Constraint ledger (principled emergence)
    Constraint,
    ConstraintType,
    ConstraintLedger,
    Motif,

    # Justification (explainability)
    EventJustification,
)

# Typed Belief (Jaynes kernel) - PUBLIC
from .typed_belief import (
    ObservationKind,
    Observation,
    NoiseModel,
    UniformNoise,
    CalibratedNoise,
    ValueDomain,
    CountDomain,
    CountDomainConfig,
    CategoricalDomain,
    TypedBeliefState,
    count_belief,
    categorical_belief,
)

# =============================================================================
# INTERNAL - For kernel computation, not for downstream API
# =============================================================================

# Internal L3 type (use Story.from_event() for API)
from .types import (
    Event,  # INTERNAL: L3 computation type
    Association,
    ParameterChange,
    AboutnessLink,
    EventSignature,
    SurfaceMembership,
)

# =============================================================================
# PRINCIPLED PATH - Primary emergence algorithms (USE THESE)
# =============================================================================

# Builders (pure algorithms for emergence) - THE PRINCIPLED PATH
from .builders import (
    # L2: Claims → Surfaces
    PrincipledSurfaceBuilder,
    MotifConfig,
    SurfaceBuilderResult,
    context_compatible,
    ContextResult,
    # L3: Surfaces → Events/Incidents
    PrincipledEventBuilder,
    EventBuilderResult,
)
# NOTE: PrincipledCaseBuilder, CaseBuilderResult, etc. are deprecated
# and now lazy-loaded via __getattr__ to avoid warning noise

# Story Builder (authoritative L4) - NEW
from .builders.story_builder import StoryBuilder, StoryBuilderResult, CompleteStory

# =============================================================================
# MEMBRANE - Authoritative membership decisions (USE THIS)
# =============================================================================

from .membrane import (
    Membership,
    CoreReason,
    LinkType,
    MembershipDecision,
    FocalSet,
    classify_incident_membership,
    is_structural_witness,
    is_semantic_only,
    WITNESS_KINDS,
    SEMANTIC_KINDS,
)

# =============================================================================
# INTERNAL - For kernel computation, not for downstream API
# =============================================================================

# Submodules (internal)
from .identity import IdentityLinker
from .meta import detect_tensions, TensionDetector, get_unresolved, count_by_type, resolve_meta_claim

# Views (explicit scale projections - internal)
from .views import (
    ViewScale, ViewTrace, ViewResult,
    IncidentViewParams, IncidentEventView, build_incident_events,
    CaseViewParams, CaseView, build_case_clusters,
)

# =============================================================================
# DEPRECATED - Lazy imports to avoid warning noise for clean users
# =============================================================================
# These are loaded on-demand via __getattr__ below.
# Only users who actually access deprecated symbols will see warnings.

_DEPRECATED_IMPORTS = {
    # From engine.py
    'Engine': ('engine', 'Engine'),
    'EmergenceEngine': ('engine', 'EmergenceEngine'),
    # From aboutness/
    'AboutnessScorer': ('aboutness', 'AboutnessScorer'),
    'compute_aboutness_edges': ('aboutness', 'compute_aboutness_edges'),
    'compute_events_from_aboutness': ('aboutness', 'compute_events_from_aboutness'),
    # From interpretation.py
    'interpret_all': ('interpretation', 'interpret_all'),
    'interpret_surface': ('interpretation', 'interpret_surface'),
    'interpret_event': ('interpretation', 'interpret_event'),
    # From extractor.py
    'ClaimExtractor': ('extractor', 'ClaimExtractor'),
    'ExtractedClaim': ('extractor', 'ExtractedClaim'),
    # From comparator.py
    'ClaimComparator': ('comparator', 'ClaimComparator'),
    'ComparisonRelation': ('comparator', 'Relation'),
    # From kernel.py
    'EpistemicKernel': ('kernel', 'EpistemicKernel'),
    'Belief': ('kernel', 'Belief'),
    # From builders.case_builder
    'PrincipledCaseBuilder': ('builders.case_builder', 'PrincipledCaseBuilder'),
    'CaseBuilderResult': ('builders.case_builder', 'CaseBuilderResult'),
    'MotifProfile': ('builders.case_builder', 'MotifProfile'),
    'L4Hubness': ('builders.case_builder', 'L4Hubness'),
    'CaseEdge': ('builders.case_builder', 'CaseEdge'),
    'EntityCase': ('builders.case_builder', 'EntityCase'),
}


def __getattr__(name: str):
    """Lazy import for deprecated symbols - only warns when actually accessed."""
    if name in _DEPRECATED_IMPORTS:
        module_name, attr_name = _DEPRECATED_IMPORTS[name]
        import importlib
        module = importlib.import_module(f'.{module_name}', __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # ==========================================================================
    # PUBLIC API - Stable, use these in downstream code
    # ==========================================================================

    # Core types
    'Claim',
    'Surface',
    'Event',           # L3 type (internal computation)
    'Story',           # L4 type (output API)
    'EntityLens',      # L5 type (entity navigation view)

    # Constraint ledger (principled emergence)
    'Constraint',
    'ConstraintType',
    'ConstraintLedger',

    # Justification (explainability)
    'EventJustification',

    # Membrane (authoritative membership) - NEW
    'Membership',
    'CoreReason',
    'LinkType',
    'MembershipDecision',
    'FocalSet',
    'classify_incident_membership',
    'is_structural_witness',
    'is_semantic_only',
    'WITNESS_KINDS',
    'SEMANTIC_KINDS',

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

    # ==========================================================================
    # PRINCIPLED PATH - Primary emergence algorithms (USE THESE)
    # ==========================================================================

    # Builders (pure algorithms for emergence)
    # L2: Claims → Surfaces
    'PrincipledSurfaceBuilder',
    'MotifConfig',
    'SurfaceBuilderResult',
    'context_compatible',
    'ContextResult',
    # L3: Surfaces → Events/Incidents
    'PrincipledEventBuilder',
    'EventBuilderResult',
    # L4: Events → Stories (authoritative)
    'StoryBuilder',
    'StoryBuilderResult',
    'CompleteStory',
    # L4: Legacy (DEPRECATED - use StoryBuilder)
    'PrincipledCaseBuilder',
    'CaseBuilderResult',
    'MotifProfile',
    'L4Hubness',
    'CaseEdge',

    # ==========================================================================
    # INTERNAL - For kernel computation, subject to change
    # ==========================================================================

    # Internal types
    'StoryScale',      # DEPRECATED: use type (Incident vs Story)
    'Parameters',
    'MetaClaim',
    'Relation',
    'MembershipLevel',
    'Motif',
    'Association',
    'ParameterChange',
    'AboutnessLink',
    'EventSignature',
    'SurfaceMembership',

    # Identity (internal)
    'IdentityLinker',

    # Views (internal - produce Story objects)
    'ViewScale',
    'ViewTrace',
    'ViewResult',
    'IncidentViewParams',
    'IncidentEventView',
    'build_incident_events',
    'CaseViewParams',
    'CaseView',
    'build_case_clusters',

    # Meta (internal)
    'detect_tensions',
    'TensionDetector',
    'get_unresolved',
    'count_by_type',
    'resolve_meta_claim',

    # ==========================================================================
    # DEPRECATED - Avoid in new code (lazy-loaded, will be removed 2026-02-01)
    # ==========================================================================

    # Engine (DEPRECATED - use builders directly)
    'Engine',
    'EmergenceEngine',

    # Aboutness (DEPRECATED - use PrincipledSurfaceBuilder + PrincipledEventBuilder)
    'AboutnessScorer',
    'compute_aboutness_edges',
    'compute_events_from_aboutness',

    # Interpretation (DEPRECATED - use TypedBeliefState)
    'interpret_all',
    'interpret_surface',
    'interpret_event',

    # Legacy (DEPRECATED)
    'ClaimExtractor',
    'ExtractedClaim',
    'ClaimComparator',
    'ComparisonRelation',
    'EpistemicKernel',
    'Belief',
]
