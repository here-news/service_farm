"""
Event Views Module
==================

Provides explicit view builders for different event scales.

Per REEE1.md Section 8: "views, not layers" - the same surfaces can be
projected into different event structures depending on the scale and
purpose of the query.

Two primary views:
- IncidentEventView: cohesion operator ("same happening")
- CaseView: narrative operator ("same story")

Each view has:
- Distinct parameters
- Distinct builder
- Distinct evaluation target
- Traceable provenance (which view produced this result)

Hubness:
- Local hubness (dispersion-based) is used in CaseView
- High frequency + low dispersion = backbone (binds)
- High frequency + high dispersion = hub (suppressed)
"""

from .incident import (
    IncidentViewParams,
    IncidentEventView,
    build_incident_events,
)

from .case import (
    CaseViewParams,
    CaseView,
    build_case_clusters,
)

from .types import (
    ViewScale,
    ViewTrace,
    ViewResult,
)

from .hubness import (
    AnchorHubness,
    HubnessResult,
    analyze_hubness,
    compute_co_anchor_dispersion,
    print_hubness_report,
)

from .relations import (
    RelationEdge,
    RelationBackbone,
    build_relation_backbone_from_incidents,
    print_backbone_report,
)

__all__ = [
    # Incident view
    'IncidentViewParams',
    'IncidentEventView',
    'build_incident_events',

    # Case view
    'CaseViewParams',
    'CaseView',
    'build_case_clusters',

    # Hubness
    'AnchorHubness',
    'HubnessResult',
    'analyze_hubness',
    'compute_co_anchor_dispersion',
    'print_hubness_report',

    # Relations
    'RelationEdge',
    'RelationBackbone',
    'build_relation_backbone_from_incidents',
    'print_backbone_report',

    # Common types
    'ViewScale',
    'ViewTrace',
    'ViewResult',
]
