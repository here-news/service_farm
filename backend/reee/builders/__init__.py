"""
REEE Builders - Pure Algorithmic Components
============================================

These are pure functions/classes for emergence computation.
No I/O, no database dependencies.

Modules:
- surface_builder: Claims → Surfaces via motif clustering (L2)
- event_builder: Surfaces → Events via membrane formation (L3)
- case_builder: Events → Cases via motif recurrence (L4)
"""

from .surface_builder import (
    PrincipledSurfaceBuilder,
    MotifConfig,
    SurfaceBuilderResult,
    context_compatible,
    ContextResult,
)

from .event_builder import (
    PrincipledEventBuilder,
    EventBuilderResult,
)

from .case_builder import (
    PrincipledCaseBuilder,
    CaseBuilderResult,
    MotifProfile,
    L4Hubness,
    CaseEdge,
)

__all__ = [
    # Surface builder (L2)
    'PrincipledSurfaceBuilder',
    'MotifConfig',
    'SurfaceBuilderResult',
    'context_compatible',
    'ContextResult',
    # Event builder (L3)
    'PrincipledEventBuilder',
    'EventBuilderResult',
    # Case builder (L4)
    'PrincipledCaseBuilder',
    'CaseBuilderResult',
    'MotifProfile',
    'L4Hubness',
    'CaseEdge',
]
