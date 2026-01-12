"""
REEE Builders - Pure Algorithmic Components
============================================

These are pure functions/classes for emergence computation.
No I/O, no database dependencies.

Modules:
- surface_builder: Claims → Surfaces via motif clustering (L2)
- event_builder: Surfaces → Events via membrane formation (L3)
- story_builder: Incidents → Stories via spine + membrane (L4) [NEW]
- case_builder: [DEPRECATED] - kept for EntityCase type compatibility
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

# Story builder (L4) - NEW: replaces case_builder
from .story_builder import (
    StoryBuilder,
    StoryBuilderResult,
    CompleteStory,
    StorySpine,
    TemporalMode,
    StoryFacet,
)

# Case builder (L4) - DEPRECATED: lazy-loaded to avoid warning noise
# Only users who actually access these symbols will see deprecation warnings.
_DEPRECATED_CASE_BUILDER = {
    'PrincipledCaseBuilder',
    'CaseBuilderResult',
    'MotifProfile',
    'L4Hubness',
    'CaseEdge',
    'EntityCase',
}


def __getattr__(name: str):
    """Lazy import for deprecated case_builder symbols."""
    if name in _DEPRECATED_CASE_BUILDER:
        from . import case_builder
        return getattr(case_builder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    # Story builder (L4) - NEW
    'StoryBuilder',
    'StoryBuilderResult',
    'CompleteStory',
    'StorySpine',
    'TemporalMode',
    'StoryFacet',
    # Case builder (L4) - DEPRECATED
    'PrincipledCaseBuilder',  # DEPRECATED
    'CaseBuilderResult',  # DEPRECATED
    'EntityCase',  # Still used
    'MotifProfile',  # DEPRECATED
    'L4Hubness',  # DEPRECATED
    'CaseEdge',  # DEPRECATED
]
