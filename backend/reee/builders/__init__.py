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

# Case builder (L4) - DEPRECATED: kept for EntityCase type only
from .case_builder import (
    PrincipledCaseBuilder,  # DEPRECATED
    CaseBuilderResult,  # DEPRECATED
    MotifProfile,  # DEPRECATED
    L4Hubness,  # DEPRECATED
    CaseEdge,  # DEPRECATED
    EntityCase,  # Still used for API compatibility
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
