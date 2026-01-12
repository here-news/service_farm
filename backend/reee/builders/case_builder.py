"""
Principled Case Builder [DEPRECATED - STUB]
============================================

This module is DEPRECATED and scheduled for removal on 2026-02-01.

Use StoryBuilder + EntityLens instead:

    from reee.builders.story_builder import StoryBuilder
    from reee.types import EntityLens

    builder = StoryBuilder()
    result = builder.build_from_incidents(incidents)
    for story in result.stories.values():
        lens = story.to_lens()
        # ... process lens

See reee/deprecated/RELIC.md for full migration guide.
"""

import logging
import os
import warnings
from datetime import date

# Deprecation configuration
DEPRECATION_DATE = date(2026, 1, 5)
REMOVAL_DATE = date(2026, 2, 1)

_logger = logging.getLogger(__name__)

# Check if past removal date
_past_removal = date.today() >= REMOVAL_DATE
_strict_mode = os.environ.get("REEE_STRICT_DEPRECATIONS", "").lower() in ("1", "true", "yes")

if _past_removal and _strict_mode:
    # Only raise in strict mode (for CI/testing)
    raise RuntimeError(
        "reee.builders.case_builder has been removed. "
        "Use StoryBuilder + EntityLens instead. "
        "See reee/deprecated/RELIC.md for migration guide."
    )

# Build deprecation message
_deprecation_msg = (
    f"reee.builders.case_builder is deprecated (since {DEPRECATION_DATE}) "
    f"and {'was scheduled for removal on' if _past_removal else 'will be removed on'} {REMOVAL_DATE}. "
    "Use StoryBuilder + EntityLens instead. "
    "See reee/deprecated/RELIC.md for migration guide."
)

# Emit deprecation warning (visible with -W default::DeprecationWarning)
warnings.warn(_deprecation_msg, DeprecationWarning, stacklevel=2)

# Also log for production visibility
_logger.warning(_deprecation_msg)

# Re-export all symbols from deprecated location for backward compatibility
# This allows existing code to continue working during migration period
from ..deprecated._case_builder import (
    MotifProfile,
    L4Hubness,
    CaseEdge,
    EntityCase,
    CaseBuilderResult,
    PrincipledCaseBuilder,
    MembershipLevel,
)

__all__ = [
    'MotifProfile',
    'L4Hubness',
    'CaseEdge',
    'EntityCase',
    'CaseBuilderResult',
    'PrincipledCaseBuilder',
    'MembershipLevel',
]
