"""
Aboutness Scorer [DEPRECATED - STUB]
====================================

This module is DEPRECATED and scheduled for removal on 2026-02-01.

Replacement:
- Membership decisions: Use `reee.membrane` (classify_incident_membership)
- Story composition: Use `reee.builders.story_builder` (StoryBuilder)

See reee/deprecated/RELIC.md for full migration guide.
"""

import logging
import os
import warnings
from datetime import date

DEPRECATION_DATE = date(2026, 1, 5)
REMOVAL_DATE = date(2026, 2, 1)

_logger = logging.getLogger(__name__)
_past_removal = date.today() >= REMOVAL_DATE
_strict_mode = os.environ.get("REEE_STRICT_DEPRECATIONS", "").lower() in ("1", "true", "yes")

if _past_removal and _strict_mode:
    raise RuntimeError(
        "reee.aboutness.scorer has been removed. "
        "Use membrane for membership, StoryBuilder for composition. "
        "See reee/deprecated/RELIC.md for migration guide."
    )

_deprecation_msg = (
    f"reee.aboutness.scorer is deprecated (since {DEPRECATION_DATE}) "
    f"and {'was scheduled for removal on' if _past_removal else 'will be removed on'} {REMOVAL_DATE}. "
    "Use membrane for membership, StoryBuilder for composition. "
    "See reee/deprecated/RELIC.md for migration guide."
)

warnings.warn(_deprecation_msg, DeprecationWarning, stacklevel=2)
_logger.warning(_deprecation_msg)

# Re-export from deprecated location
from ..deprecated._aboutness.scorer import (
    ContextCompatibilityResult,
    context_compatible,
    filter_binding_anchors_by_context,
    cosine_similarity,
    PUBLISHER_ENTITIES,
    AboutnessScorer,
    compute_aboutness_edges,
    compute_events_from_aboutness,
)

__all__ = [
    'ContextCompatibilityResult',
    'context_compatible',
    'filter_binding_anchors_by_context',
    'cosine_similarity',
    'PUBLISHER_ENTITIES',
    'AboutnessScorer',
    'compute_aboutness_edges',
    'compute_events_from_aboutness',
]
