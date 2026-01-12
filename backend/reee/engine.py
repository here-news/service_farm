"""
Engine Module [DEPRECATED - STUB]
=================================

This module is DEPRECATED and scheduled for removal on 2026-02-01.

Replacement:
- L2 emergence: Use `reee.builders.PrincipledSurfaceBuilder`
- L3 emergence: Use `reee.builders.PrincipledEventBuilder`
- L4 emergence: Use `reee.builders.StoryBuilder`
- Full pipeline: Compose builders directly in worker layer

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
        "reee.engine has been removed. "
        "Use PrincipledSurfaceBuilder, PrincipledEventBuilder, StoryBuilder directly. "
        "See reee/deprecated/RELIC.md for migration guide."
    )

_deprecation_msg = (
    f"reee.engine is deprecated (since {DEPRECATION_DATE}) "
    f"and {'was scheduled for removal on' if _past_removal else 'will be removed on'} {REMOVAL_DATE}. "
    "Use PrincipledSurfaceBuilder, PrincipledEventBuilder, StoryBuilder directly. "
    "See reee/deprecated/RELIC.md for migration guide."
)

warnings.warn(_deprecation_msg, DeprecationWarning, stacklevel=2)
_logger.warning(_deprecation_msg)

from .deprecated._engine import (
    Engine,
    EmergenceEngine,
)

__all__ = [
    'Engine',
    'EmergenceEngine',
]
