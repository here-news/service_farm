"""
Kernel Module [DEPRECATED - STUB]
=================================

This module is DEPRECATED and scheduled for removal on 2026-02-01.

Replacement:
- Belief inference: Use `reee.typed_belief` (TypedBeliefState)
- CountDomain/CategoricalDomain provide typed value domains
- Jaynes-aligned inference with proper noise models

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
        "reee.kernel has been removed. "
        "Use TypedBeliefState from reee.typed_belief for Jaynes-aligned inference. "
        "See reee/deprecated/RELIC.md for migration guide."
    )

_deprecation_msg = (
    f"reee.kernel is deprecated (since {DEPRECATION_DATE}) "
    f"and {'was scheduled for removal on' if _past_removal else 'will be removed on'} {REMOVAL_DATE}. "
    "Use TypedBeliefState from reee.typed_belief for Jaynes-aligned inference. "
    "See reee/deprecated/RELIC.md for migration guide."
)

warnings.warn(_deprecation_msg, DeprecationWarning, stacklevel=2)
_logger.warning(_deprecation_msg)

from .deprecated._kernel import (
    Belief,
    EpistemicKernel,
)

__all__ = [
    'Belief',
    'EpistemicKernel',
]
