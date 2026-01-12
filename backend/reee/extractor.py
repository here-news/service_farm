"""
Extractor Module [DEPRECATED - STUB]
====================================

This module is DEPRECATED and scheduled for removal on 2026-02-01.

Replacement:
- Claim extraction: Use worker layer (claim_loader.py) for structured extraction
- Question detection: Use `reee.builders.surface_builder` for aboutness logic

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
        "reee.extractor has been removed. "
        "Use claim_loader.py for extraction, PrincipledSurfaceBuilder for aboutness. "
        "See reee/deprecated/RELIC.md for migration guide."
    )

_deprecation_msg = (
    f"reee.extractor is deprecated (since {DEPRECATION_DATE}) "
    f"and {'was scheduled for removal on' if _past_removal else 'will be removed on'} {REMOVAL_DATE}. "
    "Use claim_loader.py for extraction, PrincipledSurfaceBuilder for aboutness. "
    "See reee/deprecated/RELIC.md for migration guide."
)

warnings.warn(_deprecation_msg, DeprecationWarning, stacklevel=2)
_logger.warning(_deprecation_msg)

from .deprecated._extractor import (
    QuestionType,
    ExtractedClaim,
    ClaimExtractor,
)

__all__ = [
    'QuestionType',
    'ExtractedClaim',
    'ClaimExtractor',
]
