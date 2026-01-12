"""
Aboutness Metrics [DEPRECATED - STUB]
=====================================

This module is DEPRECATED and scheduled for removal on 2026-02-01.

Replacement:
- Clustering evaluation: Use standard sklearn metrics or custom evaluation

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
        "reee.aboutness.metrics has been removed. "
        "See reee/deprecated/RELIC.md for migration guide."
    )

_deprecation_msg = (
    f"reee.aboutness.metrics is deprecated (since {DEPRECATION_DATE}) "
    f"and {'was scheduled for removal on' if _past_removal else 'will be removed on'} {REMOVAL_DATE}. "
    "See reee/deprecated/RELIC.md for migration guide."
)

warnings.warn(_deprecation_msg, DeprecationWarning, stacklevel=2)
_logger.warning(_deprecation_msg)

# Re-export from deprecated location
from ..deprecated._aboutness.metrics import (
    B3Metrics,
    PurityMetrics,
    ClusteringEvaluation,
    compute_b3_metrics,
    compute_purity_metrics,
    evaluate_clustering,
    print_evaluation_report,
)

__all__ = [
    'B3Metrics',
    'PurityMetrics',
    'ClusteringEvaluation',
    'compute_b3_metrics',
    'compute_purity_metrics',
    'evaluate_clustering',
    'print_evaluation_report',
]
