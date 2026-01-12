"""
Aboutness Module [DEPRECATED - STUB]
====================================

This module is DEPRECATED and scheduled for removal on 2026-02-01.

Replacement:
- Membership decisions: Use `reee.membrane` (classify_incident_membership)
- Story composition: Use `reee.builders.story_builder` (StoryBuilder)
- Candidate retrieval: Use worker/index layer

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
        "reee.aboutness has been removed. "
        "Use membrane for membership, StoryBuilder for composition. "
        "See reee/deprecated/RELIC.md for migration guide."
    )

_deprecation_msg = (
    f"reee.aboutness is deprecated (since {DEPRECATION_DATE}) "
    f"and {'was scheduled for removal on' if _past_removal else 'will be removed on'} {REMOVAL_DATE}. "
    "Use membrane for membership, StoryBuilder for composition. "
    "See reee/deprecated/RELIC.md for migration guide."
)

warnings.warn(_deprecation_msg, DeprecationWarning, stacklevel=2)
_logger.warning(_deprecation_msg)

from ..deprecated._aboutness.scorer import (
    AboutnessScorer,
    compute_aboutness_edges,
    compute_events_from_aboutness,
)
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
    'AboutnessScorer',
    'compute_aboutness_edges',
    'compute_events_from_aboutness',
    'B3Metrics',
    'PurityMetrics',
    'ClusteringEvaluation',
    'compute_b3_metrics',
    'compute_purity_metrics',
    'evaluate_clustering',
    'print_evaluation_report',
]
