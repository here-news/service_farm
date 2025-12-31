"""Meta-claims and tension detection module."""

from .detectors import (
    detect_tensions,
    TensionDetector,
    get_unresolved,
    count_by_type,
    resolve_meta_claim
)

__all__ = [
    'detect_tensions',
    'TensionDetector',
    'get_unresolved',
    'count_by_type',
    'resolve_meta_claim'
]
