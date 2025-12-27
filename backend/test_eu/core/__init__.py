"""
Epistemic Kernel Core Components

Clean, modular implementation of the epistemic kernel.
"""

from .extractor import ClaimExtractor, ExtractedClaim
from .comparator import ClaimComparator, Relation as ComparisonRelation
from .kernel import EpistemicKernel, Belief, Relation

__all__ = [
    'ClaimExtractor',
    'ExtractedClaim',
    'ClaimComparator',
    'ComparisonRelation',
    'EpistemicKernel',
    'Belief',
    'Relation',
]
