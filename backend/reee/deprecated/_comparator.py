"""
Claim Comparator - Principled Relationship Detection

Implements the principled relate() function from relate_updates.py
with q1/q2 question comparison from universal_kernel.py.

Key insight: Claims only relate if they answer the SAME question.
Different questions → NOVEL (no epistemic relation)
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum

from reee.extractor import ExtractedClaim, ClaimExtractor


class Relation(Enum):
    """Possible relations between two claims."""
    NOVEL = "novel"              # Different questions, no relation
    CONFIRMS = "confirms"        # Same question, same value
    REFINES = "refines"          # Same question, more specific
    SUPERSEDES = "supersedes"    # Same question, temporal update
    DIVERGENT = "divergent"      # Same question, different values, same time
    CONFLICTS = "conflicts"      # Same question, incompatible values


@dataclass
class ComparisonResult:
    """Result of comparing two claims."""
    relation: Relation
    confidence: float
    shared_attrs: List[str]
    details: dict


class ClaimComparator:
    """
    Compare two claims to determine their epistemic relationship.

    Design principles (from relate_updates.py):
    1. Extract structure from text
    2. Compare structured attributes
    3. Use linguistic markers for temporal order

    Key insight (from universal_kernel.py):
    - q1 = question claim1 answers
    - q2 = question claim2 answers
    - If q1 != q2 → NOVEL (no relation)
    """

    def __init__(self):
        self.extractor = ClaimExtractor()

    def compare(self, c1: ExtractedClaim, c2: ExtractedClaim) -> ComparisonResult:
        """
        Compare two extracted claims.

        From relate_updates.py:
        - CONFIRMS: same attr, same value
        - SUPERSEDES: same attr, diff value + update language
        - CONFLICTS: same attr, diff value + no temporal order
        - NOVEL: different questions
        """
        # Q1/Q2 check: do they answer the same question?
        if not self.extractor.same_question(c1, c2):
            return ComparisonResult(
                relation=Relation.NOVEL,
                confidence=0.5,
                shared_attrs=[],
                details={'reason': 'different questions', 'q1': c1.question, 'q2': c2.question}
            )

        # Find shared attributes
        shared = self.extractor.shared_attributes(c1, c2)

        if not shared:
            return ComparisonResult(
                relation=Relation.NOVEL,
                confidence=0.5,
                shared_attrs=[],
                details={'reason': 'no shared attributes'}
            )

        # Compare each shared attribute
        attr_results = []
        for attr in shared:
            v1 = c1.attrs[attr]
            v2 = c2.attrs[attr]

            if v1 == v2:
                attr_results.append(('CONFIRMS', attr, v1, v2))

            elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                # Numeric comparison
                if c2.is_update or (c1.is_monotonic and v2 > v1):
                    # c2 has update language OR monotonic increase → SUPERSEDES
                    attr_results.append(('SUPERSEDES', attr, v1, v2))
                elif c1.is_update or (c2.is_monotonic and v1 > v2):
                    # c1 is the update (reversed)
                    attr_results.append(('SUPERSEDES', attr, v2, v1))
                else:
                    # Same time, different values → true conflict
                    attr_results.append(('CONFLICTS', attr, v1, v2))
            else:
                # Non-numeric, different values
                attr_results.append(('DIVERGENT', attr, v1, v2))

        # Aggregate results
        if all(r[0] == 'CONFIRMS' for r in attr_results):
            return ComparisonResult(
                relation=Relation.CONFIRMS,
                confidence=0.9,
                shared_attrs=list(shared),
                details={'attrs': attr_results}
            )
        elif any(r[0] == 'CONFLICTS' for r in attr_results):
            return ComparisonResult(
                relation=Relation.CONFLICTS,
                confidence=0.8,
                shared_attrs=list(shared),
                details={'attrs': attr_results}
            )
        elif any(r[0] == 'SUPERSEDES' for r in attr_results):
            return ComparisonResult(
                relation=Relation.SUPERSEDES,
                confidence=0.85,
                shared_attrs=list(shared),
                details={'attrs': attr_results}
            )
        elif any(r[0] == 'DIVERGENT' for r in attr_results):
            return ComparisonResult(
                relation=Relation.DIVERGENT,
                confidence=0.7,
                shared_attrs=list(shared),
                details={'attrs': attr_results}
            )
        else:
            return ComparisonResult(
                relation=Relation.NOVEL,
                confidence=0.5,
                shared_attrs=list(shared),
                details={'attrs': attr_results}
            )

    def compare_texts(self, text1: str, text2: str) -> ComparisonResult:
        """Compare two claim texts directly."""
        c1 = self.extractor.extract(text1)
        c2 = self.extractor.extract(text2)
        return self.compare(c1, c2)


# Standalone function for backwards compatibility with relate_updates.py
def relate(c1_text: str, c2_text: str) -> Tuple[str, float, dict]:
    """
    Principled relate() function.
    Returns: (relation, confidence, details)

    Backwards compatible with relate_updates.py interface.
    """
    comparator = ClaimComparator()
    result = comparator.compare_texts(c1_text, c2_text)
    return (result.relation.value.upper(), result.confidence, result.details)
