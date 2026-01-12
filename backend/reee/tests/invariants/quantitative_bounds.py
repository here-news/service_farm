"""
Quantitative Bounds
===================

These are not hard invariants but expected ranges for healthy kernel behavior.
Violations suggest either bugs or unusual input data (which should emit warnings).
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

try:
    from reee.builders.story_builder import CompleteStory, StoryBuilderResult
except ImportError:
    CompleteStory = Any
    StoryBuilderResult = Any


@dataclass
class BoundViolation:
    """Records a quantitative bound violation."""
    bound: str
    message: str
    expected: Any
    actual: Any


def check_story_count_in_range(
    stories: List[CompleteStory],
    min_count: int,
    max_count: int,
) -> Tuple[bool, List[BoundViolation]]:
    """
    BOUND: Number of stories should be within expected range.

    Too few: underpowered extraction or overly aggressive filtering
    Too many: fragmentation, insufficient merging
    """
    actual = len(stories)
    violations = []

    if actual < min_count:
        violations.append(BoundViolation(
            bound="story_count_range",
            message=f"Too few stories: {actual} < {min_count}",
            expected=f"[{min_count}, {max_count}]",
            actual=actual,
        ))
    elif actual > max_count:
        violations.append(BoundViolation(
            bound="story_count_range",
            message=f"Too many stories: {actual} > {max_count}",
            expected=f"[{min_count}, {max_count}]",
            actual=actual,
        ))

    return len(violations) == 0, violations


def check_periphery_rate_in_range(
    stories: List[CompleteStory],
    min_rate: float,
    max_rate: float,
) -> Tuple[bool, List[BoundViolation]]:
    """
    BOUND: Periphery attachment rate should be within expected range.

    periphery_rate = (total periphery) / (total core + total periphery)

    Too low: too conservative, missing valid attachments
    Too high: too permissive, leaking unrelated incidents
    """
    total_core = sum(len(s.core_a_ids) + len(s.core_b_ids) for s in stories)
    total_periphery = sum(len(s.periphery_incident_ids) for s in stories)
    total = total_core + total_periphery

    if total == 0:
        return True, []

    actual_rate = total_periphery / total
    violations = []

    if actual_rate < min_rate:
        violations.append(BoundViolation(
            bound="periphery_rate_range",
            message=f"Periphery rate too low: {actual_rate:.2%} < {min_rate:.2%}",
            expected=f"[{min_rate:.2%}, {max_rate:.2%}]",
            actual=actual_rate,
        ))
    elif actual_rate > max_rate:
        violations.append(BoundViolation(
            bound="periphery_rate_range",
            message=f"Periphery rate too high: {actual_rate:.2%} > {max_rate:.2%}",
            expected=f"[{min_rate:.2%}, {max_rate:.2%}]",
            actual=actual_rate,
        ))

    return len(violations) == 0, violations


def check_witness_scarcity_below(
    stories: List[CompleteStory],
    max_scarcity: float,
) -> Tuple[bool, List[BoundViolation]]:
    """
    BOUND: Witness scarcity should be below threshold.

    witness_scarcity = (blocked Core-B candidates) / (all Core-B candidates)

    High scarcity means extraction is underpowered (not enough constraints).
    """
    total_blocked = sum(len(s.blocked_core_b) for s in stories)
    total_candidates = sum(s.candidate_pool_size for s in stories)

    if total_candidates == 0:
        return True, []

    actual_scarcity = total_blocked / total_candidates
    violations = []

    if actual_scarcity > max_scarcity:
        violations.append(BoundViolation(
            bound="witness_scarcity_max",
            message=f"Witness scarcity too high: {actual_scarcity:.2%} > {max_scarcity:.2%}",
            expected=f"< {max_scarcity:.2%}",
            actual=actual_scarcity,
        ))

    return len(violations) == 0, violations


def check_max_case_size_below(
    stories: List[CompleteStory],
    max_size: int,
) -> Tuple[bool, List[BoundViolation]]:
    """
    BOUND: Maximum case size should be below threshold.

    Prevents mega-case formation from hub percolation.
    """
    violations = []

    for story in stories:
        core_size = len(story.core_a_ids) + len(story.core_b_ids)
        if core_size > max_size:
            violations.append(BoundViolation(
                bound="max_case_size",
                message=f"Case '{story.spine}' too large: {core_size} > {max_size}",
                expected=f"< {max_size}",
                actual=core_size,
            ))

    return len(violations) == 0, violations


# =============================================================================
# ASSERTION FUNCTIONS (for pytest)
# =============================================================================

def assert_story_count_in_range(stories: List[CompleteStory], min_count: int, max_count: int):
    """Assert story count is in expected range."""
    passed, violations = check_story_count_in_range(stories, min_count, max_count)
    if not passed:
        raise AssertionError(
            f"Story count bound violations: {len(violations)}\n" +
            "\n".join(f"  - {v.message}" for v in violations)
        )


def assert_periphery_rate_in_range(stories: List[CompleteStory], min_rate: float, max_rate: float):
    """Assert periphery rate is in expected range."""
    passed, violations = check_periphery_rate_in_range(stories, min_rate, max_rate)
    if not passed:
        raise AssertionError(
            f"Periphery rate bound violations: {len(violations)}\n" +
            "\n".join(f"  - {v.message}" for v in violations)
        )


def assert_witness_scarcity_below(stories: List[CompleteStory], max_scarcity: float):
    """Assert witness scarcity is below threshold."""
    passed, violations = check_witness_scarcity_below(stories, max_scarcity)
    if not passed:
        raise AssertionError(
            f"Witness scarcity bound violations: {len(violations)}\n" +
            "\n".join(f"  - {v.message}" for v in violations)
        )


def assert_max_case_size_below(stories: List[CompleteStory], max_size: int):
    """Assert no case exceeds max size."""
    passed, violations = check_max_case_size_below(stories, max_size)
    if not passed:
        raise AssertionError(
            f"Max case size violations: {len(violations)}\n" +
            "\n".join(f"  - {v.message}" for v in violations)
        )
