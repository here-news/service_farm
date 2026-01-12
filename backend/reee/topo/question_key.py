"""
Question Key Extraction - Derive question_key from claim semantics.

Pure function - no DB, no LLM.
LLM-extracted keys arrive via explicit_key parameter.

The question_key identifies the typed proposition:
- "fire_death_count" for claims about death toll
- "policy_status" for claims about policy state
- "person_location" for claims about where someone is

Fallback hierarchy:
1. EXPLICIT: Trusted LLM-extracted or external key
2. PATTERN: Pattern-matched from text (death_count, policy_status)
3. ENTITY: Entity-derived (about_X_Y)
4. PAGE_SCOPE: Page/source fallback
5. SINGLETON: Never collapse (unique per claim)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, FrozenSet, List
import re


class FallbackLevel(Enum):
    """Which fallback level was used for question_key extraction."""

    EXPLICIT = 1  # Trusted LLM-extracted or external key
    PATTERN = 2  # Pattern-matched (death_count, policy_status)
    ENTITY = 3  # Entity-derived (about_X_Y)
    PAGE_SCOPE = 4  # Page/source fallback
    SINGLETON = 5  # Never collapse


@dataclass(frozen=True)
class QuestionKeyResult:
    """Result of question_key extraction."""

    question_key: str
    fallback_level: FallbackLevel
    confidence: float  # How confident are we in this key? [0, 1]


# Pattern definitions for typed questions
DEATH_PATTERNS: List[str] = [
    "kill",
    "dead",
    "death",
    "fatality",
    "fatalities",
    "died",
    "perish",
    "casualty",
    "casualties",
]

INJURY_PATTERNS: List[str] = [
    "injur",
    "wound",
    "hurt",
    "hospitali",
]

STATUS_PATTERNS: List[str] = [
    "status",
    "condition",
    "state",
    "ongoing",
    "active",
    "resolved",
]

POLICY_PATTERNS: List[str] = [
    "announc",
    "policy",
    "legislation",
    "bill",
    "reform",
    "law",
]

LOCATION_PATTERNS: List[str] = [
    "located",
    "found",
    "spotted",
    "seen at",
    "arrived",
    "visited",
    "went to",
]

# Event type patterns for prefixing typed questions
EVENT_TYPE_PATTERNS = {
    "fire": ["fire", "blaze", "burn", "inferno"],
    "flood": ["flood", "deluge", "inundation"],
    "earthquake": ["earthquake", "quake", "seismic", "tremor"],
    "storm": ["storm", "typhoon", "hurricane", "cyclone"],
    "accident": ["crash", "accident", "collision", "wreck"],
    "explosion": ["explosion", "blast", "detonate"],
}


def extract_question_key(
    text: str,
    entities: FrozenSet[str],
    anchors: FrozenSet[str],
    page_id: Optional[str],
    claim_id: str,
    explicit_key: Optional[str] = None,
    explicit_confidence: float = 0.9,
) -> QuestionKeyResult:
    """Extract question_key with explicit fallback chain.

    Pure function - no DB, no LLM.
    LLM result arrives via explicit_key parameter.

    Args:
        text: Claim text to analyze
        entities: All entities in claim
        anchors: Anchor entities (subset of entities)
        page_id: Page ID for page_scope fallback
        claim_id: Claim ID for singleton fallback
        explicit_key: LLM-extracted key (if available)
        explicit_confidence: Confidence in explicit_key

    Returns:
        QuestionKeyResult with key, fallback level, and confidence
    """
    text_lower = text.lower() if text else ""

    # Level 1: Explicit (from LLM or upstream)
    if explicit_key and explicit_confidence >= 0.7:
        return QuestionKeyResult(
            question_key=explicit_key,
            fallback_level=FallbackLevel.EXPLICIT,
            confidence=explicit_confidence,
        )

    # Level 2: Pattern matching
    pattern_result = _match_pattern(text_lower)
    if pattern_result:
        return QuestionKeyResult(
            question_key=pattern_result,
            fallback_level=FallbackLevel.PATTERN,
            confidence=0.8,
        )

    # No semantic signal available (EXPLICIT or PATTERN)
    # Go directly to SINGLETON - don't collapse without semantic evidence
    #
    # NOTE: ENTITY and PAGE_SCOPE fallbacks were removed because:
    # - Entity overlap defines SCOPE, not QUESTION
    # - "about_X_Y" keys pretend to be semantic but aren't
    # - PAGE_SCOPE would collapse unrelated claims from same source
    # - Better to under-merge and let reconciliation/improved extraction merge later
    return QuestionKeyResult(
        question_key=f"singleton_{claim_id}",
        fallback_level=FallbackLevel.SINGLETON,
        confidence=0.1,
    )


def _match_pattern(text: str) -> Optional[str]:
    """Match text against typed question patterns.

    Returns question_key if pattern matches, None otherwise.
    """
    if not text:
        return None

    # Infer event type for prefixing
    event_type = _infer_event_type(text)

    # Check death patterns
    if _matches_any(text, DEATH_PATTERNS):
        return f"{event_type}_death_count"

    # Check injury patterns
    if _matches_any(text, INJURY_PATTERNS):
        return f"{event_type}_injury_count"

    # Check status patterns
    if _matches_any(text, STATUS_PATTERNS):
        return f"{event_type}_status"

    # Check policy patterns
    if _matches_any(text, POLICY_PATTERNS):
        return "policy_announcement"

    # Check location patterns (person location claims)
    if _matches_any(text, LOCATION_PATTERNS):
        return "person_location"

    return None


def _infer_event_type(text: str) -> str:
    """Infer event type from text for prefixing typed questions."""
    for event_type, patterns in EVENT_TYPE_PATTERNS.items():
        if _matches_any(text, patterns):
            return event_type
    return "incident"


def _matches_any(text: str, patterns: List[str]) -> bool:
    """Check if text matches any of the patterns."""
    return any(p in text for p in patterns)


def _normalize_name(name: str) -> str:
    """Normalize entity name for use in question_key."""
    normalized = name.lower()
    normalized = re.sub(r"[\s'\-]", "_", normalized)
    normalized = re.sub(r"[^a-z0-9_]", "", normalized)
    return normalized[:30]  # Limit length


def get_fallback_description(level: FallbackLevel) -> str:
    """Get human-readable description of fallback level."""
    descriptions = {
        FallbackLevel.EXPLICIT: "LLM-extracted or externally provided",
        FallbackLevel.PATTERN: "Pattern-matched from text",
        FallbackLevel.ENTITY: "Derived from anchor entities",
        FallbackLevel.PAGE_SCOPE: "Fallback to page/source scope",
        FallbackLevel.SINGLETON: "Unique to this claim (no collapse)",
    }
    return descriptions.get(level, "Unknown")


def is_typed_question(question_key: str) -> bool:
    """Check if question_key represents a typed (numeric) question.

    Typed questions have Jaynes posteriors over values.
    """
    typed_suffixes = ["_count", "_percentage", "_amount", "_rate"]
    return any(question_key.endswith(suffix) for suffix in typed_suffixes)


def extract_question_type(question_key: str) -> Optional[str]:
    """Extract the question type from a question_key.

    Returns the type suffix (e.g., "death_count" -> "count").
    """
    for suffix in ["_count", "_status", "_location", "_announcement"]:
        if question_key.endswith(suffix):
            return suffix[1:]  # Remove leading underscore
    return None
