"""
Claim Extractor - Structured Value Extraction

Consolidates extraction logic from:
- relate_updates.py: extract_structure()
- universal_kernel.py: q1/q2 pattern
- uee_server.py: numeric_value, is_monotonic

The key insight from q1/q2: claims only relate if they answer the SAME question.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class QuestionType(Enum):
    """Types of questions a claim can answer."""
    COUNT_DEATHS = "count of deaths"
    COUNT_INJURED = "count of injured"
    COUNT_MISSING = "count of missing"
    COUNT_ARRESTS = "count of arrests"
    COUNT_RESPONDERS = "count of responders"
    LOCATION_BUILDING = "location (building)"
    LOCATION_DISTRICT = "location (district)"
    LOCATION_REGION = "location (region)"
    TIME_INCIDENT = "time of incident"
    CAUSE = "cause of incident"
    RESPONSE = "official response"
    OTHER = "other"


@dataclass
class ExtractedClaim:
    """Structured extraction from a claim."""
    text: str
    attrs: Dict[str, Any] = field(default_factory=dict)
    numeric_value: Optional[float] = None
    temporal_markers: List[str] = field(default_factory=list)
    question: Optional[str] = None
    question_type: QuestionType = QuestionType.OTHER
    is_update: bool = False
    is_monotonic: bool = False  # True for counts that typically increase


class ClaimExtractor:
    """
    Extract structured values from claim text.

    Design principle: Extract the QUESTION the claim answers,
    not just the values. Two claims relate only if same question.
    """

    # Patterns for numeric extraction (death toll, injuries, etc.)
    DEATH_PATTERNS = [
        (r'(\d+)\s*(?:people\s+)?(?:were\s+|have\s+been\s+)?killed', 'death_toll'),
        (r'(\d+)\s*(?:people\s+)?(?:have\s+)?died', 'death_toll'),
        (r'(\d+)\s*(?:people\s+)?dead', 'death_toll'),
        (r'death\s+toll[:\s]+(\d+)', 'death_toll'),
        (r'death\s+toll\s+(?:rises?\s+to|hits|of|reached?)\s+(\d+)', 'death_toll'),
        (r'at\s+least\s+(\d+)\s+(?:people\s+)?(?:killed|dead|died)', 'death_toll'),
        (r'(\d+)\s+(?:confirmed?\s+)?(?:deaths?|fatalities)', 'death_toll'),
    ]

    INJURY_PATTERNS = [
        (r'(\d+)\s+(?:people\s+)?injured', 'injury_count'),
        (r'(\d+)\s+(?:people\s+)?(?:were\s+)?hurt', 'injury_count'),
        (r'(\d+)\s+hospitali[sz]ed', 'injury_count'),
    ]

    MISSING_PATTERNS = [
        (r'(\d+)\s+(?:people\s+)?(?:are\s+)?missing', 'missing_count'),
        (r'(\d+)\s+(?:people\s+)?unaccounted', 'missing_count'),
    ]

    ARREST_PATTERNS = [
        (r'(\d+)\s+(?:people\s+)?arrested', 'arrests'),
        (r'arrest(?:ed|s)?\s+(\d+)', 'arrests'),
    ]

    RESPONDER_PATTERNS = [
        (r'(\d+)\s+firefighters?', 'firefighters'),
        (r'over\s+(\d+)\s+firefighters?', 'firefighters'),
    ]

    LOCATION_PATTERNS = [
        (r'(wang\s+fuk\s+court)', 'building'),
        (r'(tai\s+po(?:\s+district)?)', 'district'),
        (r'(hong\s+kong)', 'region'),
    ]

    TIME_PATTERNS = [
        (r'(\d{1,2}:\d{2}\s*(?:am|pm))', 'time'),
        (r'(\d{1,2}(?::\d{2})?\s*(?:am|pm))', 'time'),
        (r'around\s+(\d{1,2}:\d{2})', 'time'),
    ]

    # Temporal markers indicating updates
    UPDATE_PHRASES = [
        'rises to', 'risen to', 'increased to', 'now',
        'updated', 'latest', 'climbed to', 'reached',
        'has grown', 'hits', 'surpasses', 'as of',
        'confirmed', 'revised to'
    ]

    # Monotonic topics (typically increase over time)
    MONOTONIC_TOPICS = {'death_toll', 'injury_count', 'missing_count', 'arrests'}

    def extract(self, text: str) -> ExtractedClaim:
        """
        Extract structured data from claim text.

        Returns ExtractedClaim with:
        - attrs: {death_toll: 160, location: "Wang Fuk Court"}
        - numeric_value: primary numeric value
        - question: "count of deaths" (q1/q2 pattern)
        - is_update: True if contains temporal markers
        """
        text_lower = text.lower()
        attrs = {}
        numeric_value = None
        question = None
        question_type = QuestionType.OTHER

        # Extract death toll
        for pattern, attr in self.DEATH_PATTERNS:
            m = re.search(pattern, text_lower)
            if m:
                attrs[attr] = int(m.group(1))
                numeric_value = float(m.group(1))
                question = "count of deaths"
                question_type = QuestionType.COUNT_DEATHS
                break

        # Extract injury count
        for pattern, attr in self.INJURY_PATTERNS:
            m = re.search(pattern, text_lower)
            if m:
                attrs[attr] = int(m.group(1))
                if numeric_value is None:
                    numeric_value = float(m.group(1))
                    question = "count of injured"
                    question_type = QuestionType.COUNT_INJURED
                break

        # Extract missing count
        for pattern, attr in self.MISSING_PATTERNS:
            m = re.search(pattern, text_lower)
            if m:
                attrs[attr] = int(m.group(1))
                if numeric_value is None:
                    numeric_value = float(m.group(1))
                    question = "count of missing"
                    question_type = QuestionType.COUNT_MISSING
                break

        # Extract arrests
        for pattern, attr in self.ARREST_PATTERNS:
            m = re.search(pattern, text_lower)
            if m:
                attrs[attr] = int(m.group(1))
                if numeric_value is None:
                    numeric_value = float(m.group(1))
                    question = "count of arrests"
                    question_type = QuestionType.COUNT_ARRESTS
                break

        # Extract responders
        for pattern, attr in self.RESPONDER_PATTERNS:
            m = re.search(pattern, text_lower)
            if m:
                attrs[attr] = int(m.group(1))
                if question is None:
                    question = "count of responders"
                    question_type = QuestionType.COUNT_RESPONDERS
                break

        # Extract locations
        for pattern, attr in self.LOCATION_PATTERNS:
            m = re.search(pattern, text_lower)
            if m:
                attrs[attr] = m.group(1).strip()
                if question is None:
                    question = f"location ({attr})"
                    question_type = getattr(QuestionType, f"LOCATION_{attr.upper()}", QuestionType.OTHER)

        # Extract times
        for pattern, attr in self.TIME_PATTERNS:
            m = re.search(pattern, text_lower)
            if m:
                attrs[attr] = m.group(1).strip()
                if question is None:
                    question = "time of incident"
                    question_type = QuestionType.TIME_INCIDENT
                break

        # Extract temporal markers
        temporal_markers = []
        for phrase in self.UPDATE_PHRASES:
            if phrase in text_lower:
                temporal_markers.append(phrase)

        is_update = len(temporal_markers) > 0

        # Determine if monotonic (counts that increase)
        is_monotonic = any(attr in self.MONOTONIC_TOPICS for attr in attrs)

        return ExtractedClaim(
            text=text,
            attrs=attrs,
            numeric_value=numeric_value,
            temporal_markers=temporal_markers,
            question=question,
            question_type=question_type,
            is_update=is_update,
            is_monotonic=is_monotonic
        )

    def same_question(self, c1: ExtractedClaim, c2: ExtractedClaim) -> bool:
        """
        Check if two claims answer the same question (q1 == q2).

        This is the key insight from universal_kernel.py:
        - "50 apples picked" vs "50 oranges shipped" → NOVEL (different questions)
        - "13 dead" vs "160 dead" → same question, different values
        """
        if c1.question is None or c2.question is None:
            return False
        return c1.question == c2.question

    def shared_attributes(self, c1: ExtractedClaim, c2: ExtractedClaim) -> set:
        """Get attributes shared between two claims."""
        return set(c1.attrs.keys()) & set(c2.attrs.keys())
