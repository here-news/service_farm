"""
Question Key Extraction for L1 Proposition Formation
=====================================================

The question_key identifies WHICH QUESTION a claim answers.
Claims only relate (CONFIRMS/SUPERSEDES/CONFLICTS) if they
answer the SAME question.

Examples:
  "13 dead" -> question_key="death_count", value=13
  "Fire on Floor 8" -> question_key="origin_location", value="Floor 8"
  "Started at 3am" -> question_key="start_time", value="3am"
"""

import json
import re
from typing import Dict, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI

from ..types import Claim, Relation


async def extract_question_key(claim: Claim, llm: 'AsyncOpenAI' = None) -> Dict:
    """
    Extract question_key metadata from claim text.

    Returns dict with:
        question_key: str (e.g., "death_count", "origin_location")
        extracted_value: Any (e.g., 13, "Floor 8")
        value_unit: str (e.g., "people", "floors")
        has_update_language: bool
        is_monotonic: bool
    """
    if not llm:
        return _extract_question_key_rules(claim)

    prompt = f"""Extract the question this claim answers and its value.

CLAIM: "{claim.text}"

What specific question does this claim answer? Focus on the PRIMARY assertion.

Examples:
- "13 dead in fire" -> question: "death_count", value: 13, unit: "people"
- "Fire started on Floor 8" -> question: "origin_floor", value: 8, unit: "floor"
- "Death toll rises to 17" -> question: "death_count", value: 17, has_update: true
- "Jimmy Lai faces trial" -> question: "legal_status", value: "facing_trial"
- "Trump said X" -> question: "trump_statement", value: "X"

Return JSON:
{{
  "question_key": "short_snake_case_key",
  "extracted_value": <number or string>,
  "value_unit": "unit if applicable or null",
  "has_update_language": true/false,
  "is_monotonic": true/false (true for counts that only increase like death tolls),
  "reasoning": "one sentence"
}}"""

    try:
        response = await llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        result = json.loads(response.choices[0].message.content)

        # Update claim with extracted values
        claim.question_key = result.get('question_key')
        claim.extracted_value = result.get('extracted_value')
        claim.value_unit = result.get('value_unit')
        claim.has_update_language = result.get('has_update_language', False)
        claim.is_monotonic = result.get('is_monotonic')

        return result

    except Exception as e:
        return {'error': str(e)}


def _extract_question_key_rules(claim: Claim) -> Dict:
    """
    Rule-based question_key extraction (no LLM).

    Handles common patterns:
    - Death/casualty counts
    - Injury counts
    - Location mentions
    - Time mentions
    """
    text_lower = claim.text.lower()

    # Death/casualty patterns
    death_patterns = [
        r'(\d+)\s*(?:people\s+)?(?:dead|killed|died|deaths?|fatalities)',
        r'death\s+toll\s*(?:of|:)?\s*(\d+)',
        r'death\s+toll\s+(?:rises|rose|climbs|reaches|now|updated)\s*(?:to|at)?\s*(\d+)',
        r'(\d+)\s+(?:people\s+)?(?:were\s+)?killed',
    ]
    for pattern in death_patterns:
        match = re.search(pattern, text_lower)
        if match:
            value = int(match.group(1))
            has_update = any(w in text_lower for w in ['rises', 'risen', 'climbs', 'reaches', 'now', 'updated'])
            claim.question_key = "death_count"
            claim.extracted_value = value
            claim.value_unit = "people"
            claim.has_update_language = has_update
            claim.is_monotonic = True
            return {
                'question_key': 'death_count',
                'extracted_value': value,
                'value_unit': 'people',
                'has_update_language': has_update,
                'is_monotonic': True
            }

    # Injury patterns
    injury_patterns = [
        r'(\d+)\s*(?:people\s+)?(?:injured|wounded|hurt)',
        r'(\d+)\s+(?:others?\s+)?(?:were\s+)?injured',
    ]
    for pattern in injury_patterns:
        match = re.search(pattern, text_lower)
        if match:
            value = int(match.group(1))
            has_update = any(w in text_lower for w in ['rises', 'risen', 'climbs', 'reaches', 'now'])
            claim.question_key = "injury_count"
            claim.extracted_value = value
            claim.value_unit = "people"
            claim.has_update_language = has_update
            claim.is_monotonic = True
            return {
                'question_key': 'injury_count',
                'extracted_value': value,
                'value_unit': 'people',
                'has_update_language': has_update,
                'is_monotonic': True
            }

    # No pattern matched
    return {
        'question_key': None,
        'extracted_value': None,
        'reasoning': 'No rule-based pattern matched'
    }


def classify_within_bucket(claim: Claim, other: Claim) -> Tuple[Relation, float, str]:
    """
    Classify relationship between two claims with SAME question_key.

    Since they answer the same question, classification is mostly rule-based:
    - Same value -> CONFIRMS
    - Different value + update language -> SUPERSEDES
    - Different value, no update -> CONFLICTS

    Returns: (relation, confidence, reasoning)
    """
    assert claim.question_key == other.question_key, "Must have same question_key"

    # Same value = CONFIRMS
    if claim.extracted_value == other.extracted_value:
        return (
            Relation.CONFIRMS,
            0.9,
            f"Same {claim.question_key}: {claim.extracted_value}"
        )

    # Different values - check for update language
    if claim.has_update_language or other.has_update_language:
        newer_claim = claim
        older_claim = other

        # If monotonic (death toll), higher value supersedes
        if claim.is_monotonic and isinstance(claim.extracted_value, (int, float)):
            if claim.extracted_value > other.extracted_value:
                newer_claim, older_claim = claim, other
            else:
                newer_claim, older_claim = other, claim

        # Otherwise use timestamp if available
        elif claim.timestamp and other.timestamp:
            if claim.timestamp > other.timestamp:
                newer_claim, older_claim = claim, other
            else:
                newer_claim, older_claim = other, claim

        return (
            Relation.SUPERSEDES,
            0.85,
            f"{claim.question_key}: {older_claim.extracted_value} -> {newer_claim.extracted_value}"
        )

    # Different values, no update language = CONFLICTS
    return (
        Relation.CONFLICTS,
        0.8,
        f"Conflicting {claim.question_key}: {claim.extracted_value} vs {other.extracted_value}"
    )
