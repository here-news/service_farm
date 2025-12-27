"""
News Domain Adapter
====================

Domain-specific enhancements for news claim processing.

Provides:
- NewsHintExtractor: Detects news-specific update language
- NEWS_PROMPT_TEMPLATE: Optimized prompt for news claims
- generate_news_prose(): News-style summary generation
"""

import re
from typing import Dict, List


class NewsHintExtractor:
    """
    News-specific hint extraction.

    Detects patterns common in news updates:
    - Death toll updates ("rises to", "climbed to")
    - Temporal markers ("as of", "breaking")
    - Official statements ("confirmed", "authorities say")
    """

    # News-specific update phrases
    UPDATE_PHRASES = [
        # Death toll / casualty updates
        'rises to', 'risen to', 'climbed to', 'reached',
        'death toll', 'casualty count', 'fatalities',

        # Temporal updates
        'as of', 'latest', 'updated', 'now', 'breaking',
        'just in', 'developing', 'currently',

        # Progression
        'increased to', 'grown to', 'surpasses', 'hits',
        'has grown', 'continues to', 'spreads to',

        # Official statements
        'confirmed', 'authorities say', 'officials report',
        'government says', 'police say', 'according to'
    ]

    # Location patterns (news often updates locations)
    LOCATION_PATTERNS = [
        r'\bin\s+[A-Z][a-z]+',  # "in London"
        r'\bat\s+[A-Z][a-z]+',  # "at Westminster"
        r'[A-Z][a-z]+\s+district',  # "Tai Po district"
    ]

    def extract(self, text: str) -> Dict:
        """Extract news-specific hints from claim text."""
        text_lower = text.lower()

        hints = {
            'numbers': [],
            'has_update_language': False,
            'numeric_value': None,
            'is_official': False,
            'has_location': False,
            'update_type': None,
        }

        # Extract numbers
        numbers = re.findall(r'\b(\d+)\b', text)
        hints['numbers'] = [int(n) for n in numbers if int(n) > 0]
        if hints['numbers']:
            hints['numeric_value'] = max(hints['numbers'])

        # Check for update language
        for phrase in self.UPDATE_PHRASES:
            if phrase in text_lower:
                hints['has_update_language'] = True
                hints['update_type'] = phrase
                break

        # Check for official sources
        official_markers = ['confirmed', 'authorities', 'officials', 'government', 'police']
        hints['is_official'] = any(m in text_lower for m in official_markers)

        # Check for location mentions
        for pattern in self.LOCATION_PATTERNS:
            if re.search(pattern, text):
                hints['has_location'] = True
                break

        return hints


# News-optimized prompt template
NEWS_PROMPT_TEMPLATE = """Compare a new news claim to existing beliefs about an event.

EXISTING BELIEFS:
{nodes}

NEW CLAIM: "{claim}"
SOURCE: {source}
EXTRACTED: numbers={numbers}, has_update_language={has_update_language}
{similarity_context}

STEP 1 - IDENTIFY THE QUESTION:
What specific question does this claim answer?
Examples: "death count", "location of incident", "cause of fire", "response actions"

STEP 2 - FIND MATCHING BELIEF:
Does any existing belief answer the SAME question?
If not -> NOVEL (different questions = no relationship)

STEP 3 - IF SAME QUESTION, DETERMINE RELATIONSHIP:
- CONFIRMS: Same question, same value (corroboration from another source)
- REFINES: Same question, more specific (adds detail without changing value)
- SUPERSEDES: Same question, different value WITH update language ("rises to", "now", "latest")
- CONFLICTS: Same question, different value, NO temporal ordering

KEY INSIGHT: "11 dead" vs "36 dead" with "rises to" -> SUPERSEDES (death toll updated)
KEY INSIGHT: "36 dead" vs "fire on 14th floor" -> NOVEL (death count != location)
KEY INSIGHT: "fire at Wang Fuk Court" vs "blaze at Wang Fuk" -> CONFIRMS (same location, different wording)

Return JSON:
{{
  "question_answered": "what question this claim answers",
  "matching_belief_question": "what question the matching belief answers (or null)",
  "relation": "NOVEL|CONFIRMS|REFINES|SUPERSEDES|CONFLICTS",
  "affected_belief": <index number or null>,
  "reasoning": "one sentence explanation",
  "normalized_claim": "the claim in clear standalone form"
}}"""


async def generate_news_prose(kernel, llm_client) -> str:
    """
    Generate news-style prose from kernel beliefs.

    Uses epistemic confidence to guide language:
    - Confirmed: state as fact
    - Corroborated: use hedging ("sources report...")
    - Single source: note uncertainty ("one source claims...")
    """
    if not kernel.topo.nodes:
        return "Awaiting information..."

    # Group by confidence
    confirmed = [n for n in kernel.topo.nodes if n.source_count >= 3]
    corroborated = [n for n in kernel.topo.nodes if n.source_count == 2]
    single_source = [n for n in kernel.topo.nodes if n.source_count == 1]

    beliefs_text = ""
    if confirmed:
        beliefs_text += "CONFIRMED (3+ sources):\n"
        beliefs_text += "\n".join(f"- {n.text}" for n in confirmed[:5])
        beliefs_text += "\n\n"
    if corroborated:
        beliefs_text += "CORROBORATED (2 sources):\n"
        beliefs_text += "\n".join(f"- {n.text}" for n in corroborated[:5])
        beliefs_text += "\n\n"
    if single_source[:3]:
        beliefs_text += "REPORTED (single source):\n"
        beliefs_text += "\n".join(f"- {n.text}" for n in single_source[:3])

    if kernel.conflicts:
        beliefs_text += "\n\nUNRESOLVED:\n"
        for c in kernel.conflicts[:2]:
            beliefs_text += f"- Dispute: {c.reasoning[:60]}...\n"

    prompt = f"""Write a news summary based ONLY on these beliefs:

{beliefs_text}

RULES:
- ONLY include facts from beliefs above
- For CONFIRMED: state as fact
- For CORROBORATED: use hedging ("sources report...")
- For REPORTED: note uncertainty ("one source claims...")
- Maximum 100 words

Return ONLY the prose."""

    try:
        response = await llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"
