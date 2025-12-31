"""
LLM Interpretation for Surfaces and Events
==========================================

Generates semantic interpretations (titles, descriptions, narratives)
from computed structures.
"""

import json
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI

from .types import Claim, Surface, Event


async def interpret_surface(
    surface: Surface,
    claims: List[Claim],
    llm: 'AsyncOpenAI'
) -> None:
    """
    Generate semantic interpretation for a surface.

    Updates surface.canonical_title, surface.description, surface.key_facts.
    """
    # Get claim texts for this surface
    claim_texts = [c.text for c in claims if c.id in surface.claim_ids][:10]

    if not claim_texts:
        surface.canonical_title = f"Surface {surface.id}"
        surface.description = "No claims"
        return

    prompt = f"""Based on these related claims, generate semantic interpretation:

CLAIMS:
{chr(10).join(f'- {t}' for t in claim_texts)}

ENTITIES: {list(surface.entities)[:10]}
SOURCES: {len(surface.sources)} independent sources
TIME: {surface.time_window[0]} to {surface.time_window[1]}

Generate:
1. canonical_title: Short (3-6 word) reusable title for this event/topic
2. description: One paragraph summary (50-100 words)
3. key_facts: List of 3-5 main facts (confirmed by multiple sources first)

Return JSON:
{{
  "canonical_title": "...",
  "description": "...",
  "key_facts": ["...", "..."]
}}"""

    try:
        response = await llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        result = json.loads(response.choices[0].message.content)

        surface.canonical_title = result.get('canonical_title', f'Surface {surface.id}')
        surface.description = result.get('description', '')
        surface.key_facts = result.get('key_facts', [])

    except Exception as e:
        surface.canonical_title = f"Surface {surface.id}"
        surface.description = f"Error: {e}"


async def interpret_event(
    event: Event,
    surfaces: List[Surface],
    llm: 'AsyncOpenAI'
) -> None:
    """
    Generate narrative interpretation for an event.

    Updates event.canonical_title, event.narrative, event.timeline.
    """
    # Get surface summaries
    surface_summaries = []
    for s in surfaces:
        if s.id in event.surface_ids and s.canonical_title:
            desc = s.description[:100] if s.description else ""
            surface_summaries.append(f"- {s.canonical_title}: {desc}...")

    if not surface_summaries:
        event.canonical_title = f"Event {event.id}"
        return

    prompt = f"""Based on these related topics/surfaces, generate event narrative:

SURFACES:
{chr(10).join(surface_summaries)}

KEY ENTITIES: {list(event.anchor_entities)[:5]}
TIME SPAN: {event.time_window[0]} to {event.time_window[1]}
SOURCES: {event.total_sources} independent sources

Generate:
1. canonical_title: Event title (3-8 words)
2. narrative: Coherent narrative connecting these surfaces (100-150 words)
3. timeline: Key moments in chronological order

Return JSON:
{{
  "canonical_title": "...",
  "narrative": "...",
  "timeline": [{{"time": "...", "event": "..."}}, ...]
}}"""

    try:
        response = await llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        result = json.loads(response.choices[0].message.content)

        event.canonical_title = result.get('canonical_title', f'Event {event.id}')
        event.narrative = result.get('narrative', '')
        event.timeline = result.get('timeline', [])

    except Exception as e:
        event.canonical_title = f"Event {event.id}"
        event.narrative = f"Error: {e}"


async def interpret_all(
    claims: List[Claim],
    surfaces: List[Surface],
    events: List[Event],
    llm: 'AsyncOpenAI'
) -> None:
    """
    Generate semantic interpretation for all surfaces and events.
    """
    # Interpret surfaces first
    for surface in surfaces:
        await interpret_surface(surface, claims, llm)

    # Then interpret events
    for event in events:
        await interpret_event(event, surfaces, llm)
