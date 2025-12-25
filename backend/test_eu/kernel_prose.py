"""
Kernel Prose Generation - Layer 4
==================================

Generates citation-rich prose from enriched topology.
Entity and belief IDs embedded for UI linking.

Usage:
    topology = enriched_kernel.get_topology()
    prose = await generate_prose(topology, llm)
"""

import json
import re
from typing import Dict, List, Optional, Any
from openai import AsyncOpenAI

from kernel_enriched import EnrichedTopology, EnrichedBelief


def format_beliefs_for_prompt(topology: EnrichedTopology) -> str:
    """
    Format beliefs with IDs and source counts (superscript notation).
    LLM will organize thematically - no hardcoded categories.
    """
    # Superscript digits for compact source count display
    superscripts = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
                    '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'}

    def to_superscript(n: int) -> str:
        return ''.join(superscripts.get(d, d) for d in str(n))

    # Sort by certainty descending (more sources = higher certainty)
    sorted_beliefs = sorted(topology.beliefs, key=lambda b: -b.certainty)

    lines = []
    for b in sorted_beliefs:
        src_count = len(b.sources)
        lines.append(f"[{b.id}] {b.text} ˣ{to_superscript(src_count)}")

    return "\n".join(lines)


def format_entities_for_prompt(topology: EnrichedTopology) -> str:
    """Format entity lookup for the prompt."""
    if not topology.entity_lookup:
        return "(no entities extracted)"

    lines = []
    for name, eid in sorted(topology.entity_lookup.items()):
        lines.append(f"{name} → {eid}")
    return "\n".join(lines)


def format_conflicts_for_prompt(topology: EnrichedTopology) -> str:
    """Format conflicts for the prompt."""
    if not topology.conflicts:
        return "(no conflicts)"

    lines = []
    for c in topology.conflicts:
        lines.append(f"[{c.id}] {c.new_claim[:80]}...")
        if c.existing_belief_text:
            lines.append(f"   vs: {c.existing_belief_text[:60]}...")
    return "\n".join(lines)


async def generate_prose(
    topology: EnrichedTopology,
    llm: AsyncOpenAI,
    model: str = "gpt-5.2"
) -> str:
    """
    Generate citation-rich prose from enriched topology.

    Output includes:
    - Entity IDs on first mention: "Wang Fuk Court [en_tsrl5p2z]"
    - Belief citations: "killed 160 people [bl_001]"
    - Certainty hedging based on source count
    - Thematic sections
    """
    if not topology.beliefs:
        return "Awaiting information..."

    beliefs_text = format_beliefs_for_prompt(topology)
    entities_text = format_entities_for_prompt(topology)
    conflicts_text = format_conflicts_for_prompt(topology)

    prompt = f"""Write a compact news narrative from these beliefs.

BELIEFS (ˣⁿ = n sources):
{beliefs_text}

ENTITIES:
{entities_text}

CONFLICTS:
{conflicts_text}

FORMAT:
- Cite inline: "160 dead [bl_001]" not "According to sources, 160 died"
- Entity ID on first mention only: "Wang Fuk Court [en_xxx]"
- Use ˣⁿ suffix for uncertain facts: "128 trucks deployed [bl_002]ˣ¹"
- Short paragraphs, no filler phrases
- Thematic headers (## Casualties, ## Response, etc.)
- ## Developing section if conflicts exist

Be CONCISE. No "according to" or "reportedly" - use ˣⁿ notation instead."""

    try:
        response = await llm.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=2500
        )
        prose = response.choices[0].message.content
        return prose
    except Exception as e:
        return f"Error generating narrative: {e}"


async def generate_prose_from_dict(
    topology_dict: Dict,
    llm: AsyncOpenAI,
    model: str = "gpt-5.2"
) -> str:
    """
    Generate prose from topology dictionary (loaded from JSON).
    Converts to EnrichedTopology internally.
    """
    from kernel_enriched import EnrichedBelief, EnrichedConflict, EntropyPoint

    # Reconstruct EnrichedTopology from dict
    beliefs = [
        EnrichedBelief(
            id=b['id'],
            text=b['text'],
            sources=b['sources'],
            claim_ids=b.get('claim_ids', []),
            entity_ids=b.get('entity_ids', []),
            certainty=b.get('certainty', 0.5),
            category=b.get('category'),
            supersedes_id=b.get('supersedes_id'),
            supersedes_text=b.get('supersedes_text'),
            last_updated=b.get('last_updated', '')
        )
        for b in topology_dict.get('beliefs', [])
    ]

    conflicts = [
        EnrichedConflict(
            id=c.get('id', f"cf_{i:03d}"),
            new_claim=c.get('new_claim', ''),
            new_claim_id=c.get('new_claim_id'),
            existing_belief_id=c.get('existing_belief_id'),
            existing_belief_text=c.get('existing_belief_text'),
            topic=c.get('topic'),
            reasoning=c.get('reasoning', '')
        )
        for i, c in enumerate(topology_dict.get('conflicts', []))
    ]

    entropy_trajectory = [
        EntropyPoint(
            claim_index=e['claim_index'],
            entropy=e['entropy'],
            coherence=e['coherence'],
            belief_count=e['belief_count'],
            conflict_count=e['conflict_count']
        )
        for e in topology_dict.get('entropy_trajectory', [])
    ]

    topology = EnrichedTopology(
        beliefs=beliefs,
        conflicts=conflicts,
        entropy_trajectory=entropy_trajectory,
        metrics=topology_dict.get('metrics', {}),
        relations=topology_dict.get('relations', {}),
        entity_lookup=topology_dict.get('entity_lookup', {})
    )

    return await generate_prose(topology, llm, model)


# CLI for testing
if __name__ == '__main__':
    import asyncio
    import sys
    import os

    async def main():
        if len(sys.argv) < 2:
            print("Usage: python kernel_prose.py <enriched_topology.json> [--model MODEL]")
            return

        # Parse args
        model = "gpt-5.2"
        for i, arg in enumerate(sys.argv):
            if arg == "--model" and i + 1 < len(sys.argv):
                model = sys.argv[i + 1]

        # Load topology
        with open(sys.argv[1]) as f:
            topology_dict = json.load(f)

        llm = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        print(f"Generating prose from {sys.argv[1]} using {model}...")
        print("=" * 60)

        prose = await generate_prose_from_dict(topology_dict, llm, model)
        print(prose)

    asyncio.run(main())
