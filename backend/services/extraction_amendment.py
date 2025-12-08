"""
Extraction Amendment Service - Correct/extend extractions based on instructions

Instead of re-extracting everything, this service takes a correction instruction
and returns only the amended items (new mentions, corrected descriptions, etc.)

Use cases:
- "We missed Muhammad Suleman" → Extract just that entity
- "Block 7 should have alias Wang Tai House" → Add alias
- "Ho Wai-ho is a firefighter, not a businessman" → Correct description
- "Missing blocks 1, 2, 4, 5" → Extract those locations

This allows iterative refinement without reprocessing correct extractions.
"""
import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from openai import OpenAI
import os

from models.mention import Mention, MentionRelationship

logger = logging.getLogger(__name__)


@dataclass
class Amendment:
    """A single correction/addition to the extraction."""
    action: str  # "add_mention", "add_alias", "correct_description", "add_relationship"

    # For add_mention
    mention: Optional[Mention] = None

    # For add_alias
    target_surface_form: Optional[str] = None
    alias: Optional[str] = None

    # For correct_description
    new_description: Optional[str] = None

    # For add_relationship
    relationship: Optional[MentionRelationship] = None


@dataclass
class AmendmentResult:
    """Result from amendment extraction."""
    amendments: List[Amendment] = field(default_factory=list)
    instruction: str = ""
    raw_response: str = ""

    @property
    def new_mentions(self) -> List[Mention]:
        return [a.mention for a in self.amendments if a.action == "add_mention" and a.mention]

    @property
    def new_relationships(self) -> List[MentionRelationship]:
        return [a.relationship for a in self.amendments if a.action == "add_relationship" and a.relationship]


class ExtractionAmendmentService:
    """
    Service to amend extractions based on correction instructions.
    """

    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def amend(
        self,
        instruction: str,
        original_content: str,
        existing_mentions: List[Mention] = None,
        existing_relationships: List[MentionRelationship] = None
    ) -> AmendmentResult:
        """
        Apply an amendment instruction to extract missing/corrected items.

        Args:
            instruction: What to fix, e.g., "We missed Muhammad Suleman"
            original_content: The original article text
            existing_mentions: Already extracted mentions (for context)
            existing_relationships: Already extracted relationships

        Returns:
            AmendmentResult with only the new/corrected items
        """
        existing_mentions = existing_mentions or []
        existing_relationships = existing_relationships or []

        # Build context of what we already have
        existing_summary = self._summarize_existing(existing_mentions, existing_relationships)

        system_prompt = """You are an extraction amendment assistant. Given an instruction about what was missed or incorrect,
extract ONLY the missing/corrected items from the content.

Do NOT re-extract items that are already correct. Only output what the instruction asks for.

Output JSON with these possible amendment types:
1. add_mention - A new entity mention that was missed
2. add_alias - An alias for an existing entity
3. correct_description - A corrected description for an entity
4. add_relationship - A relationship between entities"""

        user_prompt = f"""INSTRUCTION: {instruction}

ALREADY EXTRACTED (do not repeat these):
{existing_summary}

ORIGINAL CONTENT:
{original_content[:4000]}

Based on the instruction, extract ONLY the missing/corrected items.

Return JSON:
{{
    "amendments": [
        {{
            "action": "add_mention",
            "mention": {{
                "surface_form": "Muhammad Suleman",
                "type_hint": "PERSON",
                "context": "Eyewitness Muhammad Suleman was driving past when he saw huge plumes of dark smoke",
                "description": "Eyewitness who was driving past Wang Fuk Court during the fire",
                "aliases": []
            }}
        }},
        {{
            "action": "add_alias",
            "target_surface_form": "Block 7",
            "alias": "Wang Tai House"
        }},
        {{
            "action": "correct_description",
            "target_surface_form": "Ho Wai-ho",
            "new_description": "37-year-old firefighter with nine years of service who died in the fire"
        }},
        {{
            "action": "add_relationship",
            "relationship": {{
                "subject": "Block 2",
                "predicate": "PART_OF",
                "object": "Wang Fuk Court"
            }}
        }}
    ]
}}

Only include amendments that address the instruction. If the instruction mentions multiple items, include all of them."""

        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=2000
            )

            raw_response = response.choices[0].message.content
            result_data = json.loads(raw_response)

            # Parse amendments
            amendments = []
            for a in result_data.get('amendments', []):
                action = a.get('action')

                if action == 'add_mention' and a.get('mention'):
                    m = a['mention']
                    mention = Mention(
                        id=f"amend_{len(amendments)}",
                        surface_form=m.get('surface_form', ''),
                        type_hint=m.get('type_hint', 'UNKNOWN'),
                        context=m.get('context', ''),
                        description=m.get('description', ''),
                        aliases=m.get('aliases', [])
                    )
                    amendments.append(Amendment(action='add_mention', mention=mention))

                elif action == 'add_alias':
                    amendments.append(Amendment(
                        action='add_alias',
                        target_surface_form=a.get('target_surface_form'),
                        alias=a.get('alias')
                    ))

                elif action == 'correct_description':
                    amendments.append(Amendment(
                        action='correct_description',
                        target_surface_form=a.get('target_surface_form'),
                        new_description=a.get('new_description')
                    ))

                elif action == 'add_relationship' and a.get('relationship'):
                    r = a['relationship']
                    rel = MentionRelationship(
                        subject_id=r.get('subject', ''),
                        predicate=r.get('predicate', ''),
                        object_id=r.get('object', '')
                    )
                    amendments.append(Amendment(action='add_relationship', relationship=rel))

            logger.info(f"✏️ Amendment extracted {len(amendments)} corrections")

            return AmendmentResult(
                amendments=amendments,
                instruction=instruction,
                raw_response=raw_response
            )

        except Exception as e:
            logger.error(f"Amendment extraction failed: {e}")
            return AmendmentResult(instruction=instruction)

    def _summarize_existing(
        self,
        mentions: List[Mention],
        relationships: List[MentionRelationship]
    ) -> str:
        """Summarize existing extractions for context."""
        lines = []

        if mentions:
            lines.append("Mentions:")
            for m in mentions:
                aliases = f" (aka {m.aliases})" if m.aliases else ""
                lines.append(f"  - {m.type_hint}: {m.surface_form}{aliases}")

        if relationships:
            lines.append("\nRelationships:")
            for r in relationships:
                lines.append(f"  - {r.subject_id} --[{r.predicate}]--> {r.object_id}")

        return "\n".join(lines) if lines else "None"


# Convenience function for quick amendments
async def amend_extraction(
    instruction: str,
    content: str,
    existing_mentions: List[Mention] = None
) -> AmendmentResult:
    """
    Quick function to amend an extraction.

    Example:
        result = await amend_extraction(
            "We missed Muhammad Suleman and Harry Cheung",
            article_text,
            existing_mentions
        )
        for mention in result.new_mentions:
            print(f"Add: {mention.surface_form}")
    """
    service = ExtractionAmendmentService()
    return await service.amend(instruction, content, existing_mentions)
