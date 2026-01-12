"""
Artifact Extractor: LLM-based entity classification for membrane compiler.

This module extracts typed artifacts from raw incident data. It is the
boundary between perception (LLM, embeddings) and compilation (membrane).

Design Principles:
1. Output entity IDs (not strings) and role polarity
2. Make "broad vs specific" an explicit artifact field (LLM-decided)
3. When unsure, prefer context + raise InquirySeed; don't gamble spine
4. All uncertainty becomes DEFER territory

The extractor guarantees:
- referents contains only specific referents (place/person/object)
- contexts contains only broad/ambient entities
- Ambiguous cases emit InquirySeed for human resolution
"""

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Optional, Any

from ..membrane import Referent, ReferentRole, IncidentArtifact


# =============================================================================
# Classification Schema (LLM outputs these)
# =============================================================================

class ReferentType(Enum):
    """Entity classification types - LLM decides these."""
    # Specific referents (→ membrane.referents)
    SPECIFIC_PLACE = "specific_place"    # Building, facility, venue
    PERSON = "person"                     # Individual or named group
    OBJECT = "object"                     # Vehicle, artifact, document

    # Broad context (→ membrane.contexts)
    BROAD_LOCATION = "broad_location"    # City, district, region
    TEMPORAL = "temporal"                 # Time reference
    TOPIC = "topic"                       # Subject area, theme

    # Ambiguous (→ InquirySeed)
    AMBIGUOUS = "ambiguous"              # Unsure, needs human review


@dataclass(frozen=True)
class EntityClassification:
    """Single entity classification from LLM."""
    entity_id: str
    entity_name: str
    classification: ReferentType
    confidence: float
    reasoning: str  # LLM's explanation


@dataclass(frozen=True)
class InquirySeed:
    """Request for human disambiguation."""
    entity_id: str
    entity_name: str
    incident_id: str
    candidates: tuple  # Possible ReferentTypes
    reasoning: str


@dataclass(frozen=True)
class ExtractionResult:
    """Complete extraction result for an incident."""
    incident_id: str
    artifact: IncidentArtifact
    classifications: FrozenSet[EntityClassification]
    inquiries: FrozenSet[InquirySeed]
    prompt_hash: str  # For reproducibility
    model_version: str


# =============================================================================
# Extraction Prompt Template
# =============================================================================

EXTRACTION_PROMPT = '''You are classifying entities for incident grouping.
For each entity, determine if it is:

1. SPECIFIC_PLACE: A specific building, facility, or venue that uniquely identifies
   a physical location (e.g., "Wang Fuk Court", "Stade de France", "JFK Airport")

2. PERSON: An individual person or named group directly involved in the incident
   (e.g., "Donald Trump", "Emergency Services", "Wang family")

3. OBJECT: A specific artifact, vehicle, or document central to the incident
   (e.g., "Flight MH370", "The document", "The stolen vehicle")

4. BROAD_LOCATION: A city, district, region, or country providing geographic context
   but not uniquely identifying the incident (e.g., "Hong Kong", "Tai Po", "China")

5. TEMPORAL: A time reference providing temporal context
   (e.g., "2024", "November", "morning")

6. TOPIC: A subject area or theme providing topical context
   (e.g., "fire safety", "housing", "emergency response")

7. AMBIGUOUS: Cannot determine - entity could be referent OR context depending
   on perspective. Mark as AMBIGUOUS if confidence < 0.7.

Critical Rules:
- "Wang Fuk Court" is SPECIFIC_PLACE (it's a specific building complex)
- "Hong Kong" and "Tai Po" are BROAD_LOCATION (they're geographic regions)
- Only mark AMBIGUOUS if genuinely unsure; prefer BROAD_LOCATION when in doubt
- SPECIFIC_PLACE should uniquely identify WHERE the incident happened
- BROAD_LOCATION just provides geographic context

Incident: {title}

Entities to classify:
{entities}

Respond in JSON format:
{{
  "classifications": [
    {{
      "entity_id": "...",
      "entity_name": "...",
      "classification": "SPECIFIC_PLACE|PERSON|OBJECT|BROAD_LOCATION|TEMPORAL|TOPIC|AMBIGUOUS",
      "confidence": 0.0-1.0,
      "reasoning": "brief explanation"
    }}
  ],
  "overall_confidence": 0.0-1.0
}}
'''


# =============================================================================
# Core Extraction Function
# =============================================================================

async def extract_artifact(
    incident_id: str,
    title: str,
    anchor_entities: set[str],
    entity_lookup: dict[str, dict],  # entity_id -> {name, type, etc}
    llm_client: Any,  # OpenAI client
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> ExtractionResult:
    """
    Extract typed artifact from raw incident data.

    Args:
        incident_id: Unique incident identifier
        title: Incident title/headline
        anchor_entities: Raw entity names from incident
        entity_lookup: Mapping of entity_id -> metadata
        llm_client: OpenAI-compatible async client
        model: LLM model to use
        temperature: Sampling temperature (0.0 for deterministic)

    Returns:
        ExtractionResult with artifact, classifications, and any inquiries
    """
    # Guard: Empty input → empty artifact with minimal confidence (forces DEFER)
    # This prevents LLM hallucination when given no entities to classify
    if not anchor_entities or (not title and not anchor_entities):
        return ExtractionResult(
            incident_id=incident_id,
            artifact=IncidentArtifact(
                incident_id=incident_id,
                referents=frozenset(),
                contexts=frozenset(),
                time_start=None,
                confidence=0.0,  # Zero confidence → guaranteed DEFER
            ),
            classifications=frozenset(),
            inquiries=frozenset(),
            prompt_hash="EMPTY_INPUT",
            model_version=f"{model}_SKIPPED:empty_input",
        )

    # Build entity list for prompt
    entities_text = "\n".join(f"- {name}" for name in sorted(anchor_entities))

    prompt = EXTRACTION_PROMPT.format(
        title=title,
        entities=entities_text,
    )

    # Compute prompt hash for reproducibility
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

    # Call LLM
    try:
        response = await llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        result_text = response.choices[0].message.content
        result = json.loads(result_text)
    except Exception as e:
        # On LLM failure, return conservative artifact (all context)
        return _make_conservative_result(
            incident_id=incident_id,
            anchor_entities=anchor_entities,
            entity_lookup=entity_lookup,
            prompt_hash=prompt_hash,
            model_version=f"{model}_ERROR:{str(e)[:50]}",
        )

    # Parse LLM response into typed structures
    classifications = []
    referents = []
    contexts = set()
    inquiries = []

    overall_confidence = result.get("overall_confidence", 0.8)

    for item in result.get("classifications", []):
        entity_name = item.get("entity_name", "")
        entity_id = _resolve_entity_id(entity_name, anchor_entities, entity_lookup)
        classification_str = item.get("classification", "AMBIGUOUS")
        confidence = float(item.get("confidence", 0.5))
        reasoning = item.get("reasoning", "")

        try:
            classification = ReferentType[classification_str.upper()]
        except KeyError:
            classification = ReferentType.AMBIGUOUS

        ec = EntityClassification(
            entity_id=entity_id,
            entity_name=entity_name,
            classification=classification,
            confidence=confidence,
            reasoning=reasoning,
        )
        classifications.append(ec)

        # Route to referents, contexts, or inquiries
        if classification == ReferentType.AMBIGUOUS or confidence < 0.6:
            # Emit inquiry for human resolution
            inquiries.append(InquirySeed(
                entity_id=entity_id,
                entity_name=entity_name,
                incident_id=incident_id,
                candidates=(ReferentType.SPECIFIC_PLACE, ReferentType.BROAD_LOCATION),
                reasoning=reasoning,
            ))
            # Conservative: put in context, not referent
            contexts.add(entity_id)

        elif classification in (ReferentType.SPECIFIC_PLACE,):
            # Specific place → PLACE referent
            referents.append(Referent(
                entity_id=entity_id,
                role=ReferentRole.PLACE,
            ))

        elif classification == ReferentType.PERSON:
            # Person → PERSON referent
            referents.append(Referent(
                entity_id=entity_id,
                role=ReferentRole.PERSON,
            ))

        elif classification == ReferentType.OBJECT:
            # Object → OBJECT referent
            referents.append(Referent(
                entity_id=entity_id,
                role=ReferentRole.OBJECT,
            ))

        else:
            # BROAD_LOCATION, TEMPORAL, TOPIC → context
            contexts.add(entity_id)

    # Build IncidentArtifact for membrane
    artifact = IncidentArtifact(
        incident_id=incident_id,
        referents=frozenset(referents),
        contexts=frozenset(contexts),
        time_start=None,  # Caller can set this from incident metadata
        confidence=min(overall_confidence, min((c.confidence for c in classifications), default=1.0)),
    )

    return ExtractionResult(
        incident_id=incident_id,
        artifact=artifact,
        classifications=frozenset(classifications),
        inquiries=frozenset(inquiries),
        prompt_hash=prompt_hash,
        model_version=model,
    )


def _resolve_entity_id(
    entity_name: str,
    anchor_entities: set[str],
    entity_lookup: dict[str, dict],
) -> str:
    """
    Resolve entity name to stable ID.

    Prefers canonical entity ID if available in lookup,
    otherwise uses normalized name as fallback ID.
    """
    # Check if name is directly in lookup
    for eid, meta in entity_lookup.items():
        if meta.get("canonical_name") == entity_name or meta.get("name") == entity_name:
            return eid

    # Fallback: use name itself as ID (normalized)
    return f"name:{entity_name.lower().replace(' ', '_')}"


def _make_conservative_result(
    incident_id: str,
    anchor_entities: set[str],
    entity_lookup: dict[str, dict],
    prompt_hash: str,
    model_version: str,
) -> ExtractionResult:
    """
    Create conservative result on LLM failure.
    All entities go to context (safe but low recall).
    """
    contexts = frozenset(
        _resolve_entity_id(name, anchor_entities, entity_lookup)
        for name in anchor_entities
    )

    artifact = IncidentArtifact(
        incident_id=incident_id,
        referents=frozenset(),
        contexts=contexts,
        time_start=None,
        confidence=0.3,  # Low confidence triggers DEFER
    )

    return ExtractionResult(
        incident_id=incident_id,
        artifact=artifact,
        classifications=frozenset(),
        inquiries=frozenset(),
        prompt_hash=prompt_hash,
        model_version=model_version,
    )


# =============================================================================
# Batch Extraction
# =============================================================================

async def extract_artifacts_batch(
    incidents: list[dict],  # [{id, title, anchor_entities}, ...]
    entity_lookup: dict[str, dict],
    llm_client: Any,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    concurrency: int = 10,
) -> dict[str, ExtractionResult]:
    """
    Extract artifacts for multiple incidents in parallel.

    Returns mapping of incident_id -> ExtractionResult.
    """
    import asyncio

    semaphore = asyncio.Semaphore(concurrency)

    async def extract_one(inc: dict) -> tuple[str, ExtractionResult]:
        async with semaphore:
            result = await extract_artifact(
                incident_id=inc["id"],
                title=inc["title"],
                anchor_entities=set(inc.get("anchor_entities", [])),
                entity_lookup=entity_lookup,
                llm_client=llm_client,
                model=model,
                temperature=temperature,
            )
            return inc["id"], result

    tasks = [extract_one(inc) for inc in incidents]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    output = {}
    for r in results:
        if isinstance(r, Exception):
            continue  # Skip failed extractions
        iid, result = r
        output[iid] = result

    return output
