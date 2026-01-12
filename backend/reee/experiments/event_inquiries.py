"""
Event-Based Proto-Inquiry Generator
====================================

Generates proto-inquiries directly from Events (L3) without requiring
typed claims or meta-claims. Uses tension detection on events/surfaces.

Tensions detected:
1. SINGLE_SOURCE: Surfaces with only one source need corroboration
2. CASUALTY_COUNT: Events mentioning casualties need count verification
3. QUOTE_VERIFICATION: Events with quoted statements need verification
4. TIMELINE_GAP: Events with time gaps need clarification

Usage:
    from reee.experiments.event_inquiries import generate_event_inquiries
    protos = await generate_event_inquiries(neo4j)
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Set, Optional, Any
from enum import Enum


class TensionType(Enum):
    """Types of epistemic tension that trigger inquiries."""
    SINGLE_SOURCE = "single_source"
    CASUALTY_COUNT = "casualty_count"
    QUOTE_VERIFICATION = "quote_verification"
    CONFLICTING_CLAIMS = "conflicting_claims"
    TIMELINE_GAP = "timeline_gap"


class InquirySchema(Enum):
    """Schema types for inquiry answers."""
    MONOTONE_COUNT = "monotone_count"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    QUOTE_AUTHENTICITY = "quote_authenticity"


@dataclass
class EventTension:
    """A detected tension in an event."""
    type: TensionType
    severity: float  # 0-1, higher = more urgent
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventProtoInquiry:
    """Proto-inquiry generated from an event."""
    id: str = field(default_factory=lambda: f"proto_{uuid.uuid4().hex[:12]}")

    # Source event
    event_id: str = ""
    event_title: str = ""

    # Question
    question_text: str = ""
    schema_type: InquirySchema = InquirySchema.CATEGORICAL

    # Tensions that triggered this
    tensions: List[EventTension] = field(default_factory=list)

    # Scope
    entities: Set[str] = field(default_factory=set)
    sources: Set[str] = field(default_factory=set)
    surface_count: int = 0

    # Priority
    priority_score: float = 0.0

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "event_id": self.event_id,
            "event_title": self.event_title,
            "question_text": self.question_text,
            "schema_type": self.schema_type.value,
            "tensions": [
                {
                    "type": t.type.value,
                    "severity": t.severity,
                    "description": t.description,
                }
                for t in self.tensions
            ],
            "entities": list(self.entities),
            "source_count": len(self.sources),
            "surface_count": self.surface_count,
            "priority_score": self.priority_score,
        }


# =============================================================================
# TENSION DETECTION
# =============================================================================

# Keywords that indicate casualty events
CASUALTY_KEYWORDS = [
    r'\b(\d+)\s*(dead|killed|died|deaths|casualties|fatalities)',
    r'\b(\d+)\s*(injured|wounded|hurt)',
    r'(fire|shooting|explosion|crash|attack)\b',
]

# Keywords that indicate quotes
QUOTE_PATTERNS = [
    r'"[^"]{20,}"',  # Quoted text > 20 chars
    r'said\s+that',
    r'claimed\s+that',
    r'according\s+to',
]


def detect_tensions(
    event: Dict,
    surfaces: List[Dict],
    claims: List[Dict] = None,
) -> List[EventTension]:
    """Detect epistemic tensions in an event."""
    tensions = []
    title = event.get('title', '') or ''
    description = event.get('description', '') or ''
    text = f"{title} {description}".lower()

    # 1. SINGLE_SOURCE tension
    single_source_surfaces = [s for s in surfaces if len(s.get('sources', [])) == 1]
    if single_source_surfaces:
        ratio = len(single_source_surfaces) / max(len(surfaces), 1)
        tensions.append(EventTension(
            type=TensionType.SINGLE_SOURCE,
            severity=min(1.0, ratio),
            description=f"{len(single_source_surfaces)}/{len(surfaces)} surfaces have single source",
            evidence={
                "single_source_count": len(single_source_surfaces),
                "total_surfaces": len(surfaces),
            }
        ))

    # 2. CASUALTY_COUNT tension
    for pattern in CASUALTY_KEYWORDS:
        if re.search(pattern, text, re.IGNORECASE):
            tensions.append(EventTension(
                type=TensionType.CASUALTY_COUNT,
                severity=0.8,  # High priority - lives matter
                description="Event involves casualties, count may need verification",
                evidence={"matched_pattern": pattern}
            ))
            break  # Only add once

    # 3. QUOTE_VERIFICATION tension
    for pattern in QUOTE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            tensions.append(EventTension(
                type=TensionType.QUOTE_VERIFICATION,
                severity=0.5,
                description="Event contains quoted statements that may need verification",
                evidence={"matched_pattern": pattern}
            ))
            break

    return tensions


def generate_question(event: Dict, tensions: List[EventTension]) -> tuple:
    """Generate question text and schema from tensions."""
    title = event.get('title', '') or 'this event'

    # Priority: casualty > quote > single_source > default
    tension_types = {t.type for t in tensions}

    if TensionType.CASUALTY_COUNT in tension_types:
        # Extract what kind of casualty
        text = f"{title} {event.get('description', '')}".lower()
        if 'dead' in text or 'killed' in text or 'died' in text or 'death' in text:
            question = f"How many people died in: {title[:100]}?"
        elif 'injured' in text or 'wounded' in text:
            question = f"How many people were injured in: {title[:100]}?"
        else:
            question = f"What is the casualty count for: {title[:100]}?"
        return question, InquirySchema.MONOTONE_COUNT

    if TensionType.QUOTE_VERIFICATION in tension_types:
        question = f"Can the quoted statements be verified in: {title[:100]}?"
        return question, InquirySchema.QUOTE_AUTHENTICITY

    if TensionType.SINGLE_SOURCE in tension_types:
        question = f"Can the claims be corroborated for: {title[:100]}?"
        return question, InquirySchema.BOOLEAN

    # Default
    question = f"What are the verified facts about: {title[:100]}?"
    return question, InquirySchema.CATEGORICAL


def calculate_priority(event: Dict, tensions: List[EventTension]) -> float:
    """Calculate priority score for proto-inquiry."""
    score = 0.0

    # Base score from source count (more sources = more important)
    source_count = event.get('source_count', 0) or 0
    score += min(source_count * 5, 30)

    # Add tension severity
    for tension in tensions:
        if tension.type == TensionType.CASUALTY_COUNT:
            score += 40 * tension.severity  # High priority
        elif tension.type == TensionType.SINGLE_SOURCE:
            score += 20 * tension.severity
        elif tension.type == TensionType.QUOTE_VERIFICATION:
            score += 15 * tension.severity
        else:
            score += 10 * tension.severity

    return score


# =============================================================================
# MAIN GENERATOR
# =============================================================================

async def generate_event_inquiries(
    neo4j,
    min_sources: int = 2,
    min_priority: float = 20.0,
    limit: int = 50,
) -> List[EventProtoInquiry]:
    """
    Generate proto-inquiries from events.

    Args:
        neo4j: Neo4jService instance
        min_sources: Minimum source count for event
        min_priority: Minimum priority score to include
        limit: Max inquiries to generate

    Returns:
        List of EventProtoInquiry objects
    """
    # Get events with surfaces
    events = await neo4j._execute_read('''
        MATCH (e:Event)
        WHERE e.source_count >= $min_sources
        RETURN e.id as id, e.title as title, e.description as description,
               e.source_count as source_count, e.surface_count as surface_count,
               e.primary_entity as primary_entity
        ORDER BY e.source_count DESC
        LIMIT $limit
    ''', {'min_sources': min_sources, 'limit': limit * 2})

    protos = []

    for event in events:
        event_id = event['id']

        # Get surfaces for this event
        surfaces = await neo4j._execute_read('''
            MATCH (s:Surface)-[:BELONGS_TO]->(e:Event {id: $eid})
            RETURN s.id as id, s.sources as sources, s.anchor_entities as anchors
        ''', {'eid': event_id})

        # Detect tensions
        tensions = detect_tensions(event, surfaces)

        if not tensions:
            continue  # No tensions = no inquiry needed

        # Generate question
        question, schema = generate_question(event, tensions)

        # Calculate priority
        priority = calculate_priority(event, tensions)

        if priority < min_priority:
            continue

        # Collect entities and sources
        entities = set()
        sources = set()
        for s in surfaces:
            entities.update(s.get('anchors') or [])
            sources.update(s.get('sources') or [])

        if event.get('primary_entity'):
            entities.add(event['primary_entity'])

        # Create proto-inquiry
        proto = EventProtoInquiry(
            event_id=event_id,
            event_title=event.get('title', ''),
            question_text=question,
            schema_type=schema,
            tensions=tensions,
            entities=entities,
            sources=sources,
            surface_count=len(surfaces),
            priority_score=priority,
        )

        protos.append(proto)

    # Sort by priority and limit
    protos.sort(key=lambda p: p.priority_score, reverse=True)
    return protos[:limit]


async def preview_event_inquiries(neo4j, limit: int = 10) -> None:
    """Preview proto-inquiries without persisting."""
    protos = await generate_event_inquiries(neo4j, limit=limit)

    print("=" * 70)
    print("EVENT-BASED PROTO-INQUIRIES")
    print("=" * 70)

    for i, proto in enumerate(protos, 1):
        print(f"\n[{i}] Priority: {proto.priority_score:.1f}")
        print(f"    Event: {proto.event_title[:60]}...")
        print(f"    Question: {proto.question_text}")
        print(f"    Schema: {proto.schema_type.value}")
        print(f"    Sources: {len(proto.sources)}, Surfaces: {proto.surface_count}")
        print(f"    Tensions:")
        for t in proto.tensions:
            print(f"      - {t.type.value}: {t.description} (severity: {t.severity:.2f})")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import asyncio
    import sys
    sys.path.insert(0, '/app')

    from services.neo4j_service import Neo4jService

    async def main():
        neo4j = Neo4jService()
        await neo4j.connect()

        await preview_event_inquiries(neo4j, limit=15)

        await neo4j.close()

    asyncio.run(main())
