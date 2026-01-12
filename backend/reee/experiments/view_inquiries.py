"""
View-Based Proto-Inquiry Generator
===================================

Generates proto-inquiries from IncidentEventView using proper abstractions:
- Uses Surface domain model with claim hydration
- Uses claim_similarities weights from weaver
- Detects tensions from actual claim content (not just event titles)

Tension Detection (from MetaClaim types):
1. SINGLE_SOURCE: Surfaces with |sources|=1 need corroboration
2. VALUE_CONFLICT: Claims with different extracted values for same question
3. LOW_SIMILARITY: Claims with low weaver similarity (identity uncertain)
4. NUMERIC_VARIANCE: Claims with numeric values that differ across sources
5. QUOTE_UNVERIFIED: Quoted statements from single source

Usage:
    from reee.experiments.view_inquiries import generate_view_inquiries
    protos = await generate_view_inquiries(neo4j, db_pool)
"""

import re
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Set, Optional, Any, Tuple
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class TensionType(Enum):
    """Types of epistemic tension that trigger inquiries."""
    SINGLE_SOURCE = "single_source"       # Surface with single attestation
    VALUE_CONFLICT = "value_conflict"     # Conflicting values in claims
    LOW_SIMILARITY = "low_similarity"     # Weak identity binding
    NUMERIC_VARIANCE = "numeric_variance" # Varying counts (e.g., casualties)
    QUOTE_UNVERIFIED = "quote_unverified" # Quoted statement from single source
    WEAK_BINDING = "weak_binding"         # Event bound mostly by quarantine


class InquirySchema(Enum):
    """Schema types for inquiry answers."""
    MONOTONE_COUNT = "monotone_count"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    QUOTE_AUTHENTICITY = "quote_authenticity"
    OPEN_ENDED = "open_ended"


@dataclass
class ClaimEvidence:
    """Evidence from a claim for tension detection."""
    claim_id: str
    text: str
    source: str
    similarity: float
    extracted_numbers: List[int]
    has_quotes: bool
    event_time: Optional[datetime]


@dataclass
class Tension:
    """A detected tension in an incident."""
    type: TensionType
    severity: float  # 0-1, higher = more urgent
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ViewProtoInquiry:
    """Proto-inquiry generated from a View."""
    id: str = field(default_factory=lambda: f"vpi_{uuid.uuid4().hex[:12]}")

    # Source incident
    incident_id: str = ""
    incident_title: str = ""

    # Question
    question_text: str = ""
    schema_type: InquirySchema = InquirySchema.OPEN_ENDED

    # Tensions that triggered this
    tensions: List[Tension] = field(default_factory=list)

    # Scope
    surface_ids: Set[str] = field(default_factory=set)  # For event lookup
    surface_count: int = 0
    claim_count: int = 0
    source_count: int = 0
    entities: Set[str] = field(default_factory=set)

    # Claim evidence (sample)
    sample_claims: List[str] = field(default_factory=list)

    # Priority
    priority_score: float = 0.0

    # Provenance
    view_scale: str = "incident"
    params_version: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "incident_id": self.incident_id,
            "incident_title": self.incident_title,
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
            "surface_count": self.surface_count,
            "claim_count": self.claim_count,
            "source_count": self.source_count,
            "entities": list(self.entities),
            "sample_claims": self.sample_claims[:3],
            "priority_score": self.priority_score,
            "view_scale": self.view_scale,
        }


# =============================================================================
# CLAIM ANALYSIS
# =============================================================================

# Patterns for numeric extraction
NUMERIC_PATTERNS = [
    (r'(\d+)\s*(?:dead|killed|died|deaths|fatalities)', 'death_count'),
    (r'(\d+)\s*(?:injured|wounded|hurt)', 'injury_count'),
    (r'(\d+)\s*(?:missing)', 'missing_count'),
    (r'(\d+)\s*(?:arrested|detained)', 'arrest_count'),
    (r'at least (\d+)', 'at_least'),
    (r'more than (\d+)', 'more_than'),
    (r'up to (\d+)', 'up_to'),
]

# Quote patterns
QUOTE_PATTERNS = [
    r'"([^"]{20,})"',  # Quoted text > 20 chars
    r'"([^"]{20,})"',  # Smart quotes
    r'said[:,]?\s*"([^"]+)"',
    r'stated[:,]?\s*"([^"]+)"',
]


def analyze_claim(claim_data: Dict) -> ClaimEvidence:
    """Extract evidence from a claim for tension detection."""
    text = claim_data.get('text', '') or ''

    # Extract numbers
    numbers = []
    for pattern, _ in NUMERIC_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        numbers.extend(int(m) for m in matches if m.isdigit())

    # Check for quotes
    has_quotes = any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in QUOTE_PATTERNS
    )

    return ClaimEvidence(
        claim_id=claim_data.get('id', ''),
        text=text,
        source=claim_data.get('source', ''),
        similarity=claim_data.get('similarity', 1.0),
        extracted_numbers=numbers,
        has_quotes=has_quotes,
        event_time=claim_data.get('event_time'),
    )


# =============================================================================
# TENSION DETECTION
# =============================================================================

def detect_tensions_from_claims(
    claims: List[ClaimEvidence],
    surface_sources: Set[str],
    membership_breakdown: Dict[str, int],
) -> List[Tension]:
    """Detect epistemic tensions from actual claim content."""
    tensions = []

    # 1. SINGLE_SOURCE tension - surface level
    if len(surface_sources) == 1:
        tensions.append(Tension(
            type=TensionType.SINGLE_SOURCE,
            severity=0.7,
            description="Only one source attests to this claim bundle",
            evidence={"source": list(surface_sources)[0]}
        ))

    # 2. LOW_SIMILARITY tension - claims with weak binding
    low_sim_claims = [c for c in claims if c.similarity < 0.8]
    if low_sim_claims:
        avg_sim = sum(c.similarity for c in low_sim_claims) / len(low_sim_claims)
        if avg_sim < 0.75:
            tensions.append(Tension(
                type=TensionType.LOW_SIMILARITY,
                severity=min(1.0, 1.0 - avg_sim),
                description=f"{len(low_sim_claims)} claims have weak identity binding (avg sim: {avg_sim:.2f})",
                evidence={
                    "low_sim_count": len(low_sim_claims),
                    "avg_similarity": avg_sim,
                    "samples": [c.claim_id for c in low_sim_claims[:3]],
                }
            ))

    # 3. NUMERIC_VARIANCE tension - different numbers across claims
    all_numbers = []
    for claim in claims:
        all_numbers.extend(claim.extracted_numbers)

    if len(all_numbers) >= 2:
        unique_numbers = set(all_numbers)
        if len(unique_numbers) > 1:
            min_num, max_num = min(unique_numbers), max(unique_numbers)
            # Significant if variance is > 20% of max
            if (max_num - min_num) / max(max_num, 1) > 0.2:
                tensions.append(Tension(
                    type=TensionType.NUMERIC_VARIANCE,
                    severity=0.9,  # High - numeric discrepancies matter
                    description=f"Numeric values vary: {min_num} to {max_num}",
                    evidence={
                        "values": sorted(unique_numbers),
                        "min": min_num,
                        "max": max_num,
                    }
                ))

    # 4. QUOTE_UNVERIFIED tension - single-source quotes
    quote_claims = [c for c in claims if c.has_quotes]
    if quote_claims:
        # Check if quote is from single source
        quote_sources = {c.source for c in quote_claims}
        if len(quote_sources) == 1:
            tensions.append(Tension(
                type=TensionType.QUOTE_UNVERIFIED,
                severity=0.5,
                description="Quoted statement from single source needs verification",
                evidence={
                    "source": list(quote_sources)[0],
                    "claim_count": len(quote_claims),
                }
            ))

    # 5. WEAK_BINDING tension - event bound mostly by quarantine
    quarantine_count = membership_breakdown.get('quarantine', 0)
    core_count = membership_breakdown.get('core', 0)
    total = sum(membership_breakdown.values())

    if total > 2 and quarantine_count > core_count:
        tensions.append(Tension(
            type=TensionType.WEAK_BINDING,
            severity=0.6,
            description=f"Incident binding is weak ({quarantine_count}/{total} surfaces are quarantine)",
            evidence={
                "membership": membership_breakdown,
            }
        ))

    return tensions


def generate_question(
    incident_title: str,
    claims: List[ClaimEvidence],
    tensions: List[Tension],
) -> Tuple[str, InquirySchema]:
    """Generate question text and schema from tensions and claim content."""

    # Priority: numeric_variance > quote > low_similarity > single_source
    tension_types = {t.type for t in tensions}

    if TensionType.NUMERIC_VARIANCE in tension_types:
        # Find the variance tension
        var_tension = next(t for t in tensions if t.type == TensionType.NUMERIC_VARIANCE)
        values = var_tension.evidence.get('values', [])

        # Check what kind of count (death, injury, etc.)
        sample_text = ' '.join(c.text[:200] for c in claims[:3]).lower()

        if any(w in sample_text for w in ('dead', 'killed', 'died', 'death', 'fatalities')):
            question = f"How many people died? Reports vary from {min(values)} to {max(values)}."
        elif any(w in sample_text for w in ('injured', 'wounded', 'hurt')):
            question = f"How many people were injured? Reports vary from {min(values)} to {max(values)}."
        elif any(w in sample_text for w in ('missing',)):
            question = f"How many people are missing? Reports vary from {min(values)} to {max(values)}."
        elif any(w in sample_text for w in ('arrested', 'detained')):
            question = f"How many were arrested? Reports vary from {min(values)} to {max(values)}."
        else:
            question = f"What is the correct count? Reports show {min(values)} to {max(values)}."

        return question, InquirySchema.MONOTONE_COUNT

    if TensionType.QUOTE_UNVERIFIED in tension_types:
        # Extract a quote sample from claims
        for claim in claims:
            for pattern in QUOTE_PATTERNS:
                match = re.search(pattern, claim.text)
                if match:
                    quote_sample = match.group(1)[:50] + "..."
                    return f'Can this statement be verified: "{quote_sample}"', InquirySchema.QUOTE_AUTHENTICITY
        return f"Are the quoted statements verified?", InquirySchema.QUOTE_AUTHENTICITY

    if TensionType.LOW_SIMILARITY in tension_types:
        question = f"Are these claims really about the same thing? Identity binding is weak."
        return question, InquirySchema.BOOLEAN

    if TensionType.SINGLE_SOURCE in tension_types:
        question = f"Can this be independently corroborated? Currently only one source."
        return question, InquirySchema.BOOLEAN

    if TensionType.WEAK_BINDING in tension_types:
        question = f"Is this incident coherent? Binding evidence is weak."
        return question, InquirySchema.BOOLEAN

    # Default
    return f"What are the verified facts?", InquirySchema.OPEN_ENDED


def calculate_priority(
    tensions: List[Tension],
    source_count: int,
    claim_count: int,
) -> float:
    """Calculate priority score for proto-inquiry."""
    score = 0.0

    # Base score from scale (more sources = more important story)
    score += min(source_count * 5, 25)
    score += min(claim_count * 2, 20)

    # Add tension severity with type-specific weights
    for tension in tensions:
        if tension.type == TensionType.NUMERIC_VARIANCE:
            score += 40 * tension.severity  # High priority - facts vary
        elif tension.type == TensionType.VALUE_CONFLICT:
            score += 35 * tension.severity
        elif tension.type == TensionType.SINGLE_SOURCE:
            score += 25 * tension.severity
        elif tension.type == TensionType.LOW_SIMILARITY:
            score += 20 * tension.severity
        elif tension.type == TensionType.QUOTE_UNVERIFIED:
            score += 15 * tension.severity
        else:
            score += 10 * tension.severity

    return score


# =============================================================================
# MAIN GENERATOR
# =============================================================================

async def generate_view_inquiries(
    neo4j,
    db_pool,
    min_sources: int = 2,
    min_priority: float = 25.0,
    limit: int = 50,
) -> List[ViewProtoInquiry]:
    """
    Generate proto-inquiries from IncidentEventView.

    Uses proper abstractions:
    - IncidentEventView for clustering
    - SurfaceRepository for claim hydration
    - Similarity weights from weaver

    Args:
        neo4j: Neo4jService instance
        db_pool: PostgreSQL connection pool
        min_sources: Minimum source count for inclusion
        min_priority: Minimum priority score to include
        limit: Max inquiries to generate

    Returns:
        List of ViewProtoInquiry objects
    """
    from repositories.surface_repository import SurfaceRepository
    from reee.views import build_incident_events, IncidentViewParams
    from reee.types import Surface as REEESurface, MembershipLevel

    surface_repo = SurfaceRepository(db_pool, neo4j)

    # 1. Load surfaces from Neo4j for View computation
    logger.info("Loading surfaces for view computation...")
    raw_surfaces = await neo4j._execute_read('''
        MATCH (s:Surface)
        RETURN s.id as id, s.claim_ids as claim_ids, s.sources as sources,
               s.entities as entities, s.anchor_entities as anchors,
               s.time_start as time_start, s.time_end as time_end,
               s.support as support
    ''')

    # Convert to REEE Surface format for View
    from dateutil.parser import parse as parse_date

    surfaces_dict = {}
    for r in raw_surfaces:
        sid = r['id']
        # Parse time - may be string or datetime
        time_start = r.get('time_start')
        time_end = r.get('time_end')

        if isinstance(time_start, str):
            try:
                time_start = parse_date(time_start)
            except (ValueError, TypeError):
                time_start = None
        if isinstance(time_end, str):
            try:
                time_end = parse_date(time_end)
            except (ValueError, TypeError):
                time_end = None

        time_window = (time_start, time_end) if time_start or time_end else (None, None)

        surfaces_dict[sid] = REEESurface(
            id=sid,
            claim_ids=set(r.get('claim_ids') or []),
            sources=set(r.get('sources') or []),
            entities=set(r.get('entities') or []),
            anchor_entities=set(r.get('anchors') or []),
            time_window=time_window,
            mass=r.get('support', 0) or 0,
        )

    logger.info(f"  Loaded {len(surfaces_dict)} surfaces")

    # 2. Build incident view
    params = IncidentViewParams(
        temporal_window_days=14,
        min_signals=2,
        require_discriminative_anchor=True,
    )
    view_result = build_incident_events(surfaces_dict, params)

    logger.info(f"  Built view: {view_result.total_incidents} incidents, {view_result.total_isolated} isolated")

    # 3. Process each multi-surface incident
    protos = []

    for event_id, event in view_result.incidents.items():
        # Skip small incidents
        if event.total_sources < min_sources:
            continue

        # Collect all claims from this incident's surfaces
        all_claims: List[ClaimEvidence] = []
        all_sources: Set[str] = set()
        sample_claim_texts = []

        # Membership breakdown for tension detection
        membership_breakdown = defaultdict(int)
        for sid, membership in event.memberships.items():
            membership_breakdown[membership.level.value] += 1

        # Hydrate claims for each surface
        for surface_id in event.surface_ids:
            try:
                surface = await surface_repo.get_with_claims(surface_id)
                if surface and surface.claims:
                    all_sources.update(surface.sources)

                    for claim in surface.claims:
                        claim_data = {
                            'id': claim.id,
                            'text': claim.text,
                            'source': claim.source if hasattr(claim, 'source') else '',
                            'similarity': surface.claim_similarities.get(claim.id, 1.0),
                            'event_time': claim.event_time if hasattr(claim, 'event_time') else None,
                        }
                        evidence = analyze_claim(claim_data)
                        all_claims.append(evidence)

                        if len(sample_claim_texts) < 5:
                            sample_claim_texts.append(claim.text[:150])
            except Exception as e:
                logger.warning(f"Failed to hydrate surface {surface_id}: {e}")
                continue

        if not all_claims:
            continue

        # 4. Detect tensions from actual claim content
        tensions = detect_tensions_from_claims(
            all_claims,
            all_sources,
            dict(membership_breakdown),
        )

        if not tensions:
            continue  # No tensions = no inquiry needed

        # 5. Generate question
        # Build title from anchor entities or first claim
        incident_title = f"Incident with {len(all_claims)} claims"
        if event.anchor_entities:
            anchors = list(event.anchor_entities)[:3]
            incident_title = f"Incident involving {', '.join(anchors)}"

        question, schema = generate_question(incident_title, all_claims, tensions)

        # 6. Calculate priority
        priority = calculate_priority(tensions, len(all_sources), len(all_claims))

        if priority < min_priority:
            continue

        # 7. Create proto-inquiry
        proto = ViewProtoInquiry(
            incident_id=event_id,
            incident_title=incident_title,
            question_text=question,
            schema_type=schema,
            tensions=tensions,
            surface_ids=event.surface_ids,  # For event lookup
            surface_count=len(event.surface_ids),
            claim_count=len(all_claims),
            source_count=len(all_sources),
            entities=event.entities,
            sample_claims=sample_claim_texts,
            priority_score=priority,
            view_scale="incident",
            params_version=params.min_signals,  # Track params
        )

        protos.append(proto)

    # Sort by priority
    protos.sort(key=lambda p: p.priority_score, reverse=True)
    return protos[:limit]


async def preview_view_inquiries(neo4j, db_pool, limit: int = 15) -> None:
    """Preview proto-inquiries without persisting."""
    protos = await generate_view_inquiries(neo4j, db_pool, limit=limit)

    print("=" * 70)
    print("VIEW-BASED PROTO-INQUIRIES")
    print("=" * 70)

    for i, proto in enumerate(protos, 1):
        print(f"\n[{i}] Priority: {proto.priority_score:.1f}")
        print(f"    Incident: {proto.incident_id}")
        print(f"    Stats: {proto.claim_count} claims, {proto.surface_count} surfaces, {proto.source_count} sources")
        print(f"    Question: {proto.question_text}")
        print(f"    Schema: {proto.schema_type.value}")
        print(f"    Tensions:")
        for t in proto.tensions:
            print(f"      - {t.type.value}: {t.description} (severity: {t.severity:.2f})")
        if proto.sample_claims:
            print(f"    Sample claim: \"{proto.sample_claims[0][:80]}...\"")


# =============================================================================
# EVENT LOOKUP HELPER
# =============================================================================

async def find_event_for_surfaces(neo4j, surface_ids: Set[str]) -> Optional[str]:
    """
    Find the Event ID that best matches a set of surfaces.

    Looks up Events in Neo4j that share surfaces with the incident.
    Returns the Event with the most overlap.
    """
    if not surface_ids:
        return None

    # Query for Events that contain any of these surfaces
    results = await neo4j._execute_read('''
        MATCH (s:Surface)-[:BELONGS_TO]->(e:Event)
        WHERE s.id IN $surface_ids
        WITH e, count(s) as overlap
        RETURN e.id as event_id, overlap
        ORDER BY overlap DESC
        LIMIT 1
    ''', {'surface_ids': list(surface_ids)})

    if results:
        return results[0]['event_id']
    return None


# =============================================================================
# SEEDING
# =============================================================================

async def seed_view_inquiries(
    neo4j,
    db_pool,
    min_sources: int = 2,
    min_priority: float = 25.0,
    limit: int = 20,
    clear_existing: bool = False,
) -> Dict[str, Any]:
    """
    Seed proto-inquiries to the webapp database.

    Args:
        neo4j: Neo4jService instance
        db_pool: PostgreSQL connection pool
        min_sources: Minimum source count for inclusion
        min_priority: Minimum priority score to include
        limit: Max inquiries to seed
        clear_existing: If True, clear existing inquiries first

    Returns:
        Summary of seeding operation
    """
    import json

    # Generate inquiries
    protos = await generate_view_inquiries(
        neo4j, db_pool,
        min_sources=min_sources,
        min_priority=min_priority,
        limit=limit,
    )

    logger.info(f"Generated {len(protos)} proto-inquiries for seeding")

    async with db_pool.acquire() as conn:
        # Optionally clear existing
        if clear_existing:
            await conn.execute("DELETE FROM inquiries WHERE created_by IS NULL")
            logger.info("Cleared existing system-generated inquiries")

        seeded = 0
        errors = []

        for proto in protos:
            try:
                inquiry_id = f"inq_{proto.id}"

                # Check if already exists
                exists = await conn.fetchval(
                    "SELECT 1 FROM inquiries WHERE id = $1",
                    inquiry_id
                )
                if exists:
                    logger.debug(f"Inquiry {inquiry_id} already exists, skipping")
                    continue

                # Build description
                tension_descs = [t.description for t in proto.tensions]
                description = f"System-detected tensions: {'; '.join(tension_descs)}"

                if proto.sample_claims:
                    description += f"\n\nSample claim: \"{proto.sample_claims[0][:200]}...\""

                # Schema config
                schema_config = {
                    "schema_type": proto.schema_type.value,
                    "tension_types": [t.type.value for t in proto.tensions],
                }
                if proto.schema_type == InquirySchema.MONOTONE_COUNT:
                    # Extract numbers from tensions
                    for t in proto.tensions:
                        if t.type == TensionType.NUMERIC_VARIANCE:
                            schema_config["min"] = t.evidence.get("min", 0)
                            schema_config["max"] = t.evidence.get("max", 500)

                await conn.execute("""
                    INSERT INTO inquiries (
                        id, title, description, status, rigor_level,
                        schema_type, schema_config, scope_entities,
                        source_event, created_at, updated_at
                    )
                    VALUES ($1, $2, $3, 'open', 'B', $4, $5, $6, $7, now(), now())
                """,
                    inquiry_id,
                    proto.question_text,
                    description,
                    proto.schema_type.value,
                    json.dumps(schema_config),
                    list(proto.entities)[:20],
                    proto.incident_id,  # Link to source event
                )

                seeded += 1
                logger.info(f"Seeded inquiry {inquiry_id}: {proto.question_text[:50]}...")

            except Exception as e:
                logger.error(f"Error seeding inquiry {proto.id}: {e}")
                errors.append({"proto_id": proto.id, "error": str(e)})

    result = {
        "generated": len(protos),
        "seeded": seeded,
        "errors": len(errors),
        "error_details": errors[:5],  # First 5 errors
    }

    logger.info(f"Seeding complete: {seeded}/{len(protos)} seeded, {len(errors)} errors")
    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import asyncio
    import os
    import sys
    sys.path.insert(0, '/app')

    import asyncpg
    from services.neo4j_service import Neo4jService

    async def main():
        db_pool = await asyncpg.create_pool(
            host=os.getenv('POSTGRES_HOST', 'db'),
            port=5432,
            database=os.getenv('POSTGRES_DB', 'phi_here'),
            user=os.getenv('POSTGRES_USER', 'phi_user'),
            password=os.getenv('POSTGRES_PASSWORD', 'phi_password_dev'),
        )

        neo4j = Neo4jService()
        await neo4j.connect()

        await preview_view_inquiries(neo4j, db_pool, limit=15)

        await neo4j.close()
        await db_pool.close()

    asyncio.run(main())
