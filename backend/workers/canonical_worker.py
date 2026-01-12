"""
Canonical Worker - Event & Entity Narrative Builder
=====================================================

Continuously builds canonical narratives for events and entities
from principled engine events. Runs as a background worker.

Architecture:
    PrincipledSurfaceBuilder (Claims → Surfaces via motif clustering)
        ↓
    PrincipledEventBuilder (Surfaces → Events via context compatibility)
        ↓
    CanonicalWorker (Events → Titles/Descriptions via LLM)
        ↓
    API (serves cached canonical data)

IMPORTANT: This worker CONSUMES events from the principled engine.
It does NOT do its own event formation - that's handled by
rebuild_topology.py or the canonical_worker pipeline.

Output:
- Event nodes enriched with title, description, primary_entity
- Entity nodes enriched with narrative, source_count, surface_count
- Uses :Event {kind: 'canonical'} to distinguish from raw events
"""

import asyncio
import hashlib
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Set, List, Optional
from dataclasses import dataclass, field

import asyncpg
from openai import AsyncOpenAI

from services.neo4j_service import Neo4jService
from reee.types import Surface, Event, EntityLens
from reee.builders import (
    PrincipledSurfaceBuilder,
    PrincipledEventBuilder,
    MotifConfig,
)
from reee.builders.story_builder import StoryBuilder, StoryBuilderResult
from workers.claim_loader import load_claims_with_embeddings
from models.domain.case import Case as DomainCase
from repositories.case_repository import CaseRepository

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_stable_event_id(entities: Set[str], surface_ids: Set[str]) -> str:
    """Generate stable event ID from deterministic signature.

    ID is derived from sorted entity names + sorted surface IDs.
    This ensures the same event gets the same ID across rebuilds.
    """
    # Sort for determinism
    sorted_entities = sorted(entities)[:5]  # Top 5 entities
    sorted_surfaces = sorted(surface_ids)

    signature = f"evt:{','.join(sorted_entities)}:{','.join(sorted_surfaces)}"
    hash_hex = hashlib.sha256(signature.encode()).hexdigest()[:12]
    return f"event_{hash_hex}"


@dataclass
class CanonicalEvent:
    """API-ready event entity."""
    id: str
    signature: str  # Stable signature for deduplication
    title: str
    description: str
    primary_entity: str
    secondary_entities: List[str] = field(default_factory=list)
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None
    source_count: int = 0
    surface_count: int = 0
    claim_count: int = 0
    surface_ids: Set[str] = field(default_factory=set)


@dataclass
class CanonicalEntity:
    """API-ready entity with narrative."""
    id: str
    canonical_name: str
    entity_type: str
    narrative: str  # LLM-generated summary
    source_count: int = 0
    surface_count: int = 0
    claim_count: int = 0
    related_events: List[str] = field(default_factory=list)
    last_active: Optional[datetime] = None


@dataclass
class CanonicalCase:
    """L4 Case: stable higher-order object grouping related incidents.

    A Case emerges only when there's repeatable evidence that multiple
    incidents belong to the same storyline.

    Requirements:
    - At least 2 incidents
    - At least 1 non-trivial binding constraint between incidents
    """
    id: str
    incident_ids: Set[str] = field(default_factory=set)
    title: str = ""
    description: str = ""
    primary_entities: List[str] = field(default_factory=list)
    case_type: str = "general"  # breaking, developing, ongoing
    binding_evidence: List[str] = field(default_factory=list)  # Why these incidents belong together
    surface_count: int = 0
    source_count: int = 0
    claim_count: int = 0
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None


@dataclass
class IncidentContext:
    """Context for an L3 Incident used in case formation."""
    id: str
    anchor_entities: Set[str] = field(default_factory=set)
    companion_entities: Set[str] = field(default_factory=set)
    surface_count: int = 0
    claim_count: int = 0
    source_count: int = 0
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None
    sample_claims: List[str] = field(default_factory=list)  # For LLM context


class CanonicalWorker:
    """
    Background worker that builds canonical events and entities.

    Runs periodic full rebuilds to keep canonical layer fresh.
    Uses LLM to generate narrative titles and descriptions.
    """

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        neo4j: Neo4jService,
        rebuild_interval: int = 300,  # 5 minutes
        use_llm: bool = True,
    ):
        self.db_pool = db_pool
        self.neo4j = neo4j
        self.rebuild_interval = rebuild_interval
        self.use_llm = use_llm

        # OpenAI client for narrative generation
        self.openai = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # Claim texts cache (loaded during rebuild)
        self.claim_texts: Dict[str, str] = {}

        # Stats
        self.last_rebuild: Optional[datetime] = None
        self.events_count = 0
        self.entities_count = 0
        self.cases_count = 0

    async def start(self, mode: str = "rebuild"):
        """Start the canonical worker.

        Modes:
        - "rebuild": Periodic full rebuild (legacy)
        - "enrich": Queue-based enrichment (new - listens to weaver:enrichment)
        """
        if mode == "enrich":
            await self._start_enrichment_mode()
        else:
            await self._start_rebuild_mode()

    async def _start_rebuild_mode(self):
        """Legacy periodic rebuild mode."""
        logger.info(f"🏗️  CanonicalWorker started in REBUILD mode (every {self.rebuild_interval}s)")

        while True:
            try:
                await self.rebuild_canonical_layer()
                self.last_rebuild = datetime.utcnow()

                logger.info(
                    f"✅ Canonical rebuild complete: "
                    f"{self.events_count} events, {self.entities_count} entities, "
                    f"{self.cases_count} cases"
                )

            except Exception as e:
                logger.error(f"❌ Canonical rebuild failed: {e}", exc_info=True)

            await asyncio.sleep(self.rebuild_interval)

    async def _start_enrichment_mode(self):
        """Subscribe to weaver:events channel for enrichment (shared with viz)."""
        import redis.asyncio as aioredis
        import json

        redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
        redis_client = aioredis.from_url(redis_url)
        pubsub = redis_client.pubsub()

        # Subscribe to same channel as viz
        await pubsub.subscribe("weaver:events")
        logger.info("🎯 CanonicalWorker started in ENRICHMENT mode (subscribed to weaver:events)")

        async for message in pubsub.listen():
            try:
                if message['type'] != 'message':
                    continue

                event = json.loads(message['data'])
                event_type = event.get("type", "unknown")

                # Only process case events
                if event_type in ("case_formed", "case_updated"):
                    case_id = event.get("case_id")
                    logger.info(f"📥 Case event: {event_type} for {case_id}")
                    await self.enrich_case(event)

            except Exception as e:
                logger.error(f"❌ Enrichment error: {e}", exc_info=True)

    async def enrich_case(self, job: dict):
        """Enrich a case with LLM-generated title and description.

        Called when weaver emits a case_formed or case_updated event.
        """
        case_id = job.get("case_id")
        kernel_sig = job.get("kernel_signature")
        incident_ids = job.get("incident_ids", [])
        core_entities = job.get("core_entities", [])

        if not case_id or not incident_ids:
            logger.warning(f"Invalid enrichment job: missing case_id or incident_ids")
            return

        # Load sample claims from incidents for narrative generation
        sample_claims = await self._load_sample_claims_for_case(incident_ids)

        # Generate title and description
        if self.use_llm and sample_claims:
            title, description = await self._generate_case_narrative(
                sample_claims=sample_claims[:10],
                primary_entities=core_entities[:5],
                incident_count=len(incident_ids),
                source_count=len(sample_claims),  # Approximate
            )
        else:
            title = f"{core_entities[0]} Story" if core_entities else "Developing Story"
            description = f"Coverage across {len(incident_ids)} related developments."

        # Update case in Neo4j
        await self.neo4j._execute_write("""
            MATCH (c:Case {kernel_signature: $sig})
            SET c.title = $title,
                c.description = $description,
                c.enriched_at = datetime()
        """, {
            'sig': kernel_sig,
            'title': title,
            'description': description,
        })

        logger.info(f"✅ Enriched case {kernel_sig}: {title[:50]}...")

    async def _load_sample_claims_for_case(self, incident_ids: List[str]) -> List[str]:
        """Load sample claim texts for a case's incidents."""
        if not incident_ids:
            return []

        rows = await self.neo4j._execute_read("""
            UNWIND $incident_ids as iid
            MATCH (i:Incident {id: iid})-[:CONTAINS]->(s:Surface)-[:CONTAINS]->(c:Claim)
            RETURN c.text as text
            LIMIT 20
        """, {'incident_ids': incident_ids})

        return [r['text'] for r in rows if r.get('text')]

    async def rebuild_canonical_layer(self):
        """Full rebuild of canonical events and entities."""
        logger.info("🔄 Starting canonical rebuild...")

        # 1. Load all surfaces with metadata
        surfaces = await self._load_surfaces()
        logger.info(f"  Loaded {len(surfaces)} surfaces")

        # 1b. Load claim texts for narrative generation
        if self.use_llm:
            await self._load_claim_texts()
            logger.info(f"  Loaded {len(self.claim_texts)} claim texts")

        if not surfaces:
            return

        # 2. Load entity names
        entity_names = await self._load_entity_names()
        logger.info(f"  Loaded {len(entity_names)} entity names")

        # 3. Build canonical events
        events = await self._build_canonical_events(surfaces, entity_names)
        logger.info(f"  Built {len(events)} canonical events")

        # 4. Persist events
        await self._persist_events(events)
        self.events_count = len(events)

        # 5. Build and persist canonical entities
        entities = await self._build_canonical_entities(surfaces, entity_names, events)
        await self._persist_entities(entities)
        self.entities_count = len(entities)

        # 6. L4 Cases - now handled by weaver's CaseBuilder
        # Legacy rebuild mode: just count existing cases, don't rebuild
        case_count = await self.neo4j._execute_read("""
            MATCH (c:Case)
            RETURN count(c) as cnt
        """)
        self.cases_count = case_count[0]['cnt'] if case_count else 0
        logger.info(f"  L4 cases (from weaver): {self.cases_count}")

    async def _load_surfaces(self) -> Dict[str, Surface]:
        """Load all surfaces from Neo4j."""
        rows = await self.neo4j._execute_read('''
            MATCH (s:Surface)
            OPTIONAL MATCH (s)-[:CONTAINS]->(c:Claim)
            OPTIONAL MATCH (s)-[:HAS_ENTITY]->(e:Entity)
            WITH s,
                 collect(DISTINCT c.id) as claim_ids,
                 collect(DISTINCT e.canonical_name) as entities
            RETURN s.id as id,
                   s.sources as sources,
                   entities,
                   s.time_start as time_start,
                   s.time_end as time_end,
                   claim_ids
        ''')

        surfaces = {}
        for row in rows:
            time_start = self._parse_time(row.get('time_start'))
            time_end = self._parse_time(row.get('time_end'))

            # Use entities from HAS_ENTITY relationships
            entities = set(e for e in (row['entities'] or []) if e)

            surfaces[row['id']] = Surface(
                id=row['id'],
                claim_ids=set(row['claim_ids'] or []),
                sources=set(row['sources'] or []),
                anchor_entities=entities,  # Use entities from relationships
                entities=entities,
                time_window=(time_start, time_end),
            )

        return surfaces

    async def _load_entity_names(self) -> Dict[str, str]:
        """Load entity ID -> name mapping."""
        rows = await self.neo4j._execute_read('''
            MATCH (e:Entity)
            RETURN e.id as id, e.canonical_name as name, e.type as type
        ''')
        return {r['id']: r['name'] for r in rows}

    async def _load_claim_texts(self):
        """Load claim texts for narrative generation."""
        rows = await self.neo4j._execute_read('''
            MATCH (c:Claim)
            RETURN c.id as id, c.text as text
        ''')
        self.claim_texts = {r['id']: r['text'] for r in rows if r['text']}

    async def _generate_event_narrative(
        self,
        claim_ids: Set[str],
        primary_entity: str,
        secondary_entities: List[str],
        source_count: int,
    ) -> tuple:
        """Generate event title and description using LLM.

        Signals used:
        - Claim texts: The actual reported facts
        - Entity names: Who/what the event is about
        - Source count: Credibility/importance indicator

        Returns:
            (title, description) tuple
        """
        # Gather claim texts (up to 15 for context)
        claim_samples = []
        for cid in list(claim_ids)[:15]:
            if cid in self.claim_texts:
                claim_samples.append(self.claim_texts[cid])

        if not claim_samples:
            # Fallback if no claims available
            return (
                f"{primary_entity}: Developing Story",
                f"Coverage from {source_count} sources about {primary_entity}."
            )

        # Build prompt for LLM
        claims_text = "\n".join(f"- {c[:300]}" for c in claim_samples)
        entities = [primary_entity] + secondary_entities[:3]
        entities_text = ", ".join(entities)

        prompt = f"""Based on these news claims about {entities_text}, generate a concise event title and description.

Claims:
{claims_text}

Requirements:
- Title: 5-12 words, news headline style, factual (no sensationalism)
- Description: 1-2 sentences summarizing what happened, neutral tone

Respond in JSON format:
{{"title": "...", "description": "..."}}"""

        try:
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
                response_format={"type": "json_object"},
            )

            import json
            result = json.loads(response.choices[0].message.content)
            return (
                result.get("title", f"{primary_entity}: Story"),
                result.get("description", f"Coverage from {source_count} sources.")
            )

        except Exception as e:
            logger.warning(f"LLM narrative failed: {e}")
            return (
                f"{primary_entity}: Developing Story",
                f"Coverage from {source_count} sources about {primary_entity}."
            )

    async def _generate_entity_narrative(
        self,
        entity_name: str,
        claim_ids: Set[str],
        source_count: int,
        surface_count: int,
    ) -> str:
        """Generate entity narrative using LLM.

        Signals used:
        - Claim texts: What's being said about this entity
        - Source/surface counts: Activity level

        Returns:
            Narrative string
        """
        # Gather claim texts (up to 10)
        claim_samples = []
        for cid in list(claim_ids)[:10]:
            if cid in self.claim_texts:
                claim_samples.append(self.claim_texts[cid])

        if not claim_samples:
            return f"Mentioned in {source_count} sources across {surface_count} topics."

        claims_text = "\n".join(f"- {c[:250]}" for c in claim_samples)

        prompt = f"""Based on these news claims about "{entity_name}", write a brief narrative summary.

Claims:
{claims_text}

Requirements:
- 2-3 sentences maximum
- Describe the entity's current news relevance
- Neutral, factual tone

Respond with just the narrative text, no JSON."""

        try:
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"LLM entity narrative failed: {e}")
            return f"Mentioned in {source_count} sources across {surface_count} topics."

    async def _build_canonical_events(
        self,
        surfaces: Dict[str, Surface],
        entity_names: Dict[str, str],
    ) -> List[CanonicalEvent]:
        """Build canonical events by consuming principled engine events.

        This method:
        1. Loads existing principled events from Neo4j (created by rebuild_topology)
        2. OR builds them fresh using PrincipledSurfaceBuilder + PrincipledEventBuilder
        3. Enriches with LLM-generated titles and descriptions
        """
        # Check if we have principled events already
        existing_events = await self._load_principled_events()

        if existing_events:
            logger.info(f"  Found {len(existing_events)} principled events in Neo4j")
            principled_events = existing_events
        else:
            # Build fresh using principled engine
            logger.info("  No principled events found, building fresh...")
            principled_events = await self._build_principled_events_fresh(surfaces)

        # Convert to canonical events
        canonical_events = []
        for event in principled_events.values():
            canonical = self._convert_to_canonical(event, surfaces, entity_names)
            canonical_events.append(canonical)

        # Sort by source count
        canonical_events.sort(key=lambda e: -e.source_count)

        # Generate LLM narratives for top events
        if self.use_llm:
            top_events = canonical_events[:25]
            logger.info(f"  Generating LLM narratives for {len(top_events)} events...")

            for event in top_events:
                event_claim_ids: Set[str] = set()
                for sid in event.surface_ids:
                    if sid in surfaces:
                        event_claim_ids.update(surfaces[sid].claim_ids)

                title, description = await self._generate_event_narrative(
                    claim_ids=event_claim_ids,
                    primary_entity=event.primary_entity,
                    secondary_entities=event.secondary_entities,
                    source_count=event.source_count,
                )
                event.title = title
                event.description = description

        return canonical_events

    async def _load_principled_events(self) -> Dict[str, Event]:
        """Load existing principled events from Neo4j."""
        rows = await self.neo4j._execute_read('''
            MATCH (e:Event)
            WHERE e.formation_method IS NULL OR e.formation_method <> 'canonical'
            OPTIONAL MATCH (e)-[r:INCLUDES]->(s:Surface)
            WITH e, collect({id: s.id, level: r.level, score: r.score}) as surfaces
            RETURN e.id as id,
                   e.surface_count as surface_count,
                   e.core_count as core_count,
                   surfaces
        ''')

        events = {}
        for row in rows:
            event = Event(
                id=row['id'],
                surface_ids={s['id'] for s in row['surfaces'] if s['id']},
            )
            events[row['id']] = event

        return events

    async def _build_principled_events_fresh(
        self,
        surfaces: Dict[str, Surface],
    ) -> Dict[str, Event]:
        """Build principled events from scratch using the engine."""
        # Load claims using data layer
        from repositories.claim_repository import ClaimRepository
        claim_repo = ClaimRepository(self.db_pool, self.neo4j)
        claims = await load_claims_with_embeddings(claim_repo, self.neo4j)
        claims_with_entities = [c for c in claims if len(c.entities) >= 2]

        if not claims_with_entities:
            return {}

        # Build surfaces using motif clustering
        config = MotifConfig(min_k=2, min_support=2, graded=True)
        surface_builder = PrincipledSurfaceBuilder(config=config)
        surface_result = await surface_builder.build_from_claims(claims_with_entities)

        logger.info(f"    Built {len(surface_result.surfaces)} surfaces")

        # Build events using context compatibility
        event_builder = PrincipledEventBuilder(min_companions=1, overlap_threshold=0.15)
        event_result = await event_builder.build_from_surfaces(
            surface_result.surfaces,
            surface_ledger=surface_result.ledger
        )

        logger.info(f"    Built {len(event_result.events)} events")

        return event_result.events

    def _convert_to_canonical(
        self,
        event: Event,
        surfaces: Dict[str, Surface],
        entity_names: Dict[str, str],
    ) -> CanonicalEvent:
        """Convert principled Event to CanonicalEvent."""
        all_sources: Set[str] = set()
        all_claims: Set[str] = set()
        all_entities: Set[str] = set()
        times: List[datetime] = []

        for sid in event.surface_ids:
            s = surfaces.get(sid)
            if not s:
                continue

            all_sources.update(s.sources)
            all_claims.update(s.claim_ids)
            # Use entities (canonical names) from surface
            all_entities.update(s.entities or s.anchor_entities or set())

            if s.time_window and s.time_window[0]:
                times.append(s.time_window[0])

        # Pick primary entity (first from sorted list for determinism)
        # Entities are already canonical names from Neo4j
        sorted_entities = sorted(all_entities)
        primary = sorted_entities[0] if sorted_entities else "Unknown"
        secondary = sorted_entities[1:5]

        time_start = min(times) if times else None
        time_end = max(times) if times else None

        # Generate stable ID
        stable_id = generate_stable_event_id(all_entities, event.surface_ids)
        signature = f"{','.join(sorted_entities[:5])}:{','.join(sorted(event.surface_ids))}"

        # Placeholder title/description (LLM fills in later)
        title = f"{primary}: Story" if primary else "Developing Story"
        description = f"Coverage from {len(all_sources)} sources about {primary}."

        return CanonicalEvent(
            id=stable_id,
            signature=signature,
            title=title,
            description=description,
            primary_entity=primary,
            secondary_entities=secondary,
            time_start=time_start,
            time_end=time_end,
            source_count=len(all_sources),
            surface_count=len(event.surface_ids),
            claim_count=len(all_claims),
            surface_ids=event.surface_ids,
        )

    # NOTE: _merge_overlapping and _build_event removed - legacy event formation
    # Event formation now handled by PrincipledEventBuilder in canonical layer

    async def _build_canonical_entities(
        self,
        surfaces: Dict[str, Surface],
        entity_names: Dict[str, str],
        events: List[CanonicalEvent],
    ) -> List[CanonicalEntity]:
        """Build canonical entities from surfaces."""

        # Aggregate entity stats from surfaces
        entity_stats: Dict[str, dict] = {}

        for sid, surface in surfaces.items():
            for eid in surface.anchor_entities:
                if eid not in entity_stats:
                    entity_stats[eid] = {
                        'sources': set(),
                        'surfaces': set(),
                        'claims': set(),
                        'times': [],
                    }

                entity_stats[eid]['sources'].update(surface.sources)
                entity_stats[eid]['surfaces'].add(sid)
                entity_stats[eid]['claims'].update(surface.claim_ids)

                if surface.time_window[0]:
                    entity_stats[eid]['times'].append(surface.time_window[0])

        # Build surface -> event mapping
        surface_to_events: Dict[str, List[str]] = {}
        for event in events:
            for sid in event.surface_ids:
                if sid not in surface_to_events:
                    surface_to_events[sid] = []
                surface_to_events[sid].append(event.id)

        # Build canonical entities
        entities = []
        for eid, stats in entity_stats.items():
            name = entity_names.get(eid, eid)

            # Get related events
            related_events = set()
            for sid in stats['surfaces']:
                if sid in surface_to_events:
                    related_events.update(surface_to_events[sid])

            last_active = max(stats['times']) if stats['times'] else None

            # Placeholder narrative (LLM generated below for top entities)
            narrative = f"Mentioned in {len(stats['sources'])} sources across {len(stats['surfaces'])} topics."

            entities.append(CanonicalEntity(
                id=eid,
                canonical_name=name,
                entity_type="UNKNOWN",  # Would come from entity data
                narrative=narrative,
                source_count=len(stats['sources']),
                surface_count=len(stats['surfaces']),
                claim_count=len(stats['claims']),
                related_events=list(related_events)[:10],
                last_active=last_active,
            ))

        # Sort by source count
        entities.sort(key=lambda e: -e.source_count)

        # Generate LLM narratives for top entities
        if self.use_llm:
            top_entities = entities[:30]  # Top 30 entities get LLM narratives
            logger.info(f"  Generating LLM narratives for {len(top_entities)} entities...")

            for entity in top_entities:
                # Get claims for this entity
                entity_claim_ids = entity_stats.get(entity.id, {}).get('claims', set())

                narrative = await self._generate_entity_narrative(
                    entity_name=entity.canonical_name,
                    claim_ids=entity_claim_ids,
                    source_count=entity.source_count,
                    surface_count=entity.surface_count,
                )
                entity.narrative = narrative

        return entities

    async def _persist_events(self, events: List[CanonicalEvent]):
        """Persist canonical events by enriching existing principled events.

        Strategy: Update existing principled events with canonical data
        (title, description, primary_entity) rather than creating duplicates.
        """
        for event in events:
            # Find the principled event that matches these surfaces and enrich it
            # Match by surface_ids overlap
            await self.neo4j._execute_write('''
                MATCH (e:Event)-[:INCLUDES]->(s:Surface)
                WHERE s.id IN $surface_ids
                WITH e, count(s) as matched
                WHERE matched = $expected_count
                SET e.title = $title,
                    e.description = $description,
                    e.primary_entity = $primary_entity,
                    e.secondary_entities = $secondary_entities,
                    e.signature = $signature,
                    e.canonical_id = $canonical_id,
                    e.time_start = $time_start,
                    e.time_end = $time_end,
                    e.updated_at = datetime()
            ''', {
                'surface_ids': list(event.surface_ids),
                'expected_count': len(event.surface_ids),
                'title': event.title,
                'description': event.description,
                'primary_entity': event.primary_entity,
                'secondary_entities': event.secondary_entities,
                'signature': event.signature,
                'canonical_id': event.id,
                'time_start': event.time_start.isoformat() if event.time_start else None,
                'time_end': event.time_end.isoformat() if event.time_end else None,
            })

    async def _persist_entities(self, entities: List[CanonicalEntity]):
        """Update entities with canonical narrative data."""
        for entity in entities:
            await self.neo4j._execute_write('''
                MATCH (e:Entity {id: $id})
                SET e.narrative = $narrative,
                    e.source_count = $source_count,
                    e.surface_count = $surface_count,
                    e.claim_count = $claim_count,
                    e.related_events = $related_events,
                    e.last_active = $last_active,
                    e.canonical_updated_at = datetime()
            ''', {
                'id': entity.id,
                'narrative': entity.narrative,
                'source_count': entity.source_count,
                'surface_count': entity.surface_count,
                'claim_count': entity.claim_count,
                'related_events': entity.related_events,
                'last_active': entity.last_active.isoformat() if entity.last_active else None,
            })

    def _parse_time(self, val) -> Optional[datetime]:
        """Parse time from various formats."""
        if val is None:
            return None
        if isinstance(val, datetime):
            return val
        if isinstance(val, str):
            try:
                return datetime.fromisoformat(val.replace('Z', '+00:00'))
            except:
                return None
        return None

    # =========================================================================
    # L4 CASE FORMATION
    # =========================================================================

    async def _flag_incomplete_cases(self):
        """
        Flag cases that don't meet principled criteria (< 2 incidents).

        NOTE: This is PRESENTATION-ONLY. The canonical layer does NOT delete
        structural nodes. It only flags them for the UI to handle appropriately.

        Cases with insufficient incidents are marked with:
        - is_incomplete = true
        - needs_enrichment = true

        This allows the principled weaver to handle structural cleanup, while
        the canonical worker focuses purely on enrichment (titles, descriptions).
        """
        # Count incomplete cases
        to_flag = await self.neo4j._execute_read('''
            MATCH (c:Case)
            OPTIONAL MATCH (c)-[:CONTAINS]->(i:Incident)
            WITH c, count(i) as inc_count
            WHERE inc_count < 2
            RETURN count(c) as count
        ''')

        flag_count = to_flag[0]['count'] if to_flag else 0

        if flag_count > 0:
            # Flag cases as incomplete (DO NOT DELETE - presentation only!)
            await self.neo4j._execute_write('''
                MATCH (c:Case)
                OPTIONAL MATCH (c)-[:CONTAINS]->(i:Incident)
                WITH c, count(i) as inc_count
                WHERE inc_count < 2
                SET c.is_incomplete = true,
                    c.needs_enrichment = true
            ''')
            logger.info(f"  📋 Flagged {flag_count} incomplete cases (< 2 incidents)")

    async def _build_cases_from_incidents(self) -> List[CanonicalCase]:
        """
        Build L4 Cases by clustering L3 Incidents using StoryBuilder.

        StoryBuilder uses spine + mode + membrane approach:
        - Spine: focal entity (non-hub) that defines the story
        - Mode: temporal clustering to separate same-spine stories
        - Membrane: single gate for core vs periphery membership

        This handles star-shaped patterns where k=2 motif recurrence fails
        (rotating companions, no pair recurs, but narrative coheres via spine).
        """
        # Flag incomplete cases (presentation-only - DO NOT delete structural nodes)
        await self._flag_incomplete_cases()

        # Load incidents as Event objects with justifications
        incidents = await self._load_incidents_as_events()
        logger.info(f"  Loaded {len(incidents)} incidents for principled case building")

        if len(incidents) < 2:
            return []

        # Load entity types (legacy - temporal anchor binding is now entity-agnostic)
        entity_types = await self._load_entity_types()
        logger.info(f"  Loaded {len(entity_types)} entity types")

        # Use StoryBuilder for spine-based story formation (replaces PrincipledCaseBuilder)
        # StoryBuilder uses spine + mode + membrane approach which handles star-shaped patterns
        # where k=2 motif recurrence fails (rotating companions, no pair recurs)
        story_builder = StoryBuilder(
            hub_fraction_threshold=0.20,  # Entity in >20% of incidents = hub
            hub_min_incidents=5,  # Need at least 5 incidents to compute hubness
            min_incidents_for_story=3,  # Minimum incidents to form a story
            mode_gap_days=30,  # Gap to separate temporal modes
        )

        result = story_builder.build_from_incidents(incidents)
        logger.info(
            f"  StoryBuilder: {result.stats['stories_formed']} stories, "
            f"{result.stats['spine_candidates']} spine candidates, "
            f"{result.stats['hub_entities']} hub entities suppressed"
        )

        # Convert stories to CanonicalCase objects
        cases: List[CanonicalCase] = []
        incident_contexts = await self._load_incidents_for_case_formation()

        # Stories (spine-based, unified model replacing CaseCores + EntityCases)
        for story_id, story in result.stories.items():
            # Convert CompleteStory to EntityLens (immutable navigation view)
            base_lens = story.to_lens()

            # Build companion counts externally (lens is immutable)
            companion_counts: Dict[str, int] = {}
            for iid in story.core_a_ids | story.core_b_ids:
                ctx = incident_contexts.get(iid)
                if ctx:
                    for anchor in ctx.anchor_entities:
                        if anchor != story.spine:
                            companion_counts[anchor] = companion_counts.get(anchor, 0) + 1

            # Create enriched lens with companions (immutable)
            lens = EntityLens.create(
                entity=base_lens.entity,
                incident_ids=base_lens.incident_ids,
                companion_counts=dict(sorted(companion_counts.items(), key=lambda x: -x[1])[:20]),
            )

            case = await self._create_case_from_lens(
                lens, incidents, incident_contexts
            )
            if case:
                cases.append(case)

        logger.info(f"  Total cases to persist: {len(cases)}")

        return cases

    async def _load_incidents_for_case_formation(self) -> Dict[str, IncidentContext]:
        """Load L3 Incidents with context needed for case clustering.

        Loads from Neo4j graph: Incident -> Surface -> Claim
        Claims are stored in Neo4j with text property.
        """
        rows = await self.neo4j._execute_read('''
            MATCH (i:Incident)
            OPTIONAL MATCH (i)-[:CONTAINS]->(s:Surface)
            OPTIONAL MATCH (s)-[:CONTAINS]->(c:Claim)
            WITH i,
                 collect(DISTINCT s.id) as surface_ids,
                 collect(DISTINCT s.sources) as sources_list,
                 collect(DISTINCT c.text) as all_claims
            RETURN i.id as id,
                   i.anchor_entities as anchors,
                   i.companion_entities as companions,
                   size(surface_ids) as surface_count,
                   reduce(s = [], src IN sources_list | s + coalesce(src, [])) as all_sources,
                   all_claims[0..10] as sample_claims
        ''')

        incidents = {}
        for row in rows:
            sources = set(row.get('all_sources') or [])
            incidents[row['id']] = IncidentContext(
                id=row['id'],
                anchor_entities=set(row.get('anchors') or []),
                companion_entities=set(row.get('companions') or []),
                surface_count=row.get('surface_count') or 0,
                source_count=len(sources),
                sample_claims=[c for c in (row.get('sample_claims') or []) if c],
            )

        return incidents

    async def _load_incidents_as_events(self) -> Dict[str, Event]:
        """Load L3 Incidents as Event objects with anchor_entities for StoryBuilder.

        The builder needs Event objects with:
        - id
        - anchor_entities
        - justification.core_motifs (extracted from Neo4j)
        """
        from reee.types import EventJustification

        rows = await self.neo4j._execute_read('''
            MATCH (i:Incident)
            OPTIONAL MATCH (i)-[:CONTAINS]->(s:Surface)
            WITH i,
                 collect(DISTINCT s.id) as surface_ids,
                 collect(s.time_start) as time_starts,
                 collect(s.time_end) as time_ends
            RETURN i.id as id,
                   i.anchor_entities as anchors,
                   i.companion_entities as companions,
                   i.core_motifs as core_motifs,
                   surface_ids,
                   [t IN time_starts WHERE t IS NOT NULL] as time_starts,
                   [t IN time_ends WHERE t IS NOT NULL] as time_ends
        ''')

        events = {}
        for row in rows:
            # Parse core_motifs from stored data (stored as JSON strings in Neo4j)
            stored_motifs = row.get('core_motifs') or []
            core_motifs = []
            for m in stored_motifs:
                if isinstance(m, str):
                    # JSON string format from principled_weaver
                    import json
                    try:
                        parsed = json.loads(m)
                        if isinstance(parsed, dict) and 'entities' in parsed:
                            core_motifs.append(parsed)
                    except:
                        pass
                elif isinstance(m, dict) and 'entities' in m:
                    core_motifs.append(m)

            anchor_entities = set(row.get('anchors') or [])

            # Build justification if we have motifs
            justification = None
            if core_motifs:
                justification = EventJustification(
                    core_motifs=core_motifs,
                    representative_surfaces=list(row.get('surface_ids') or [])[:3],
                    canonical_handle=f"Incident {row['id']}",
                )

            # Derive time bounds from surfaces
            time_starts = row.get('time_starts') or []
            time_ends = row.get('time_ends') or []

            # Parse datetime strings
            parsed_starts = []
            parsed_ends = []
            for t in time_starts:
                parsed = self._parse_time(t)
                if parsed:
                    parsed_starts.append(parsed)
            for t in time_ends:
                parsed = self._parse_time(t)
                if parsed:
                    parsed_ends.append(parsed)

            time_start = min(parsed_starts) if parsed_starts else None
            time_end = max(parsed_ends) if parsed_ends else None

            events[row['id']] = Event(
                id=row['id'],
                surface_ids=set(row.get('surface_ids') or []),
                anchor_entities=anchor_entities,
                entities=anchor_entities | set(row.get('companions') or []),
                justification=justification,
                time_window=(time_start, time_end),
            )

        return events

    async def _load_entity_types(self) -> Dict[str, str]:
        """Load entity types for location-event binding.

        Returns a dict mapping entity canonical_name → entity_type
        for use in the StoryBuilder.
        """
        rows = await self.neo4j._execute_read('''
            MATCH (e:Entity)
            WHERE e.entity_type IS NOT NULL
            RETURN e.canonical_name as name, e.entity_type as type
        ''')

        return {row['name']: row['type'] for row in rows if row.get('name')}

    # DEPRECATED: This method is no longer used after migration to StoryBuilder
    # Keeping for reference until case_builder.py is fully removed
    async def _create_case_from_builder_result_DEPRECATED(
        self,
        builder_case,
        incidents: Dict[str, Event],
        incident_contexts: Dict[str, IncidentContext],
        builder_result,
    ) -> Optional[CanonicalCase]:
        """DEPRECATED: Was used with PrincipledCaseBuilder CaseCores."""
        incident_ids = builder_case.incident_ids

        # Gather aggregate stats
        all_anchors: Set[str] = set()
        total_surfaces = 0
        total_sources = 0
        sample_claims: List[str] = []

        for iid in incident_ids:
            ctx = incident_contexts.get(iid)
            if ctx:
                all_anchors.update(ctx.anchor_entities)
                total_surfaces += ctx.surface_count
                total_sources += ctx.source_count
                sample_claims.extend(ctx.sample_claims[:3])

        # Use anchor entities from builder case (sorted by participation)
        primary_entities = list(builder_case.anchor_entities)[:5]
        if not primary_entities:
            # Fallback to most connected entities
            entity_counts: Dict[str, int] = {}
            for iid in incident_ids:
                ctx = incident_contexts.get(iid)
                if ctx:
                    for e in ctx.anchor_entities:
                        entity_counts[e] = entity_counts.get(e, 0) + 1
            primary_entities = sorted(
                entity_counts.keys(),
                key=lambda e: -entity_counts[e]
            )[:5]

        # Generate stable case ID
        sorted_incidents = sorted(incident_ids)
        signature = f"case:{','.join(sorted_incidents[:5])}"
        import hashlib
        case_id = f"case_{hashlib.sha256(signature.encode()).hexdigest()[:12]}"

        # Extract binding evidence from builder result
        binding_evidence = []

        # Get shared motifs that form the core
        core_motifs_evidence = set()
        for edge in builder_result.edges:
            if edge.is_core and edge.incident1_id in incident_ids and edge.incident2_id in incident_ids:
                for motif in edge.shared_motifs:
                    core_motifs_evidence.add(tuple(sorted(motif)))

        if core_motifs_evidence:
            for motif in list(core_motifs_evidence)[:3]:
                binding_evidence.append(f"Shared motif: {', '.join(motif)}")
        else:
            shared_ents = set.intersection(*[
                incidents[iid].anchor_entities
                for iid in incident_ids
                if iid in incidents
            ]) if incident_ids else set()
            if shared_ents:
                binding_evidence.append(f"Shared entities: {', '.join(sorted(shared_ents)[:5])}")

        # Generate title and description using LLM
        if self.use_llm:
            title, description = await self._generate_case_narrative(
                sample_claims=sample_claims[:10],
                primary_entities=primary_entities,
                incident_count=len(incident_ids),
                source_count=total_sources,
            )
        else:
            title = f"{primary_entities[0]} Story" if primary_entities else "Developing Story"
            description = f"Coverage from {total_sources} sources across {len(incident_ids)} related developments."

        return CanonicalCase(
            id=case_id,
            incident_ids=incident_ids,
            title=title,
            description=description,
            primary_entities=primary_entities,
            case_type="developing",
            binding_evidence=binding_evidence,
            surface_count=total_surfaces,
            source_count=total_sources,
        )

    async def _create_case_from_lens(
        self,
        lens: "EntityLens",
        incidents: Dict[str, Event],
        incident_contexts: Dict[str, IncidentContext],
    ) -> Optional[CanonicalCase]:
        """Create a CanonicalCase from EntityLens (immutable navigation view).

        EntityLens represents focal entity storylines where:
        - One entity appears across many incidents
        - Companion entities rotate (no pair recurs)
        - k=2 motif recurrence fails but narrative coheres via spine

        Uses case_type="entity_storyline" to differentiate from other case types.

        Args:
            lens: Immutable EntityLens from story.to_lens() with companions enriched
            incidents: Dict of incident_id → Event
            incident_contexts: Dict of incident_id → IncidentContext for aggregation
        """
        incident_ids = lens.incident_ids  # frozenset
        focal_entity = lens.entity

        # Gather aggregate stats
        total_surfaces = 0
        total_sources = 0
        sample_claims: List[str] = []

        for iid in incident_ids:
            ctx = incident_contexts.get(iid)
            if ctx:
                total_surfaces += ctx.surface_count
                total_sources += ctx.source_count
                sample_claims.extend(ctx.sample_claims[:2])

        # Primary entities: focal entity first, then top companions
        companion_entities = lens.companion_entities  # Dict view from immutable tuple
        primary_entities = [focal_entity]
        for companion, count in list(companion_entities.items())[:4]:
            if companion not in primary_entities:
                primary_entities.append(companion)

        # Use lens_id (stable hash of entity name)
        case_id = lens.lens_id

        # Binding evidence for lens
        binding_evidence = [
            f"Focal entity: {focal_entity} ({len(incident_ids)} incidents)",
            f"Top companions: {', '.join(list(companion_entities.keys())[:5])}",
        ]

        # Generate title and description
        if self.use_llm:
            title, description = await self._generate_case_narrative(
                sample_claims=sample_claims[:10],
                primary_entities=primary_entities,
                incident_count=len(incident_ids),
                source_count=total_sources,
            )
        else:
            title = f"{focal_entity}: Ongoing Coverage"
            description = (
                f"Coverage of {focal_entity} across {len(incident_ids)} developments "
                f"from {total_sources} sources."
            )

        return CanonicalCase(
            id=case_id,
            incident_ids=incident_ids,
            title=title,
            description=description,
            primary_entities=primary_entities,
            case_type="entity_storyline",  # Differentiates from CaseCores
            binding_evidence=binding_evidence,
            surface_count=total_surfaces,
            source_count=total_sources,
        )

    # DEPRECATED: Old clustering method - kept for reference
    def _cluster_incidents_by_binding(
        self,
        incidents: Dict[str, IncidentContext]
    ) -> List[Set[str]]:
        """
        Cluster incidents that share binding evidence.

        Binding evidence = shared anchor entities + compatible companions.

        Uses Union-Find to build connected components.
        """
        # Build adjacency based on binding constraints
        incident_ids = list(incidents.keys())

        # Union-Find parent
        parent = {iid: iid for iid in incident_ids}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Check all pairs for binding
        min_shared_anchors = 2
        companion_threshold = 0.1  # Looser than streaming - batch can be more liberal

        for i, id1 in enumerate(incident_ids):
            inc1 = incidents[id1]

            for id2 in incident_ids[i + 1:]:
                inc2 = incidents[id2]

                # Check shared anchors
                shared_anchors = inc1.anchor_entities & inc2.anchor_entities
                if len(shared_anchors) < min_shared_anchors:
                    continue

                # Check companion compatibility (if both have companions)
                if inc1.companion_entities and inc2.companion_entities:
                    intersection = inc1.companion_entities & inc2.companion_entities
                    union_size = len(inc1.companion_entities | inc2.companion_entities)
                    overlap = len(intersection) / union_size if union_size else 0

                    if overlap < companion_threshold:
                        continue

                # Binding evidence found - merge
                union(id1, id2)

        # Build clusters from Union-Find
        clusters_map: Dict[str, Set[str]] = {}
        for iid in incident_ids:
            root = find(iid)
            if root not in clusters_map:
                clusters_map[root] = set()
            clusters_map[root].add(iid)

        return list(clusters_map.values())

    async def _create_case_from_cluster(
        self,
        incident_ids: Set[str],
        incidents: Dict[str, IncidentContext]
    ) -> Optional[CanonicalCase]:
        """Create a CanonicalCase from a cluster of incidents."""
        # Gather aggregate stats
        all_anchors: Set[str] = set()
        all_companions: Set[str] = set()
        total_surfaces = 0
        total_sources = 0
        sample_claims: List[str] = []

        for iid in incident_ids:
            inc = incidents.get(iid)
            if not inc:
                continue
            all_anchors.update(inc.anchor_entities)
            all_companions.update(inc.companion_entities)
            total_surfaces += inc.surface_count
            total_sources += inc.source_count
            sample_claims.extend(inc.sample_claims[:3])

        # Identify primary entities (most connected)
        entity_counts: Dict[str, int] = {}
        for iid in incident_ids:
            inc = incidents.get(iid)
            if inc:
                for e in inc.anchor_entities:
                    entity_counts[e] = entity_counts.get(e, 0) + 1

        primary_entities = sorted(
            entity_counts.keys(),
            key=lambda e: -entity_counts[e]
        )[:5]

        # Generate stable case ID
        sorted_incidents = sorted(incident_ids)
        signature = f"case:{','.join(sorted_incidents[:5])}"
        import hashlib
        case_id = f"case_{hashlib.sha256(signature.encode()).hexdigest()[:12]}"

        # Determine binding evidence (what connects these incidents)
        shared_anchors = set.intersection(*[
            incidents[iid].anchor_entities
            for iid in incident_ids
            if iid in incidents
        ]) if incident_ids else set()

        binding_evidence = [f"Shared entities: {', '.join(sorted(shared_anchors)[:5])}"]

        # Generate title and description using LLM
        # Always try LLM if enabled - even without claims, entities provide context
        if self.use_llm:
            title, description = await self._generate_case_narrative(
                sample_claims=sample_claims[:10],
                primary_entities=primary_entities,
                incident_count=len(incident_ids),
                source_count=total_sources,
            )
        else:
            # Fallback title (only used when LLM disabled)
            title = f"{primary_entities[0]} Story" if primary_entities else "Developing Story"
            description = f"Coverage from {total_sources} sources across {len(incident_ids)} related developments."

        return CanonicalCase(
            id=case_id,
            incident_ids=incident_ids,
            title=title,
            description=description,
            primary_entities=primary_entities,
            case_type="developing",
            binding_evidence=binding_evidence,
            surface_count=total_surfaces,
            source_count=total_sources,
        )

    async def _generate_case_narrative(
        self,
        sample_claims: List[str],
        primary_entities: List[str],
        incident_count: int,
        source_count: int,
    ) -> tuple:
        """Generate case title and description using LLM."""
        entities_text = ", ".join(primary_entities[:5]) if primary_entities else "Unknown"

        if sample_claims:
            claims_text = "\n".join(f"- {c[:300]}" for c in sample_claims[:10])
            prompt = f"""Based on these news claims about {entities_text}, generate a case title and description.

This represents a news story or developing situation with {incident_count} related developments from {source_count} sources.

Claims:
{claims_text}

Requirements:
- Title: 5-15 words, describes THE STORY/EVENT (not just entity names)
- Use active voice, present tense for breaking news
- Be specific about what happened (fire, announcement, scandal, etc.)
- Description: 2-3 sentences summarizing the story arc

Respond in JSON format:
{{"title": "...", "description": "..."}}"""
        else:
            # No claims available - generate based on entities only
            prompt = f"""Generate a brief news case title for a developing story involving: {entities_text}

This represents {incident_count} related developments from {source_count} sources.

Requirements:
- Title: 5-10 words, describes a plausible news story
- DO NOT just concatenate entity names like "Entity1 + Entity2 Case"
- Instead, describe a likely scenario (e.g., "Investigation into...", "Developments in...", "Updates on...")
- Description: 1-2 sentences about what this story might cover

Respond in JSON format:
{{"title": "...", "description": "..."}}"""

        try:
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a news editor creating headlines. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=250,
                response_format={"type": "json_object"},
            )

            import json
            result = json.loads(response.choices[0].message.content)
            return (
                result.get("title", f"{primary_entities[0]} Story" if primary_entities else "Developing Story"),
                result.get("description", f"Coverage from {source_count} sources.")
            )

        except Exception as e:
            logger.warning(f"LLM case narrative failed: {e}")
            return (
                f"{primary_entities[0]} Case" if primary_entities else "Developing Story",
                f"Coverage from {source_count} sources across {incident_count} developments."
            )

    def _compute_case_signature(self, case: CanonicalCase) -> str:
        """Compute stable scope signature for case (L4).

        The signature is deterministic based on:
        - Scale: "case"
        - Sorted primary entities (top 10)
        - Time bin: month (YYYY-MM)

        This ensures the same case gets the same signature across rebuilds.
        """
        sorted_entities = sorted(case.primary_entities)[:10]
        time_bin = "unknown"
        if case.time_start:
            time_bin = f"{case.time_start.year}-{case.time_start.month:02d}"

        sig_parts = ["case", ",".join(sorted_entities), time_bin]
        sig_string = "|".join(sig_parts)
        sig_hash = hashlib.sha256(sig_string.encode()).hexdigest()[:12]
        return f"story_{sig_hash}"

    async def _persist_cases(self, cases: List[CanonicalCase]):
        """Persist L4 Cases to Neo4j with :Story label and scope_signature.

        Strategy:
        - MERGE by case ID (stable across rebuilds)
        - Flag stale cases (DO NOT delete - presentation only!)
        - Connect cases to their incidents

        NOTE: Canonical layer is PRESENTATION-ONLY. It enriches with titles/descriptions
        but does NOT delete structural nodes. Stale cases are flagged, not deleted.
        """
        # First, collect IDs of valid cases
        valid_case_ids = {c.id for c in cases}

        # Flag stale cases (DO NOT DELETE - presentation only!)
        # Stale cases may need to be cleaned up by the principled weaver
        await self.neo4j._execute_write('''
            MATCH (c:Case)
            WHERE c.id STARTS WITH 'case_'
              AND NOT c.id IN $valid_ids
            SET c.is_stale = true,
                c.stale_since = datetime()
        ''', {'valid_ids': list(valid_case_ids)})

        # Persist each case
        for case in cases:
            # Compute stable scope signature for case
            scope_sig = self._compute_case_signature(case)

            # L4 AUTHORITY: Canonical worker updates PRESENTATION fields ONLY
            # CONTAINS relationships are owned by the weaver (structural authority)
            # This ensures: "canonical worker can delete and rerun without changing
            # any case membership—only presentation changes"
            await self.neo4j._execute_write('''
                MATCH (c:Case {id: $id})
                SET c:Event,
                    c:Story,
                    c.scale = 'case',
                    c.scope_signature = $scope_sig,
                    c.title = $title,
                    c.canonical_title = $title,
                    c.description = $description,
                    c.primary_entities = $primary_entities,
                    c.case_type = $case_type,
                    c.binding_evidence = $binding_evidence,
                    c.surface_count = $surface_count,
                    c.source_count = $source_count,
                    c.presentation_updated_at = datetime(),
                    c.presentation_updated_by = 'canonical_worker'
                // NOTE: DO NOT touch CONTAINS relationships - weaver owns structure
            ''', {
                'id': case.id,
                'scope_sig': scope_sig,
                'title': case.title,
                'description': case.description,
                'primary_entities': case.primary_entities,
                'case_type': case.case_type,
                'binding_evidence': case.binding_evidence,
                'surface_count': case.surface_count,
                'source_count': case.source_count,
            })


async def main():
    """Main entry point."""
    logger.info("Starting CanonicalWorker...")

    # Connect to PostgreSQL
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'db'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        database=os.getenv('POSTGRES_DB', 'phi_here'),
        user=os.getenv('POSTGRES_USER', 'phi_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'phi_password_dev'),
        min_size=2,
        max_size=10,
    )

    # Connect to Neo4j
    neo4j = Neo4jService()
    await neo4j.connect()

    # Start worker
    worker = CanonicalWorker(
        db_pool=db_pool,
        neo4j=neo4j,
        rebuild_interval=300,  # 5 minutes
    )

    try:
        await worker.start()
    finally:
        await db_pool.close()
        await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
