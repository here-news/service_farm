"""
Event Repository - Neo4j Primary Storage

Storage strategy:
- Neo4j: Event nodes with ALL metadata, relationships (PRIMARY SOURCE OF TRUTH)
- PostgreSQL: Event embeddings ONLY (core.event_embeddings for vector search)

Architecture:
- create(): Write to Neo4j + embedding to PostgreSQL
- get_by_id(): Read from Neo4j, join embedding from PostgreSQL
- find_candidates(): Vector search on PostgreSQL, hydrate from Neo4j

ID format: ev_xxxxxxxx (11 chars)
"""
import logging
import json
from typing import Optional, List, Set, Tuple, Dict
from datetime import datetime
import asyncpg
import numpy as np

from pgvector.asyncpg import register_vector

from models.domain.event import Event, StructuredNarrative
from services.neo4j_service import Neo4jService
from utils.datetime_utils import neo4j_datetime_to_python
from utils.id_generator import is_uuid, uuid_to_short_id

logger = logging.getLogger(__name__)


class EventRepository:
    """
    Repository for Event domain model - Neo4j primary storage
    """

    def __init__(self, db_pool: asyncpg.Pool, neo4j_service: Neo4jService):
        self.db_pool = db_pool
        self.neo4j = neo4j_service

    async def create(self, event: Event) -> Event:
        """
        Create event in Neo4j + embedding in PostgreSQL

        Args:
            event: Event domain model

        Returns:
            Created event
        """
        # Store ALL event data in Neo4j (primary storage)
        # summary, location, coherence stored as direct properties
        await self.neo4j._execute_write("""
            CREATE (e:Event {
                id: $event_id,
                canonical_name: $canonical_name,
                event_type: $event_type,
                status: $status,
                confidence: $confidence,
                event_scale: $event_scale,
                event_start: $event_start,
                event_end: $event_end,
                summary: $summary,
                location: $location,
                coherence: $coherence,
                version_major: $version_major,
                version_minor: $version_minor,
                created_at: datetime(),
                updated_at: datetime()
            })
        """, {
            'event_id': str(event.id),
            'canonical_name': event.canonical_name,
            'event_type': event.event_type,
            'status': event.status,
            'confidence': event.confidence,
            'event_scale': event.event_scale,
            'event_start': event.event_start,
            'event_end': event.event_end,
            'summary': event.summary,
            'location': event.location,
            'coherence': event.coherence,
            'version_major': event.version_major,
            'version_minor': event.version_minor
        })

        # Store ONLY embedding in PostgreSQL (for vector similarity)
        if event.embedding:
            async with self.db_pool.acquire() as conn:
                await register_vector(conn)
                await conn.execute("""
                    INSERT INTO core.event_embeddings (event_id, embedding)
                    VALUES ($1, $2)
                    ON CONFLICT (event_id) DO UPDATE SET
                        embedding = EXCLUDED.embedding
                """, str(event.id), event.embedding)

        logger.info(f"✨ Created event: {event.canonical_name} ({event.id})")
        return event

    async def update(self, event: Event) -> Event:
        """
        Update event in Neo4j + embedding in PostgreSQL

        Args:
            event: Event domain model with updated fields

        Returns:
            Updated event
        """
        # Update Neo4j event properties
        # NOTE: summary, location, coherence stored as direct properties (not in metadata)

        await self.neo4j._execute_write("""
            MATCH (e:Event {id: $event_id})
            SET e.canonical_name = $canonical_name,
                e.event_type = $event_type,
                e.status = $status,
                e.confidence = $confidence,
                e.event_scale = $event_scale,
                e.event_start = $event_start,
                e.event_end = $event_end,
                e.coherence = $coherence,
                e.summary = $summary,
                e.location = $location,
                e.version_major = $version_major,
                e.version_minor = $version_minor,
                e.updated_at = datetime()
        """, {
            'event_id': str(event.id),
            'canonical_name': event.canonical_name,
            'event_type': event.event_type,
            'status': event.status,
            'confidence': event.confidence,
            'event_scale': event.event_scale,
            'event_start': event.event_start,
            'event_end': event.event_end,
            'coherence': event.coherence,
            'summary': event.summary,
            'location': event.location,
            'version_major': event.version_major,
            'version_minor': event.version_minor
        })

        # Update embedding in PostgreSQL
        if event.embedding:
            async with self.db_pool.acquire() as conn:
                await register_vector(conn)
                await conn.execute("""
                    UPDATE core.event_embeddings
                    SET embedding = $2
                    WHERE event_id = $1
                """, str(event.id), event.embedding)

        logger.debug(f"📊 Updated event: {event.canonical_name}")
        return event

    async def bump_version(
        self,
        event_id: str,
        major: bool = False,
        old_coherence: float = None,
        new_coherence: float = None
    ) -> str:
        """
        Bump event version and return new version string.

        Versioning rules:
        - Minor bump: narrative regeneration, claim additions
        - Major bump: coherence leap (≥0.1 increase), pattern change

        Args:
            event_id: Event ID
            major: Force major bump
            old_coherence: Previous coherence (for auto-detecting major bump)
            new_coherence: New coherence (for auto-detecting major bump)

        Returns:
            New version string (e.g., "1.3")
        """
        # Auto-detect major bump from coherence leap
        if not major and old_coherence is not None and new_coherence is not None:
            coherence_delta = new_coherence - old_coherence
            if coherence_delta >= 0.1:
                major = True
                logger.info(f"📈 Coherence leap detected: {old_coherence:.2f} → {new_coherence:.2f} (+{coherence_delta:.2f})")

        if major:
            result = await self.neo4j._execute_write("""
                MATCH (e:Event {id: $event_id})
                SET e.version_major = COALESCE(e.version_major, 0) + 1,
                    e.version_minor = 0,
                    e.updated_at = datetime()
                RETURN e.version_major as major, e.version_minor as minor
            """, {'event_id': event_id})
        else:
            result = await self.neo4j._execute_write("""
                MATCH (e:Event {id: $event_id})
                SET e.version_major = COALESCE(e.version_major, 0),
                    e.version_minor = COALESCE(e.version_minor, 0) + 1,
                    e.updated_at = datetime()
                RETURN e.version_major as major, e.version_minor as minor
            """, {'event_id': event_id})

        if result:
            major_v = result['major'] if result['major'] is not None else 0
            minor_v = result['minor'] if result['minor'] is not None else 1
            version = f"{major_v}.{minor_v}"
            bump_type = "MAJOR" if major else "minor"
            logger.info(f"🏷️ Event {event_id} version bump ({bump_type}): {version}")
            return version

        return "0.1"

    async def get_by_id(self, event_id: str) -> Optional[Event]:
        """
        Retrieve event from Neo4j by ID

        Args:
            event_id: Event ID (ev_xxxxxxxx format)

        Returns:
            Event model or None
        """
        # Fetch event from Neo4j (primary source)
        result = await self.neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})
            RETURN e
        """, {'event_id': event_id})

        if not result:
            return None

        node = result[0]['e']

        # Parse metadata
        metadata = node.get('metadata_json', {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata) if metadata else {}

        # NOTE: claim_ids NO LONGER in metadata - use EventRepository.get_event_claims() to fetch

        # Get parent_event_id from Neo4j :SPAWNS relationships
        parent_event_id = await self.neo4j.get_parent_event_id(event_id=event_id)

        # Fetch embedding from PostgreSQL
        embedding = None
        async with self.db_pool.acquire() as conn:
            await register_vector(conn)
            row = await conn.fetchrow("""
                SELECT embedding FROM core.event_embeddings WHERE event_id = $1
            """, event_id)
            if row and row['embedding'] is not None:
                raw_emb = row['embedding']
                # pgvector returns numpy array, convert to list
                embedding = [float(x) for x in raw_emb]

        # Model handles UUID conversion in __post_init__
        # Read summary, location, coherence from Neo4j node properties (not metadata)

        # Parse structured_narrative if present
        structured_narrative = None
        structured_narrative_json = node.get('structured_narrative')
        if structured_narrative_json:
            try:
                if isinstance(structured_narrative_json, str):
                    narrative_data = json.loads(structured_narrative_json)
                else:
                    narrative_data = structured_narrative_json
                structured_narrative = StructuredNarrative.from_dict(narrative_data)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse structured_narrative for {event_id}: {e}")

        return Event(
            id=node['id'],
            canonical_name=node.get('canonical_name', ''),
            event_type=node.get('event_type', 'INCIDENT'),
            event_start=neo4j_datetime_to_python(node.get('event_start')),
            event_end=neo4j_datetime_to_python(node.get('event_end')),
            status=node.get('status', 'provisional'),
            confidence=node.get('confidence', 0.3),
            event_scale=node.get('event_scale', 'micro'),
            embedding=embedding,
            metadata=metadata,
            parent_event_id=parent_event_id,
            pages_count=0,  # Legacy field
            claims_count=0,  # Use get_event_claims() to fetch actual count from graph
            summary=node.get('summary'),
            narrative=structured_narrative,
            location=node.get('location'),
            coherence=node.get('coherence', 0.3),
            version_major=node.get('version_major', 0),
            version_minor=node.get('version_minor', 1)
        )

    async def get_sub_events(self, parent_id: str) -> List[Event]:
        """
        Get all sub-events for a parent event

        Uses Neo4j :SPAWNS relationships as source of truth

        Args:
            parent_id: Parent event ID (ev_xxxxxxxx format)

        Returns:
            List of sub-events
        """
        # Get sub-event IDs from Neo4j relationships
        sub_event_ids = await self.neo4j.get_sub_event_ids(event_id=parent_id)

        if not sub_event_ids:
            return []

        # Fetch each sub-event
        sub_events = []
        for sub_event_id in sub_event_ids:
            sub_event = await self.get_by_id(sub_event_id)
            if sub_event:
                sub_events.append(sub_event)

        return sub_events

    async def get_candidate_events(
        self,
        entity_ids: Set[str],
        limit: int = 20
    ) -> List[Tuple[Event, Set[str], List[str]]]:
        """
        Get candidate events that share entities with the given set.

        Pure data access - no scoring logic. Returns events with their
        entity IDs and source page IDs for the caller to score.

        Args:
            entity_ids: Entity IDs (en_xxxxxxxx format) to match against
            limit: Maximum candidates to return

        Returns:
            List of (event, event_entity_ids, source_page_ids) tuples,
            sorted by overlap count desc
        """
        if not entity_ids:
            return []

        entity_id_strings = list(entity_ids)

        # Query Neo4j for events with entity overlap and their source pages
        events_with_overlap = await self.neo4j._execute_read("""
            MATCH (e:Event)-[:INVOLVES]->(entity:Entity)
            WHERE entity.id IN $entity_ids
            WITH e, count(DISTINCT entity) as overlap_count
            WHERE overlap_count > 0
            // Get source pages for this event
            OPTIONAL MATCH (e)-[:INTAKES]->(c:Claim)<-[:EMITS]-(p:Page)
            WITH e, overlap_count, collect(DISTINCT p.id) as page_ids
            RETURN e.id as event_id, overlap_count, page_ids
            ORDER BY overlap_count DESC
            LIMIT $limit
        """, {'entity_ids': entity_id_strings, 'limit': limit})

        if not events_with_overlap:
            return []

        candidates = []
        for row in events_with_overlap:
            event_id = row['event_id']
            source_page_ids = row['page_ids'] or []

            event = await self.get_by_id(event_id)
            if not event:
                continue

            event_entity_ids = await self._get_event_entity_ids(event_id)
            candidates.append((event, event_entity_ids, source_page_ids))

        return candidates

    async def get_candidate_events_by_embedding(
        self,
        page_embedding: List[float],
        threshold: float = 0.50,
        limit: int = 10
    ) -> List[Tuple[Event, float]]:
        """
        Get candidate events by semantic similarity (embedding search).

        This is a FALLBACK when entity-based search returns no candidates.
        Uses pgvector cosine similarity to find semantically similar events.

        Args:
            page_embedding: Page embedding vector (1536 dims)
            threshold: Minimum cosine similarity (0.0-1.0)
            limit: Maximum candidates to return

        Returns:
            List of (event, similarity_score) tuples, sorted by similarity desc
        """
        if not page_embedding:
            return []

        async with self.db_pool.acquire() as conn:
            # Use pgvector cosine distance (1 - cosine_similarity)
            # So we filter by (1 - threshold) and sort ascending
            rows = await conn.fetch("""
                SELECT event_id, 1 - (embedding <=> $1::vector) as similarity
                FROM core.event_embeddings
                WHERE 1 - (embedding <=> $1::vector) >= $2
                ORDER BY embedding <=> $1::vector ASC
                LIMIT $3
            """, page_embedding, threshold, limit)

        if not rows:
            logger.debug(f"No semantic candidates found (threshold={threshold})")
            return []

        candidates = []
        for row in rows:
            event_id = row['event_id']
            similarity = float(row['similarity'])

            event = await self.get_by_id(event_id)
            if event:
                candidates.append((event, similarity))
                logger.debug(f"  Semantic candidate: {event.canonical_name} (sim={similarity:.3f})")

        return candidates

    async def _get_event_entity_ids(self, event_id: str) -> Set[str]:
        """Get all entity IDs for an event from Neo4j"""
        result = await self.neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[:INVOLVES]->(entity:Entity)
            RETURN entity.id as entity_id
        """, {'event_id': event_id})

        return {row['entity_id'] for row in result}

    def _parse_embedding(self, emb_data) -> Optional[List[float]]:
        """Parse embedding from PostgreSQL vector"""
        try:
            # If already a list (pgvector returns as list), return directly
            if isinstance(emb_data, list):
                return [float(x) for x in emb_data]

            # If string format, parse it
            if isinstance(emb_data, str):
                if emb_data.startswith('[') and emb_data.endswith(']'):
                    return [float(x.strip()) for x in emb_data[1:-1].split(',')]
        except Exception as e:
            logger.warning(f"Failed to parse embedding: {e}")
        return None

    async def link_claim(self, event: Event, claim, relationship_type: str = "INTAKES") -> None:
        """
        Link a claim to an event in Neo4j knowledge graph

        Args:
            event: Event domain model
            claim: Claim domain model
            relationship_type: Type of relationship (INTAKES only for Event->Claim)

        This creates:
        1. Claim node in Neo4j (if not exists)
        2. Event-[relationship_type]->Claim relationship
        """
        # Import here to avoid circular dependency
        from models.domain.claim import Claim

        # Create or merge Claim node in Neo4j (idempotent)
        await self.neo4j.create_claim(
            claim_id=str(claim.id),
            text=claim.text,
            modality=claim.modality,
            confidence=claim.confidence,
            event_time=claim.event_time,
            page_id=str(claim.page_id)
        )

        # Link Event->Claim in graph
        await self.neo4j.link_claim_to_event(
            event_id=str(event.id),
            claim_id=str(claim.id),
            relationship_type=relationship_type
        )

        # Also link entities from this claim to the event via INVOLVES
        # This ensures all entities mentioned in claims are linked to the event
        if hasattr(claim, 'entity_ids') and claim.entity_ids:
            for entity_id in claim.entity_ids:
                await self.neo4j._execute_write("""
                    MATCH (e:Event {id: $event_id})
                    MATCH (entity:Entity {id: $entity_id})
                    MERGE (e)-[r:INVOLVES]->(entity)
                    ON CREATE SET r.created_at = datetime()
                """, {
                    'event_id': str(event.id),
                    'entity_id': str(entity_id)
                })

        logger.debug(f"🔗 Linked claim {claim.id} to event {event.canonical_name}")

    async def get_event_claims(self, event_id: str) -> List:
        """
        Get all claims linked to an event from Neo4j graph, hydrated from PostgreSQL

        Args:
            event_id: Event ID (ev_xxxxxxxx format)

        Returns:
            List of Claim domain models with entity_ids populated
        """
        # Import here to avoid circular dependency
        from models.domain.claim import Claim

        # Fetch all claim data from Neo4j (primary storage)
        # Join with Page via CONTAINS relationship to get page_id
        result = await self.neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[r:INTAKES]->(c:Claim)
            OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
            RETURN c.id as id,
                   COALESCE(c.page_id, p.id) as page_id,
                   c.text as text,
                   c.modality as modality, c.confidence as confidence,
                   c.event_time as event_time, c.metadata as metadata
            ORDER BY c.event_time
        """, {'event_id': event_id})

        if not result:
            return []

        # Build claims from Neo4j data
        claims = []
        for row in result:
            # Handle metadata - Neo4j stores as string (JSON)
            metadata = row['metadata']
            if metadata is None:
                metadata = {}
            elif isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            claim = Claim(
                id=row['id'],
                page_id=row['page_id'],
                text=row['text'],
                modality=row.get('modality', 'observation'),
                confidence=row.get('confidence', 0.8),
                event_time=row['event_time'],
                metadata=metadata
            )
            claims.append(claim)

        logger.debug(f"Fetched {len(claims)} claims for event {event_id} with entity_ids")
        return claims

    async def get_event_claims_with_timeline_data(self, event_id: str) -> List:
        """
        Get all claims for an event WITH reported_time (page pub_time) for timeline generation.

        This joins Neo4j graph data with PostgreSQL claim + page data to get:
        - event_time (when fact occurred)
        - reported_time (when we learned it = page.pub_time)

        Args:
            event_id: Event ID (ev_xxxxxxxx format)

        Returns:
            List of Claim domain models with reported_time populated
        """
        from models.domain.claim import Claim

        # Step 1: Get claim IDs from Neo4j graph
        neo4j_result = await self.neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[r:INTAKES]->(c:Claim)
            RETURN c.id as claim_id, type(r) as relationship
        """, {'event_id': event_id})

        if not neo4j_result:
            return []

        claim_ids = [row['claim_id'] for row in neo4j_result]

        # Step 2: Get full claim data from PostgreSQL WITH page pub_time
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    c.id, c.page_id, c.text, c.event_time, c.confidence,
                    c.modality, c.metadata, c.embedding, c.created_at,
                    p.pub_time as reported_time
                FROM core.claims c
                JOIN core.pages p ON c.page_id = p.id
                WHERE c.id = ANY($1::text[])
                ORDER BY c.event_time NULLS LAST, p.pub_time
            """, claim_ids)

            claims = []
            for row in rows:
                metadata = json.loads(row['metadata']) if row['metadata'] else {}

                claim = Claim(
                    id=row['id'],
                    page_id=row['page_id'],
                    text=row['text'],
                    event_time=row['event_time'],
                    reported_time=row['reported_time'],  # ← Populated from page.pub_time
                    confidence=row['confidence'],
                    modality=row['modality'],
                    embedding=self._parse_embedding(row['embedding']),
                    metadata=metadata,
                    created_at=row['created_at']
                )
                claims.append(claim)

            return claims

    async def create_sub_event_relationship(self, parent_id: str, child_id: str) -> None:
        """
        Create parent-child relationship between events

        Args:
            parent_id: Parent event ID (ev_xxxxxxxx format)
            child_id: Child event ID (ev_xxxxxxxx format)
        """
        await self.neo4j.create_event_relationship(
            parent_id=parent_id,
            child_id=child_id,
            relationship_type="CONTAINS"
        )

        logger.info(f"🔗 Created CONTAINS relationship: {parent_id} → {child_id}")

    async def link_event_to_entities(self, event_id: str, entity_ids: Set[str]) -> None:
        """
        Link event to multiple entities in Neo4j graph

        Args:
            event_id: Event ID (ev_xxxxxxxx format)
            entity_ids: Set of entity IDs (en_xxxxxxxx format)
        """
        if not entity_ids:
            return

        for entity_id in entity_ids:
            await self.neo4j._execute_write("""
                MATCH (e:Event {id: $event_id})
                MATCH (entity:Entity {id: $entity_id})
                MERGE (e)-[r:INVOLVES]->(entity)
                ON CREATE SET r.created_at = datetime()
            """, {
                'event_id': event_id,
                'entity_id': entity_id
            })

        logger.debug(f"🔗 Linked event {event_id} to {len(entity_ids)} entities")

    async def update_claim_plausibility(
        self,
        event_id: str,
        claim_id: str,
        plausibility: float
    ) -> None:
        """
        Update plausibility score on INTAKES relationship between event and claim.

        This stores Bayesian posterior as a property on the relationship,
        allowing queries like "get high-plausibility claims for event".

        Args:
            event_id: Event ID (ev_xxxxxxxx format)
            claim_id: Claim ID (cl_xxxxxxxx format)
            plausibility: Posterior probability (0.0-1.0)
        """
        await self.neo4j._execute_write("""
            MATCH (e:Event {id: $event_id})-[r:INTAKES]->(c:Claim {id: $claim_id})
            SET r.plausibility = $plausibility,
                r.plausibility_updated = datetime()
        """, {
            'event_id': event_id,
            'claim_id': claim_id,
            'plausibility': plausibility
        })

    async def get_claim_plausibility(
        self,
        event_id: str,
        claim_id: str
    ) -> Optional[float]:
        """
        Get plausibility score for a claim in the context of an event.

        Args:
            event_id: Event ID
            claim_id: Claim ID

        Returns:
            Plausibility score (0.0-1.0) or None if not set
        """
        result = await self.neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[r:INTAKES]->(c:Claim {id: $claim_id})
            RETURN r.plausibility as plausibility
        """, {
            'event_id': event_id,
            'claim_id': claim_id
        })

        if result and result[0].get('plausibility') is not None:
            return float(result[0]['plausibility'])
        return None

    async def get_all_claim_plausibilities(
        self,
        event_id: str
    ) -> Dict[str, float]:
        """
        Get all plausibility scores for claims in an event.

        Used during hydration to restore cached topology results.

        Args:
            event_id: Event ID

        Returns:
            Dict mapping claim_id -> plausibility score
        """
        result = await self.neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[r:INTAKES]->(c:Claim)
            WHERE r.plausibility IS NOT NULL
            RETURN c.id as claim_id, r.plausibility as plausibility
        """, {
            'event_id': event_id
        })

        plausibilities = {}
        for row in result:
            plausibilities[row['claim_id']] = float(row['plausibility'])

        return plausibilities

    async def update_claim_affinity(
        self,
        event_id: str,
        claim_id: str,
        affinity: float
    ) -> None:
        """
        Update affinity score on INTAKES relationship between event and claim.

        Affinity measures how well a claim belongs to this event based on:
        - Entity overlap with event's core entities
        - Semantic similarity to event embedding
        - Topology integration (corroborations vs contradictions)
        - Narrative inclusion

        Args:
            event_id: Event ID
            claim_id: Claim ID
            affinity: Affinity score (0.0-1.0)
        """
        await self.neo4j._execute_write("""
            MATCH (e:Event {id: $event_id})-[r:INTAKES]->(c:Claim {id: $claim_id})
            SET r.affinity = $affinity,
                r.affinity_updated = datetime()
        """, {
            'event_id': event_id,
            'claim_id': claim_id,
            'affinity': affinity
        })

    async def get_low_affinity_claims(
        self,
        event_id: str,
        threshold: float = 0.15
    ) -> List[Dict]:
        """
        Get claims with affinity below threshold.

        Returns claims that may not belong to this event.

        Args:
            event_id: Event ID
            threshold: Affinity threshold (default 0.15)

        Returns:
            List of dicts with claim_id, affinity, text
        """
        result = await self.neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[r:INTAKES]->(c:Claim)
            WHERE r.affinity IS NOT NULL AND r.affinity < $threshold
            RETURN c.id as claim_id, r.affinity as affinity, c.text as text
            ORDER BY r.affinity ASC
        """, {
            'event_id': event_id,
            'threshold': threshold
        })

        return [dict(row) for row in result]

    async def prune_claim(
        self,
        event_id: str,
        claim_id: str
    ) -> bool:
        """
        Remove INTAKES relationship between event and claim.

        The claim becomes orphaned from this event but:
        - Page-[:EMITS]->Claim relationship remains
        - Claim-[:CORROBORATES|CONTRADICTS|UPDATES]->Claim relationships remain
        - Claim can be re-absorbed by another event

        Args:
            event_id: Event ID
            claim_id: Claim ID to prune

        Returns:
            True if relationship was deleted, False if not found
        """
        result = await self.neo4j._execute_write("""
            MATCH (e:Event {id: $event_id})-[r:INTAKES]->(c:Claim {id: $claim_id})
            DELETE r
            RETURN count(r) as deleted
        """, {
            'event_id': event_id,
            'claim_id': claim_id
        })

        deleted = result[0]['deleted'] if result else 0
        logger.info(f"🗑️ Pruned claim {claim_id} from event {event_id}: {'success' if deleted else 'not found'}")
        return deleted > 0

    async def prune_claims_batch(
        self,
        event_id: str,
        claim_ids: List[str]
    ) -> int:
        """
        Remove multiple INTAKES relationships in one operation.

        Args:
            event_id: Event ID
            claim_ids: List of claim IDs to prune

        Returns:
            Number of relationships deleted
        """
        if not claim_ids:
            return 0

        result = await self.neo4j._execute_write("""
            MATCH (e:Event {id: $event_id})-[r:INTAKES]->(c:Claim)
            WHERE c.id IN $claim_ids
            DELETE r
            RETURN count(r) as deleted
        """, {
            'event_id': event_id,
            'claim_ids': claim_ids
        })

        # _execute_write returns single record (dict), not list
        deleted = result['deleted'] if result else 0
        logger.info(f"🗑️ Batch pruned {deleted} claims from event {event_id}")
        return deleted

    async def prune_orphaned_entities(self, event_id: str) -> List[str]:
        """
        Remove INVOLVES relationships for entities no longer connected via claims.

        After claim pruning, some entities may no longer be mentioned by any
        of the event's remaining claims. This method removes those orphaned
        Event-[INVOLVES]->Entity relationships.

        Args:
            event_id: Event ID

        Returns:
            List of entity IDs that were disconnected
        """
        # Find entities that are INVOLVES'd but not mentioned by any remaining claim
        result = await self.neo4j._execute_write("""
            MATCH (e:Event {id: $event_id})-[inv:INVOLVES]->(entity:Entity)
            WHERE NOT EXISTS {
                MATCH (e)-[:INTAKES]->(c:Claim)-[:MENTIONS]->(entity)
            }
            WITH entity, inv
            DELETE inv
            RETURN collect(entity.id) as pruned_entities
        """, {'event_id': event_id})

        pruned = result['pruned_entities'] if result else []
        if pruned:
            logger.info(f"🔗✂️ Disconnected {len(pruned)} orphaned entities from event {event_id}")
        return pruned

    async def list_root_events(
        self,
        status: Optional[str] = None,
        scale: Optional[str] = None,
        limit: int = 50
    ) -> List[dict]:
        """
        List root events (events without parents) with optional filters

        Args:
            status: Optional status filter (provisional, confirmed, etc.)
            scale: Optional scale filter (local, regional, national, international)
            limit: Maximum number of events to return

        Returns:
            List of event dictionaries with child counts
        """
        query = """
            MATCH (e:Event)
            WHERE NOT exists((e)<-[:SPAWNS]-())
        """

        params = {}

        if status:
            query += " AND e.status = $status"
            params['status'] = status

        if scale:
            query += " AND e.event_scale = $scale"
            params['scale'] = scale

        query += """
            WITH e
            OPTIONAL MATCH (e)-[:SPAWNS]->(sub:Event)
            WITH e, count(sub) as child_count
            RETURN e.id as id,
                   e.canonical_name as canonical_name,
                   e.event_type as event_type,
                   e.status as status,
                   e.confidence as confidence,
                   e.coherence as coherence,
                   e.event_scale as event_scale,
                   e.version_major as version_major,
                   e.version_minor as version_minor,
                   coalesce(e.event_start, e.earliest_time) as event_start,
                   coalesce(e.event_end, e.latest_time) as event_end,
                   e.created_at as created_at,
                   e.updated_at as updated_at,
                   e.metadata as metadata,
                   e.summary as summary,
                   child_count
            ORDER BY e.updated_at DESC
            LIMIT $limit
        """
        params['limit'] = limit

        results = await self.neo4j._execute_read(query, params)

        events = []
        for row in results:
            updated_at_dt = neo4j_datetime_to_python(row['updated_at']) if row.get('updated_at') else None
            updated_at_str = updated_at_dt.isoformat() if updated_at_dt else None
            event_dict = {
                'id': row['id'],
                'title': row['canonical_name'],  # Frontend expects 'title'
                'canonical_name': row['canonical_name'],
                'event_type': row['event_type'],
                'status': row['status'],
                'confidence': row['confidence'],
                'coherence': row.get('coherence'),
                'event_scale': row.get('event_scale'),
                'event_start': neo4j_datetime_to_python(row['event_start']) if row.get('event_start') else None,
                'event_end': neo4j_datetime_to_python(row['event_end']) if row.get('event_end') else None,
                'created_at': neo4j_datetime_to_python(row['created_at']) if row.get('created_at') else None,
                'updated_at': updated_at_str,
                'last_updated': updated_at_str,  # Frontend expects 'last_updated'
                'metadata': row.get('metadata', {}),
                'summary': row.get('summary'),
                'child_count': row['child_count']
            }
            events.append(event_dict)

        return events

    async def update_coherence(self, event_id: str, coherence: float) -> None:
        """
        Update only the coherence value for an event.

        Lightweight update method for metabolism cycles.

        Args:
            event_id: Event ID
            coherence: New coherence value (0.0 to 1.0)
        """
        await self.neo4j._execute_write("""
            MATCH (e:Event {id: $event_id})
            SET e.coherence = $coherence,
                e.updated_at = datetime()
        """, {
            'event_id': event_id,
            'coherence': coherence
        })

        logger.debug(f"📊 Updated coherence for {event_id}: {coherence:.3f}")

    async def update_narrative(
        self,
        event_id: str,
        narrative: str,
        structured_narrative: dict = None
    ) -> None:
        """
        Update event narrative/summary.

        Args:
            event_id: Event ID
            narrative: Flat narrative text (for backwards compatibility)
            structured_narrative: Optional structured narrative dict (sections, key_figures, etc.)
        """
        import json

        params = {
            'event_id': event_id,
            'narrative': narrative
        }

        if structured_narrative:
            # Store structured narrative as JSON string
            params['structured_narrative'] = json.dumps(structured_narrative)
            query = """
                MATCH (e:Event {id: $event_id})
                SET e.summary = $narrative,
                    e.structured_narrative = $structured_narrative,
                    e.updated_at = datetime()
            """
        else:
            query = """
                MATCH (e:Event {id: $event_id})
                SET e.summary = $narrative,
                    e.updated_at = datetime()
            """

        await self.neo4j._execute_write(query, params)

        section_info = ""
        if structured_narrative:
            section_count = len(structured_narrative.get('sections', []))
            section_info = f", {section_count} sections"

        logger.debug(f"📝 Updated narrative for {event_id}: {len(narrative)} chars{section_info}")

    async def update_embedding(self, event_id: str, embedding: List[float]) -> None:
        """
        Update event embedding in PostgreSQL.

        Args:
            event_id: Event ID
            embedding: Embedding vector
        """
        async with self.db_pool.acquire() as conn:
            await register_vector(conn)
            await conn.execute("""
                INSERT INTO core.event_embeddings (event_id, embedding)
                VALUES ($1, $2)
                ON CONFLICT (event_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding
            """, event_id, embedding)
        logger.debug(f"📊 Updated embedding for {event_id}: {len(embedding)} dims")

    async def update_status(self, event_id: str, status: str) -> None:
        """
        Update event status.

        Args:
            event_id: Event ID
            status: New status (active, stable, archived, etc.)
        """
        await self.neo4j._execute_write("""
            MATCH (e:Event {id: $event_id})
            SET e.status = $status,
                e.updated_at = datetime()
        """, {
            'event_id': event_id,
            'status': status
        })

        logger.debug(f"📊 Updated status for {event_id}: {status}")

    async def add_thought(self, event_id: str, thought) -> None:
        """
        Add a thought to an event's thoughts list in Neo4j.

        Thoughts are epistemic observations emitted by the living event's
        metabolism - e.g., detected contradictions, coherence drops, etc.

        Args:
            event_id: Event ID
            thought: Thought object from metabolism.py
        """
        thought_data = {
            'id': thought.id,
            'type': thought.type.value,
            'content': thought.content,
            'related_claims': thought.related_claims,
            'related_entities': thought.related_entities,
            'temperature': thought.temperature,
            'coherence': thought.coherence,
            'created_at': thought.created_at.isoformat(),
            'acknowledged': thought.acknowledged
        }

        # Store thought in event's thoughts array property
        await self.neo4j._execute_write("""
            MATCH (e:Event {id: $event_id})
            SET e.thoughts = COALESCE(e.thoughts, []) + [$thought_json],
                e.updated_at = datetime()
        """, {
            'event_id': event_id,
            'thought_json': json.dumps(thought_data)
        })

        logger.info(f"💭 Added thought to {event_id}: {thought.type.value} - {thought.content[:50]}...")

    async def get_thoughts(self, event_id: str) -> List[dict]:
        """
        Get all thoughts for an event.

        Args:
            event_id: Event ID

        Returns:
            List of thought dicts
        """
        result = await self.neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})
            RETURN e.thoughts as thoughts
        """, {'event_id': event_id})

        if not result or not result[0].get('thoughts'):
            return []

        thoughts = []
        for thought_json in result[0]['thoughts']:
            try:
                thought_data = json.loads(thought_json) if isinstance(thought_json, str) else thought_json
                thoughts.append(thought_data)
            except (json.JSONDecodeError, TypeError):
                continue

        return thoughts

    async def acknowledge_thought(self, event_id: str, thought_id: str) -> bool:
        """
        Mark a thought as acknowledged.

        Args:
            event_id: Event ID
            thought_id: Thought ID (th_xxxxxxxx format)

        Returns:
            True if thought was found and acknowledged
        """
        # Get current thoughts
        thoughts = await self.get_thoughts(event_id)

        # Find and update the thought
        updated = False
        for thought in thoughts:
            if thought.get('id') == thought_id:
                thought['acknowledged'] = True
                updated = True
                break

        if not updated:
            return False

        # Write back updated thoughts array
        thought_jsons = [json.dumps(t) for t in thoughts]
        await self.neo4j._execute_write("""
            MATCH (e:Event {id: $event_id})
            SET e.thoughts = $thoughts
        """, {
            'event_id': event_id,
            'thoughts': thought_jsons
        })

        logger.debug(f"✓ Acknowledged thought {thought_id} for event {event_id}")
        return True

    async def get_page_thumbnails_for_event(
        self,
        event_id: str,
        limit: int = 5
    ) -> List[Dict[str, str]]:
        """
        Get page thumbnails for an event (for homepage display).

        Finds pages via Event-[INTAKES]->Claim<-[EMITS]-Page path in Neo4j,
        then fetches thumbnail URLs from PostgreSQL (where they're stored).

        Args:
            event_id: Event ID
            limit: Max number of thumbnails to return (default 5)

        Returns:
            List of dicts with {page_id, thumbnail_url, title, domain}
        """
        # First get page IDs from Neo4j graph
        result = await self.neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[:INTAKES]->(c:Claim)<-[:EMITS]-(p:Page)
            WITH DISTINCT p.id as page_id
            RETURN page_id
            LIMIT $limit
        """, {
            'event_id': event_id,
            'limit': limit * 2  # Get more to filter those without thumbnails
        })

        if not result:
            return []

        page_ids = [row['page_id'] for row in result]

        # Fetch thumbnail URLs from PostgreSQL
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id as page_id, thumbnail_url, title, domain
                FROM core.pages
                WHERE id = ANY($1::text[])
                  AND thumbnail_url IS NOT NULL
                  AND thumbnail_url != ''
                ORDER BY pub_time DESC NULLS LAST
                LIMIT $2
            """, page_ids, limit)

        return [dict(row) for row in rows]

    async def get_latest_thought_for_event(self, event_id: str) -> Optional[Dict]:
        """
        Get the most recent unacknowledged thought for an event.

        Used for homepage display as a stimulating byline.

        Args:
            event_id: Event ID

        Returns:
            Most recent thought dict or None
        """
        thoughts = await self.get_thoughts(event_id)
        if not thoughts:
            return None

        # Filter unacknowledged thoughts and sort by created_at desc
        unacknowledged = [t for t in thoughts if not t.get('acknowledged', False)]
        if not unacknowledged:
            # Fall back to most recent thought even if acknowledged
            return thoughts[-1] if thoughts else None

        # Sort by created_at descending
        unacknowledged.sort(
            key=lambda t: t.get('created_at', ''),
            reverse=True
        )
        return unacknowledged[0]

    async def get_event_claim_count(self, event_id: str) -> int:
        """
        Get count of claims supporting an event.

        Args:
            event_id: Event ID

        Returns:
            Number of claims
        """
        result = await self.neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[:INTAKES]->(c:Claim)
            RETURN count(c) as count
        """, {'event_id': event_id})

        return result[0]['count'] if result else 0

    async def get_event_page_count(self, event_id: str) -> int:
        """
        Get count of distinct source pages for an event.

        Args:
            event_id: Event ID

        Returns:
            Number of distinct pages
        """
        result = await self.neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[:INTAKES]->(c:Claim)<-[:EMITS]-(p:Page)
            RETURN count(DISTINCT p) as count
        """, {'event_id': event_id})

        return result[0]['count'] if result else 0

    async def get_events_by_entity(self, entity_id: str, limit: int = 20) -> List[dict]:
        """
        Get events that involve a specific entity.

        Traverses the graph: Entity <- INVOLVES - Event
        Also gets claims for each event that mention this entity.

        Args:
            entity_id: Entity ID (en_xxxxxxxx format)
            limit: Maximum number of events to return

        Returns:
            List of event dicts with their claims mentioning the entity
        """
        result = await self.neo4j._execute_read("""
            MATCH (entity:Entity {id: $entity_id})<-[:INVOLVES]-(e:Event)
            WITH e
            ORDER BY e.updated_at DESC
            LIMIT $limit

            // Get claims for this event that mention the entity
            OPTIONAL MATCH (e)-[sup:INTAKES]->(c:Claim)-[:MENTIONS]->(entity:Entity {id: $entity_id})
            OPTIONAL MATCH (c)<-[:EMITS]-(p:Page)

            WITH e, collect(DISTINCT {
                id: c.id,
                text: c.text,
                event_time: c.event_time,
                confidence: c.confidence,
                page_id: p.id,
                plausibility: sup.plausibility
            }) as claims

            RETURN e.id as id,
                   e.canonical_name as canonical_name,
                   e.event_type as event_type,
                   e.event_scale as event_scale,
                   e.event_start as event_start,
                   e.event_end as event_end,
                   e.coherence as coherence,
                   e.summary as summary,
                   claims
        """, {
            'entity_id': entity_id,
            'limit': limit
        })

        events = []
        for row in result:
            # Filter out null claims (from OPTIONAL MATCH)
            claims = [c for c in row['claims'] if c.get('id')]

            event_dict = {
                'id': row['id'],
                'slug': row['id'],  # Use ID as slug for now
                'canonical_name': row['canonical_name'],
                'event_type': row['event_type'] or 'event',
                'event_scale': row.get('event_scale'),
                'event_start': neo4j_datetime_to_python(row['event_start']).isoformat() if row.get('event_start') else None,
                'event_end': neo4j_datetime_to_python(row['event_end']).isoformat() if row.get('event_end') else None,
                'coherence': row.get('coherence'),
                'summary': row.get('summary'),
                'claims': [
                    {
                        'id': c['id'],
                        'text': c['text'],
                        'event_time': neo4j_datetime_to_python(c['event_time']).isoformat() if c.get('event_time') else None,
                        'confidence': c.get('confidence'),
                        'page_id': c.get('page_id'),
                    }
                    for c in claims
                ]
            }
            events.append(event_dict)

        return events

