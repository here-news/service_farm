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
from typing import Optional, List, Set, Tuple
from datetime import datetime
import asyncpg
import numpy as np

from pgvector.asyncpg import register_vector

from models.domain.event import Event
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
            'coherence': event.coherence
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
            'location': event.location
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

        # Get parent_event_id from Neo4j :CONTAINS relationships
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
            location=node.get('location'),
            coherence=node.get('coherence', 0.3)
        )

    async def get_sub_events(self, parent_id: str) -> List[Event]:
        """
        Get all sub-events for a parent event

        Uses Neo4j :CONTAINS relationships as source of truth

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

    async def find_candidates(
        self,
        entity_ids: Set[str],
        reference_time: datetime,
        time_window_days: int = 14,
        page_embedding: Optional[List[float]] = None
    ) -> List[Tuple[Event, float]]:
        """
        Find candidate events that might match new information

        Scoring (rebalanced for better same-event detection):
        - Entity overlap: 40% (reliable signal for same event)
        - Time proximity: 30% (critical for matching event instances)
        - Semantic similarity: 30% (page vs narrative embedding can differ)

        Rationale: Page embeddings differ from event narrative embeddings even
        for same event. Entity overlap + time are more reliable signals.

        Args:
            entity_ids: Entity IDs (en_xxxxxxxx format) to match against
            reference_time: Reference time for temporal proximity
            time_window_days: Time window in days
            page_embedding: Optional embedding for semantic matching

        Returns:
            List of (event, match_score) sorted by score descending
        """
        if not entity_ids:
            return []

        # Log inputs for debugging
        logger.info(f"🔍 Finding candidates: {len(entity_ids)} entities, "
                   f"ref_time={reference_time}, page_emb={'YES' if page_embedding else 'NO'}")

        # Get events that share entities (from Neo4j)
        entity_id_strings = list(entity_ids)

        # Query Neo4j for events with entity overlap
        events_with_overlap = await self.neo4j._execute_read("""
            MATCH (e:Event)-[:INVOLVES]->(entity:Entity)
            WHERE entity.id IN $entity_ids
            WITH e, count(DISTINCT entity) as overlap_count
            WHERE overlap_count > 0
            RETURN e.id as event_id, overlap_count
            ORDER BY overlap_count DESC
            LIMIT 20
        """, {'entity_ids': entity_id_strings})

        if not events_with_overlap:
            logger.debug("No candidate events found with entity overlap")
            return []

        candidates = []

        for row in events_with_overlap:
            event_id = row['event_id']
            entity_overlap_count = row['overlap_count']

            # Get full event from Neo4j (model handles UUID conversion)
            event = await self.get_by_id(event_id)
            if not event:
                continue

            # Get event's entities for overlap calculation
            event_entity_ids = await self._get_event_entity_ids(event_id)

            # Calculate entity overlap score (Jaccard similarity)
            if event_entity_ids:
                intersection = len(entity_ids & event_entity_ids)
                union = len(entity_ids | event_entity_ids)
                entity_overlap_score = intersection / union if union > 0 else 0.0
            else:
                entity_overlap_score = 0.0

            # Calculate temporal proximity score
            time_score = 0.0
            if event.event_start and reference_time:
                # Defensive conversion for both timestamps
                event_start_py = neo4j_datetime_to_python(event.event_start)
                reference_time_py = neo4j_datetime_to_python(reference_time) if isinstance(reference_time, str) else reference_time
                if event_start_py and reference_time_py:
                    time_diff_days = abs((reference_time_py - event_start_py).days)
                    time_score = max(0, 1 - (time_diff_days / time_window_days))

            # Calculate semantic similarity score
            semantic_score = 0.0
            if page_embedding and event.embedding:
                vec1 = np.array(page_embedding)
                vec2 = np.array(event.embedding)
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                if norm1 > 0 and norm2 > 0:
                    semantic_score = float(dot_product / (norm1 * norm2))

            # Combined score - rebalanced to prioritize entity+time over semantic
            # Page embeddings differ from event narrative embeddings even for same event
            # Entity overlap + temporal proximity are more reliable "same event" signals
            #
            # Adaptive weighting: if time or semantic scores are missing (0.0),
            # redistribute their weight to entity overlap for robustness
            if time_score == 0.0 and semantic_score == 0.0:
                # Both missing - rely entirely on entity overlap
                match_score = entity_overlap_score
            elif time_score == 0.0:
                # Time missing - split its weight to entity and semantic
                match_score = 0.55 * entity_overlap_score + 0.45 * semantic_score
            elif semantic_score == 0.0:
                # Semantic missing - split its weight to entity and time
                match_score = 0.55 * entity_overlap_score + 0.45 * time_score
            else:
                # All signals present - use balanced weights
                match_score = (
                    0.40 * entity_overlap_score +
                    0.30 * time_score +
                    0.30 * semantic_score
                )

            logger.info(
                f"📊 Candidate: {event.canonical_name} - "
                f"entity={entity_overlap_score:.2f}, time={time_score:.2f}, "
                f"semantic={semantic_score:.2f}, total={match_score:.2f}"
            )

            candidates.append((event, match_score))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Found {len(candidates)} candidate events (best: {candidates[0][1]:.2f})" if candidates else "No candidates found")
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

    async def link_claim(self, event: Event, claim, relationship_type: str = "SUPPORTS") -> None:
        """
        Link a claim to an event in Neo4j knowledge graph

        Args:
            event: Event domain model
            claim: Claim domain model
            relationship_type: Type of relationship (SUPPORTS, CONTRADICTS, UPDATES)

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
        result = await self.neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[r:SUPPORTS|CONTRADICTS|UPDATES]->(c:Claim)
            RETURN c.id as id, c.page_id as page_id, c.text as text,
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
            MATCH (e:Event {id: $event_id})-[r:SUPPORTS|CONTRADICTS|UPDATES]->(c:Claim)
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
        Update plausibility score on SUPPORTS relationship between event and claim.

        This stores Bayesian posterior as a property on the relationship,
        allowing queries like "get high-plausibility claims for event".

        Args:
            event_id: Event ID (ev_xxxxxxxx format)
            claim_id: Claim ID (cl_xxxxxxxx format)
            plausibility: Posterior probability (0.0-1.0)
        """
        await self.neo4j._execute_write("""
            MATCH (e:Event {id: $event_id})-[r:SUPPORTS]->(c:Claim {id: $claim_id})
            SET r.plausibility = $plausibility,
                r.plausibility_updated = datetime()
        """, {
            'event_id': event_id,
            'claim_id': claim_id,
            'plausibility': plausibility
        })

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
            WHERE NOT exists((e)<-[:CONTAINS]-())
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
            OPTIONAL MATCH (e)-[:CONTAINS]->(sub:Event)
            WITH e, count(sub) as child_count
            RETURN e.id as id,
                   e.canonical_name as canonical_name,
                   e.event_type as event_type,
                   e.status as status,
                   e.confidence as confidence,
                   e.event_scale as event_scale,
                   coalesce(e.event_start, e.earliest_time) as event_start,
                   coalesce(e.event_end, e.latest_time) as event_end,
                   e.created_at as created_at,
                   e.updated_at as updated_at,
                   e.metadata as metadata,
                   child_count
            ORDER BY e.created_at DESC
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
                'event_scale': row.get('event_scale'),
                'event_start': neo4j_datetime_to_python(row['event_start']) if row.get('event_start') else None,
                'event_end': neo4j_datetime_to_python(row['event_end']) if row.get('event_end') else None,
                'created_at': neo4j_datetime_to_python(row['created_at']) if row.get('created_at') else None,
                'updated_at': updated_at_str,
                'last_updated': updated_at_str,  # Frontend expects 'last_updated'
                'metadata': row.get('metadata', {}),
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

    async def update_narrative(self, event_id: str, narrative: str) -> None:
        """
        Update event narrative/summary.

        Args:
            event_id: Event ID
            narrative: New narrative text
        """
        await self.neo4j._execute_write("""
            MATCH (e:Event {id: $event_id})
            SET e.summary = $narrative,
                e.updated_at = datetime()
        """, {
            'event_id': event_id,
            'narrative': narrative
        })

        logger.debug(f"📝 Updated narrative for {event_id}: {len(narrative)} chars")

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

