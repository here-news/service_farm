"""
Event Repository - Neo4j Primary Storage

Storage strategy:
- Neo4j: Event nodes with ALL metadata, relationships (PRIMARY SOURCE OF TRUTH)
- PostgreSQL: Event embeddings ONLY (core.event_embeddings for vector search)

Architecture:
- create(): Write to Neo4j + embedding to PostgreSQL
- get_by_id(): Read from Neo4j, join embedding from PostgreSQL
- find_candidates(): Vector search on PostgreSQL, hydrate from Neo4j
"""
import uuid
import logging
import json
from typing import Optional, List, Set, Tuple
from datetime import datetime
import asyncpg
import numpy as np

from models.event import Event
from services.neo4j_service import Neo4jService
from utils.datetime_utils import neo4j_datetime_to_python

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
        # Prepare metadata with claim_ids, summary, location
        metadata = event.metadata.copy() if event.metadata else {}
        if event.claim_ids:
            metadata['claim_ids'] = [str(cid) for cid in event.claim_ids]
        if event.summary:
            metadata['summary'] = event.summary
        if event.location:
            metadata['location'] = event.location
        if event.coherence:
            metadata['coherence'] = event.coherence

        # Store ALL event data in Neo4j (primary storage)
        await self.neo4j.create_event(
            event_id=str(event.id),
            canonical_name=event.canonical_name,
            event_type=event.event_type,
            status=event.status,
            confidence=event.confidence,
            event_scale=event.event_scale,
            earliest_time=event.event_start,
            latest_time=event.event_end,
            metadata=metadata
        )

        # Store ONLY embedding in PostgreSQL (for vector similarity)
        if event.embedding:
            embedding_str = '[' + ','.join(str(x) for x in event.embedding) + ']'
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO core.event_embeddings (event_id, embedding)
                    VALUES ($1, $2::vector)
                    ON CONFLICT (event_id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        updated_at = NOW()
                """, event.id, embedding_str)

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
        metadata = event.metadata.copy() if event.metadata else {}
        if event.claim_ids:
            metadata['claim_ids'] = [str(cid) for cid in event.claim_ids]
        if event.summary:
            metadata['summary'] = event.summary
        if event.location:
            metadata['location'] = event.location
        if event.coherence:
            metadata['coherence'] = event.coherence

        # JSON serialize metadata for Neo4j (doesn't support nested structures)
        metadata_json = json.dumps(metadata) if metadata else '{}'

        await self.neo4j._execute_write("""
            MATCH (e:Event {id: $event_id})
            SET e.canonical_name = $canonical_name,
                e.event_type = $event_type,
                e.status = $status,
                e.confidence = $confidence,
                e.event_scale = $event_scale,
                e.earliest_time = $earliest_time,
                e.latest_time = $latest_time,
                e.metadata = $metadata,
                e.coherence = $coherence,
                e.updated_at = datetime()
        """, {
            'event_id': str(event.id),
            'canonical_name': event.canonical_name,
            'event_type': event.event_type,
            'status': event.status,
            'confidence': event.confidence,
            'event_scale': event.event_scale,
            'earliest_time': event.event_start,
            'latest_time': event.event_end,
            'metadata': metadata_json,
            'coherence': event.coherence
        })

        # Update embedding in PostgreSQL
        if event.embedding:
            embedding_str = '[' + ','.join(str(x) for x in event.embedding) + ']'
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE core.event_embeddings
                    SET embedding = $2::vector, updated_at = NOW()
                    WHERE event_id = $1
                """, event.id, embedding_str)

        logger.debug(f"📊 Updated event: {event.canonical_name}")
        return event

    async def get_by_id(self, event_id: uuid.UUID) -> Optional[Event]:
        """
        Retrieve event from Neo4j by ID

        Args:
            event_id: Event UUID

        Returns:
            Event model or None
        """
        # Fetch event from Neo4j (primary source)
        result = await self.neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})
            RETURN e
        """, {'event_id': str(event_id)})

        if not result:
            return None

        node = result[0]['e']

        # Parse metadata (may contain claim_ids)
        metadata = node.get('metadata_json', {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata) if metadata else {}

        # Extract claim_ids from metadata
        claim_ids = []
        if 'claim_ids' in metadata:
            claim_ids = [uuid.UUID(cid) if isinstance(cid, str) else cid
                        for cid in metadata.get('claim_ids', [])]

        # Get parent_event_id from Neo4j :CONTAINS relationships
        parent_event_id = await self.neo4j.get_parent_event_id(event_id=str(event_id))

        # Fetch embedding from PostgreSQL
        embedding = None
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT embedding FROM core.event_embeddings WHERE event_id = $1
            """, event_id)
            if row and row['embedding']:
                embedding = self._parse_embedding(row['embedding'])

        return Event(
            id=uuid.UUID(node['id']),
            canonical_name=node.get('canonical_name', ''),
            event_type=node.get('event_type', 'INCIDENT'),
            event_start=neo4j_datetime_to_python(node.get('earliest_time')),
            event_end=neo4j_datetime_to_python(node.get('latest_time')),
            status=node.get('status', 'provisional'),
            confidence=node.get('confidence', 0.3),
            event_scale=node.get('event_scale', 'micro'),
            embedding=embedding,
            metadata=metadata,
            parent_event_id=uuid.UUID(parent_event_id) if parent_event_id else None,
            claim_ids=claim_ids,
            pages_count=0,  # Derivable from claims
            claims_count=len(claim_ids),
            summary=metadata.get('summary'),
            location=metadata.get('location'),
            coherence=metadata.get('coherence', 0.3)
        )

    async def get_sub_events(self, parent_id: uuid.UUID) -> List[Event]:
        """
        Get all sub-events for a parent event

        Uses Neo4j :CONTAINS relationships as source of truth

        Args:
            parent_id: Parent event UUID

        Returns:
            List of sub-events
        """
        # Get sub-event IDs from Neo4j relationships
        sub_event_ids = await self.neo4j.get_sub_event_ids(event_id=str(parent_id))

        if not sub_event_ids:
            return []

        # Fetch each sub-event
        sub_events = []
        for sub_event_id in sub_event_ids:
            sub_event = await self.get_by_id(uuid.UUID(sub_event_id))
            if sub_event:
                sub_events.append(sub_event)

        return sub_events

    async def find_candidates(
        self,
        entity_ids: Set[uuid.UUID],
        reference_time: datetime,
        time_window_days: int = 7,
        page_embedding: Optional[List[float]] = None
    ) -> List[Tuple[Event, float]]:
        """
        Find candidate events that might match new information

        Scoring:
        - Entity overlap: 40%
        - Time proximity: 30%
        - Semantic similarity: 30%

        Args:
            entity_ids: Entity IDs to match against
            reference_time: Reference time for temporal proximity
            time_window_days: Time window in days
            page_embedding: Optional embedding for semantic matching

        Returns:
            List of (event, match_score) sorted by score descending
        """
        if not entity_ids:
            return []

        # Get events that share entities (from Neo4j)
        entity_id_strings = [str(eid) for eid in entity_ids]

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
            event_id = uuid.UUID(row['event_id'])
            entity_overlap_count = row['overlap_count']

            # Get full event from Neo4j
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
                # Defensive conversion for event.event_start
                event_start_py = neo4j_datetime_to_python(event.event_start)
                if event_start_py:
                    time_diff_days = abs((reference_time - event_start_py).days)
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

            # Combined score
            match_score = (
                0.4 * entity_overlap_score +
                0.3 * time_score +
                0.3 * semantic_score
            )

            logger.debug(
                f"Candidate: {event.canonical_name} - "
                f"entity={entity_overlap_score:.2f}, time={time_score:.2f}, "
                f"semantic={semantic_score:.2f}, total={match_score:.2f}"
            )

            candidates.append((event, match_score))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Found {len(candidates)} candidate events (best: {candidates[0][1]:.2f})" if candidates else "No candidates found")
        return candidates

    async def _get_event_entity_ids(self, event_id: uuid.UUID) -> Set[uuid.UUID]:
        """Get all entity IDs for an event from Neo4j"""
        result = await self.neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[:INVOLVES]->(entity:Entity)
            RETURN entity.id as entity_id
        """, {'event_id': str(event_id)})

        return {uuid.UUID(row['entity_id']) for row in result}

    def _parse_embedding(self, emb_str: str) -> Optional[List[float]]:
        """Parse embedding from PostgreSQL vector string"""
        try:
            if emb_str.startswith('[') and emb_str.endswith(']'):
                return [float(x.strip()) for x in emb_str[1:-1].split(',')]
        except Exception as e:
            logger.warning(f"Failed to parse embedding: {e}")
        return None

