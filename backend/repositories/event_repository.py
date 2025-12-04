"""
Event Repository - Dual-write to PostgreSQL and Neo4j

Storage strategy:
- PostgreSQL: Event metadata, embedding, counts (core.events table)
- Neo4j: Event nodes with relationships to phases, entities
- Both kept in sync
"""
import uuid
import logging
import json
from typing import Optional, List
import asyncpg
import numpy as np

from models.event import Event
from services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)


class EventRepository:
    """
    Repository for Event domain model

    Handles dual-write to PostgreSQL (metadata+embedding) and Neo4j (graph)
    """

    def __init__(self, db_pool: asyncpg.Pool, neo4j_service: Neo4jService):
        self.db_pool = db_pool
        self.neo4j = neo4j_service

    async def create(self, event: Event) -> Event:
        """
        Create event in both PostgreSQL and Neo4j

        Args:
            event: Event domain model

        Returns:
            Created event with timestamps
        """
        # Convert embedding to pgvector format
        embedding_str = None
        if event.embedding:
            embedding_str = '[' + ','.join(str(x) for x in event.embedding) + ']'

        # Create in Neo4j (graph structure)
        await self.neo4j.create_event(
            event_id=str(event.id),
            canonical_name=event.title,
            event_type=event.event_type,
            status=event.status,
            confidence=event.confidence,
            event_scale=event.event_scale,
            earliest_time=event.event_start,
            latest_time=event.event_end,
            metadata=event.metadata
        )

        # Create in PostgreSQL (metadata + embedding)
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO core.events (
                    id, title, event_type, event_start, event_end,
                    status, confidence, event_scale, embedding, metadata,
                    pages_count, claims_count, summary, location
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::vector, $10, $11, $12, $13, $14)
            """,
                event.id,
                event.title,
                event.event_type,
                event.event_start,
                event.event_end,
                event.status,
                event.confidence,
                event.event_scale,
                embedding_str,
                json.dumps(event.metadata) if event.metadata else '{}',
                event.pages_count,
                event.claims_count,
                event.summary,
                event.location
            )

            # Fetch timestamps
            row = await conn.fetchrow("""
                SELECT created_at, updated_at FROM core.events WHERE id = $1
            """, event.id)

            event.created_at = row['created_at']
            event.updated_at = row['updated_at']

        logger.info(f"✨ Created event: {event.title} ({event.id})")
        return event

    async def update(self, event: Event) -> Event:
        """
        Update event in both PostgreSQL and Neo4j

        Args:
            event: Event domain model with updated fields

        Returns:
            Updated event
        """
        # Convert embedding to pgvector format
        embedding_str = None
        if event.embedding:
            embedding_str = '[' + ','.join(str(x) for x in event.embedding) + ']'

        # Update PostgreSQL
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE core.events
                SET
                    title = $2,
                    event_type = $3,
                    event_start = $4,
                    event_end = $5,
                    status = $6,
                    confidence = $7,
                    event_scale = $8,
                    embedding = $9::vector,
                    metadata = $10,
                    pages_count = $11,
                    claims_count = $12,
                    summary = $13,
                    location = $14,
                    updated_at = NOW()
                WHERE id = $1
            """,
                event.id,
                event.title,
                event.event_type,
                event.event_start,
                event.event_end,
                event.status,
                event.confidence,
                event.event_scale,
                embedding_str,
                json.dumps(event.metadata) if event.metadata else '{}',
                event.pages_count,
                event.claims_count,
                event.summary,
                event.location
            )

            # Fetch updated timestamp
            row = await conn.fetchrow("""
                SELECT updated_at FROM core.events WHERE id = $1
            """, event.id)

            event.updated_at = row['updated_at']

        # Update Neo4j (status, confidence)
        # Note: Neo4j updates are handled by event_worker for now

        logger.debug(f"📊 Updated event: {event.title}")
        return event

    async def get_by_id(self, event_id: uuid.UUID) -> Optional[Event]:
        """
        Retrieve event from PostgreSQL by ID

        Args:
            event_id: Event UUID

        Returns:
            Event model or None
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT
                    id, title, event_type, event_start, event_end,
                    status, confidence, event_scale, embedding,
                    metadata, pages_count, claims_count, summary, location,
                    created_at, updated_at
                FROM core.events
                WHERE id = $1
            """, event_id)

            if not row:
                return None

            # Parse embedding
            embedding = None
            if row['embedding']:
                embedding = self._parse_embedding(row['embedding'])

            return Event(
                id=row['id'],
                title=row['title'],
                event_type=row['event_type'],
                event_start=row['event_start'],
                event_end=row['event_end'],
                status=row['status'],
                confidence=row['confidence'],
                event_scale=row['event_scale'],
                embedding=embedding,
                metadata=row['metadata'] or {},
                pages_count=row['pages_count'],
                claims_count=row['claims_count'],
                summary=row['summary'],
                location=row['location'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )

    def _parse_embedding(self, emb_str: str) -> Optional[List[float]]:
        """Parse embedding from PostgreSQL vector string"""
        try:
            if emb_str.startswith('[') and emb_str.endswith(']'):
                return [float(x.strip()) for x in emb_str[1:-1].split(',')]
        except Exception as e:
            logger.warning(f"Failed to parse embedding: {e}")
        return None
