"""
Entity Repository - Dual-write to PostgreSQL and Neo4j

Storage strategy:
- PostgreSQL: Entity metadata, aliases (core.entities table)
- Neo4j: Entity nodes for graph relationships
- Both are kept in sync via this repository
"""
import uuid
import json
import logging
from typing import Optional, List
import asyncpg

from models.entity import Entity
from services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)


class EntityRepository:
    """
    Repository for Entity domain model

    Handles dual-write to PostgreSQL (metadata) and Neo4j (graph)
    """

    def __init__(self, db_pool: asyncpg.Pool, neo4j_service: Neo4jService):
        self.db_pool = db_pool
        self.neo4j = neo4j_service

    async def create(self, entity: Entity) -> Entity:
        """
        Create entity in PostgreSQL (Neo4j creation happens via event worker)

        Args:
            entity: Entity domain model

        Returns:
            Created entity with timestamps
        """
        # Extract metadata fields for PostgreSQL schema
        semantic_confidence = entity.metadata.get('semantic_confidence', 0.7)
        status = entity.metadata.get('status', 'stub')

        async with self.db_pool.acquire() as conn:
            # Insert into PostgreSQL with proper schema fields
            await conn.execute("""
                INSERT INTO core.entities (
                    id, canonical_name, entity_type, aliases, mention_count,
                    semantic_confidence, first_seen, last_seen, status, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, NOW(), NOW(), $7, $8)
                ON CONFLICT (id) DO UPDATE SET
                    canonical_name = EXCLUDED.canonical_name,
                    aliases = EXCLUDED.aliases,
                    mention_count = EXCLUDED.mention_count,
                    semantic_confidence = EXCLUDED.semantic_confidence,
                    last_seen = NOW(),
                    status = EXCLUDED.status,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
            """,
                entity.id,
                entity.canonical_name,
                entity.entity_type,
                entity.aliases,
                entity.mention_count,
                semantic_confidence,
                status,
                json.dumps(entity.metadata) if entity.metadata else '{}'
            )

            # Fetch timestamps
            row = await conn.fetchrow("""
                SELECT created_at, updated_at FROM core.entities WHERE id = $1
            """, entity.id)

            entity.created_at = row['created_at']
            entity.updated_at = row['updated_at']

        # Note: Neo4j entity nodes are created by event worker via link_claim_to_entity()
        # This allows proper relationship creation in the graph

        logger.debug(f"📦 Created entity: {entity.canonical_name} ({entity.entity_type})")
        return entity

    async def get_by_id(self, entity_id: uuid.UUID) -> Optional[Entity]:
        """
        Retrieve entity from PostgreSQL by ID

        Args:
            entity_id: Entity UUID

        Returns:
            Entity model or None if not found
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT
                    id, canonical_name, entity_type, aliases,
                    mention_count, metadata, created_at, updated_at
                FROM core.entities
                WHERE id = $1
            """, entity_id)

            if not row:
                return None

            return Entity(
                id=row['id'],
                canonical_name=row['canonical_name'],
                entity_type=row['entity_type'],
                aliases=list(row['aliases']) if row['aliases'] else [],
                mention_count=row['mention_count'] or 0,
                metadata=row['metadata'] or {},
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )

    async def get_by_canonical_name(
        self, canonical_name: str, entity_type: Optional[str] = None
    ) -> Optional[Entity]:
        """
        Retrieve entity by canonical name and optionally type

        Args:
            canonical_name: Entity canonical name
            entity_type: Optional entity type filter

        Returns:
            Entity model or None
        """
        async with self.db_pool.acquire() as conn:
            if entity_type:
                row = await conn.fetchrow("""
                    SELECT
                        id, canonical_name, entity_type, aliases,
                        mention_count, metadata, created_at, updated_at
                    FROM core.entities
                    WHERE canonical_name = $1 AND entity_type = $2
                """, canonical_name, entity_type)
            else:
                row = await conn.fetchrow("""
                    SELECT
                        id, canonical_name, entity_type, aliases,
                        mention_count, metadata, created_at, updated_at
                    FROM core.entities
                    WHERE canonical_name = $1
                """, canonical_name)

            if not row:
                return None

            return Entity(
                id=row['id'],
                canonical_name=row['canonical_name'],
                entity_type=row['entity_type'],
                aliases=list(row['aliases']) if row['aliases'] else [],
                mention_count=row['mention_count'] or 0,
                metadata=row['metadata'] or {},
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )

    async def increment_mention_count(self, entity_id: uuid.UUID) -> int:
        """
        Increment mention count for an entity

        Args:
            entity_id: Entity UUID

        Returns:
            New mention count
        """
        async with self.db_pool.acquire() as conn:
            new_count = await conn.fetchval("""
                UPDATE core.entities
                SET mention_count = mention_count + 1, updated_at = NOW()
                WHERE id = $1
                RETURNING mention_count
            """, entity_id)

            return new_count or 0

    async def batch_create(self, entities: List[Entity]) -> List[Entity]:
        """
        Batch create entities for efficiency

        Args:
            entities: List of Entity models

        Returns:
            List of created entities with timestamps
        """
        if not entities:
            return []

        async with self.db_pool.acquire() as conn:
            # Use COPY for bulk insert (most efficient)
            await conn.executemany("""
                INSERT INTO core.entities (
                    id, canonical_name, entity_type, aliases, mention_count, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (id) DO UPDATE SET
                    canonical_name = EXCLUDED.canonical_name,
                    aliases = EXCLUDED.aliases,
                    mention_count = EXCLUDED.mention_count,
                    updated_at = NOW()
            """, [
                (
                    e.id, e.canonical_name, e.entity_type,
                    e.aliases, e.mention_count, e.metadata
                )
                for e in entities
            ])

            # Fetch timestamps
            ids = [e.id for e in entities]
            rows = await conn.fetch("""
                SELECT id, created_at, updated_at
                FROM core.entities
                WHERE id = ANY($1::uuid[])
            """, ids)

            # Update timestamps
            timestamp_map = {r['id']: r for r in rows}
            for entity in entities:
                ts = timestamp_map.get(entity.id)
                if ts:
                    entity.created_at = ts['created_at']
                    entity.updated_at = ts['updated_at']

        logger.info(f"📦 Batch created {len(entities)} entities")
        return entities
