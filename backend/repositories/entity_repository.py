"""
Entity Repository - Neo4j primary storage

Storage strategy:
- Neo4j: Primary entity storage (canonical_name, type, mention_count, created_at)
- PostgreSQL: Enrichment data only (Wikidata descriptions, external IDs)

Entities are fundamentally graph nodes, so Neo4j is the source of truth.
PostgreSQL only stores large text fields (descriptions) from enrichment workers.
"""
import uuid
import logging
from typing import Optional, List
import asyncpg

from models.entity import Entity
from services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)


class EntityRepository:
    """
    Repository for Entity domain model

    Neo4j is the primary storage for entities.
    PostgreSQL enrichment handled separately.
    """

    def __init__(self, db_pool: asyncpg.Pool, neo4j_service: Neo4jService):
        self.db_pool = db_pool
        self.neo4j = neo4j_service

    async def create(self, entity: Entity) -> Entity:
        """
        Create entity in Neo4j (primary storage)

        Uses MERGE for deduplication based on (canonical_name, entity_type)

        Args:
            entity: Entity domain model

        Returns:
            Created entity with id
        """
        # Create entity in Neo4j (primary storage)
        returned_id = await self.neo4j.create_or_update_entity(
            entity_id=str(entity.id),
            canonical_name=entity.canonical_name,
            entity_type=entity.entity_type
        )

        # Note: PostgreSQL enrichment (descriptions, Wikidata) handled by enrichment worker

        logger.debug(f"📦 Created entity in Neo4j: {entity.canonical_name} ({entity.entity_type})")
        return entity

    async def get_by_canonical_name(
        self, canonical_name: str, entity_type: Optional[str] = None
    ) -> Optional[Entity]:
        """
        Retrieve entity by canonical name from Neo4j

        Args:
            canonical_name: Entity canonical name
            entity_type: Required entity type

        Returns:
            Entity model or None
        """
        if not entity_type:
            logger.warning("get_by_canonical_name requires entity_type for Neo4j queries")
            return None

        # Query Neo4j
        entity_data = await self.neo4j.get_entity_by_name_and_type(
            canonical_name=canonical_name,
            entity_type=entity_type
        )

        if not entity_data:
            return None

        # Convert Neo4j data to Entity model
        return Entity(
            id=uuid.UUID(entity_data['id']),
            canonical_name=entity_data['canonical_name'],
            entity_type=entity_data['entity_type'],
            mention_count=entity_data.get('mention_count', 0),
            aliases=[],  # Aliases not stored in Neo4j base entity
            metadata={}
        )

    async def increment_mention_count(self, entity_id: uuid.UUID) -> int:
        """
        Increment mention count (handled automatically by Neo4j MERGE)

        This is a no-op since Neo4j create_or_update_entity increments on MATCH

        Args:
            entity_id: Entity UUID

        Returns:
            Updated mention count (always returns 0 as placeholder)
        """
        # Neo4j handles mention_count increment automatically in create_or_update_entity
        # This method exists for compatibility but is a no-op
        logger.debug(f"Mention count for {entity_id} incremented via Neo4j MERGE")
        return 0
