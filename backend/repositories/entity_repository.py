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

    async def get_by_id(self, entity_id: uuid.UUID) -> Optional[Entity]:
        """
        Retrieve entity by ID from Neo4j

        Args:
            entity_id: Entity UUID

        Returns:
            Entity model or None
        """
        entity_data = await self.neo4j.get_entity_by_id(entity_id=str(entity_id))

        if not entity_data:
            return None

        # Convert Neo4j data to Entity model
        return Entity(
            id=uuid.UUID(entity_data['id']),
            canonical_name=entity_data['canonical_name'],
            entity_type=entity_data['entity_type'],
            mention_count=entity_data.get('mention_count', 0),
            profile_summary=entity_data.get('profile_summary'),
            wikidata_qid=entity_data.get('wikidata_qid'),
            wikidata_label=entity_data.get('wikidata_label'),
            wikidata_description=entity_data.get('wikidata_description'),
            status=entity_data.get('status', 'pending'),
            confidence=entity_data.get('confidence', 0.0),
            aliases=entity_data.get('aliases', []),
            metadata={}
        )

    async def get_by_ids(self, entity_ids: List[uuid.UUID]) -> List[Entity]:
        """
        Retrieve multiple entities by IDs from Neo4j

        Args:
            entity_ids: List of Entity UUIDs

        Returns:
            List of Entity models
        """
        if not entity_ids:
            return []

        # Convert UUIDs to strings
        id_strings = [str(eid) for eid in entity_ids]

        # Query Neo4j
        entities_data = await self.neo4j.get_entities_by_ids(entity_ids=id_strings)

        # Convert to Entity models
        entities = []
        for entity_data in entities_data:
            entities.append(Entity(
                id=uuid.UUID(entity_data['id']),
                canonical_name=entity_data['canonical_name'],
                entity_type=entity_data['entity_type'],
                mention_count=entity_data.get('mention_count', 0),
                profile_summary=entity_data.get('profile_summary'),
                wikidata_qid=entity_data.get('wikidata_qid'),
                wikidata_label=entity_data.get('wikidata_label'),
                wikidata_description=entity_data.get('wikidata_description'),
                status=entity_data.get('status', 'pending'),
                confidence=entity_data.get('confidence', 0.0),
                aliases=entity_data.get('aliases', []),
                metadata={}
            ))

        return entities

    async def get_by_event_id(self, event_id: uuid.UUID) -> List[Entity]:
        """
        Retrieve all entities for an event from Neo4j

        Args:
            event_id: Event UUID

        Returns:
            List of Entity models involved in the event
        """
        # Query Neo4j for entities linked to this event
        results = await self.neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[:INVOLVES]->(entity:Entity)
            RETURN entity.id as id,
                   entity.canonical_name as canonical_name,
                   entity.entity_type as entity_type,
                   entity.mention_count as mention_count,
                   entity.confidence as confidence,
                   entity.wikidata_qid as wikidata_qid,
                   entity.wikidata_label as wikidata_label,
                   entity.wikidata_description as wikidata_description,
                   entity.profile_summary as profile_summary,
                   entity.status as status,
                   entity.aliases as aliases
            ORDER BY entity.canonical_name
        """, {'event_id': str(event_id)})

        # Convert to Entity models
        entities = []
        for row in results:
            entities.append(Entity(
                id=uuid.UUID(row['id']),
                canonical_name=row['canonical_name'],
                entity_type=row['entity_type'],
                mention_count=row.get('mention_count', 0),
                profile_summary=row.get('profile_summary'),
                wikidata_qid=row.get('wikidata_qid'),
                wikidata_label=row.get('wikidata_label'),
                wikidata_description=row.get('wikidata_description'),
                status=row.get('status', 'pending'),
                confidence=row.get('confidence', 0.0),
                aliases=row.get('aliases', []),
                metadata={}
            ))

        return entities

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
            wikidata_qid=entity_data.get('wikidata_qid'),
            status=entity_data.get('status', 'pending'),
            aliases=[],
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

    async def enrich(
        self,
        entity_id: uuid.UUID,
        wikidata_qid: str,
        wikidata_label: str,
        wikidata_description: str,
        confidence: float,
        aliases: list = None,
        metadata: dict = None
    ) -> None:
        """
        Enrich entity with Wikidata information

        Args:
            entity_id: Entity UUID
            wikidata_qid: Wikidata QID (e.g., 'Q123')
            wikidata_label: Official Wikidata label
            wikidata_description: Wikidata description
            confidence: Confidence score of enrichment
            aliases: List of alternative names
            metadata: Additional metadata (thumbnail, coordinates, etc.)
        """
        await self.neo4j.enrich_entity(
            entity_id=str(entity_id),
            wikidata_qid=wikidata_qid,
            wikidata_label=wikidata_label,
            wikidata_description=wikidata_description,
            confidence=confidence,
            aliases=aliases or [],
            metadata=metadata or {}
        )
        logger.info(f"📦 Enriched entity {entity_id} with Wikidata QID {wikidata_qid}")

    async def mark_checked(self, entity_id: uuid.UUID) -> None:
        """
        Mark entity as checked (no Wikidata match found)

        Args:
            entity_id: Entity UUID
        """
        await self.neo4j.mark_entity_checked(entity_id=str(entity_id))
        logger.debug(f"✓ Marked entity {entity_id} as checked")

    async def update_profile(self, entity_id: uuid.UUID, profile_summary: str) -> None:
        """
        Update entity profile summary (AI-generated from claim contexts)

        Args:
            entity_id: Entity UUID
            profile_summary: Generated description
        """
        await self.neo4j.update_entity_profile(
            entity_id=str(entity_id),
            profile_summary=profile_summary
        )
        logger.debug(f"📝 Updated profile for entity {entity_id}")
