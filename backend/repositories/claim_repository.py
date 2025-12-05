"""
Claim Repository - PostgreSQL and Neo4j operations

Storage strategy:
- PostgreSQL: Claim content, embedding, entity_ids in metadata JSON (core.claims table)
- Neo4j: Entity nodes fetched by entity_ids
- NO junction tables - entity references stored in claim.metadata
"""
import uuid
import logging
import json
from typing import Optional, List
import asyncpg

from models.claim import Claim
from models.entity import Entity
from services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)


class ClaimRepository:
    """
    Repository for Claim domain model

    Handles storage in PostgreSQL (content+embedding) and Neo4j (graph)
    """

    def __init__(self, db_pool: asyncpg.Pool, neo4j_service: Neo4jService):
        self.db_pool = db_pool
        self.neo4j = neo4j_service

    async def create(self, claim: Claim, entity_ids: List[uuid.UUID] = None, entity_names: List[str] = None) -> Claim:
        """
        Create claim in PostgreSQL with entity references in metadata

        Args:
            claim: Claim domain model
            entity_ids: List of entity UUIDs to store in metadata
            entity_names: List of entity names (for debugging)

        Returns:
            Created claim with timestamp
        """
        # Add entity references to metadata
        metadata = claim.metadata.copy()
        if entity_ids:
            metadata['entity_ids'] = [str(eid) for eid in entity_ids]
            if entity_names:
                metadata['entity_names'] = entity_names

        # NOTE: Embeddings are NOT stored in database
        # They are generated on-demand from claim.text when needed
        # This saves storage and embeddings can always be regenerated

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO core.claims (
                    id, page_id, text, event_time, confidence, modality, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
                claim.id,
                claim.page_id,
                claim.text,
                claim.event_time,
                claim.confidence,
                claim.modality,
                json.dumps(metadata)
            )

            # Fetch timestamp
            row = await conn.fetchrow("""
                SELECT created_at FROM core.claims WHERE id = $1
            """, claim.id)

            claim.created_at = row['created_at']
            claim.metadata = metadata  # Update claim with entity references

        logger.debug(f"📝 Created claim: {claim.text[:50]}... (entities: {len(entity_ids or [])})")
        return claim

    async def hydrate_entities(self, claim: Claim) -> Claim:
        """
        Fetch and attach entities from Neo4j based on claim.metadata.entity_ids

        Args:
            claim: Claim with entity_ids in metadata

        Returns:
            Claim with entities populated
        """
        entity_ids = claim.entity_ids
        if not entity_ids:
            claim.entities = []
            return claim

        # Fetch entities from Neo4j
        entity_id_strings = [str(eid) for eid in entity_ids]
        entities_data = await self.neo4j.get_entities_by_ids(entity_ids=entity_id_strings)

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
                aliases=entity_data.get('aliases', []),
                status=entity_data.get('status', 'pending'),
                confidence=entity_data.get('confidence', 0.0)
            ))

        claim.entities = entities
        logger.debug(f"✅ Hydrated {len(entities)} entities for claim {claim.id}")
        return claim

    async def get_by_id(self, claim_id: uuid.UUID) -> Optional[Claim]:
        """
        Retrieve claim from PostgreSQL by ID

        Args:
            claim_id: Claim UUID

        Returns:
            Claim model or None
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT
                    id, page_id, text, event_time, confidence, modality,
                    metadata, embedding, created_at
                FROM core.claims
                WHERE id = $1
            """, claim_id)

            if not row:
                return None

            # Parse embedding if exists
            embedding = None
            if row['embedding']:
                embedding = self._parse_embedding(row['embedding'])

            # Parse metadata if it's a string (from JSON column)
            metadata = row['metadata'] or {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            return Claim(
                id=row['id'],
                page_id=row['page_id'],
                text=row['text'],
                event_time=row['event_time'],
                confidence=row['confidence'],
                modality=row['modality'],
                metadata=metadata,
                embedding=embedding,
                created_at=row['created_at']
            )

    async def get_by_page(self, page_id: uuid.UUID) -> List[Claim]:
        """
        Retrieve all claims for a page

        Args:
            page_id: Page UUID

        Returns:
            List of Claim models
        """
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    id, page_id, text, event_time, confidence, modality,
                    metadata, embedding, created_at
                FROM core.claims
                WHERE page_id = $1
                ORDER BY created_at
            """, page_id)

            claims = []
            for row in rows:
                embedding = None
                if row['embedding']:
                    embedding = self._parse_embedding(row['embedding'])

                # Parse metadata if it's a string
                metadata = row['metadata'] or {}
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                claims.append(Claim(
                    id=row['id'],
                    page_id=row['page_id'],
                    text=row['text'],
                    event_time=row['event_time'],
                    confidence=row['confidence'],
                    modality=row['modality'],
                    metadata=metadata,
                    embedding=embedding,
                    created_at=row['created_at']
                ))

            return claims

    async def get_entities_for_claim(self, claim_id: uuid.UUID) -> List[Entity]:
        """
        Get all entities linked to a claim (from metadata + Neo4j)

        Args:
            claim_id: Claim UUID

        Returns:
            List of Entity objects
        """
        claim = await self.get_by_id(claim_id)
        if not claim:
            return []

        await self.hydrate_entities(claim)
        return claim.entities or []

    def _parse_embedding(self, emb_str: str) -> Optional[List[float]]:
        """Parse embedding from PostgreSQL vector string"""
        try:
            if emb_str.startswith('[') and emb_str.endswith(']'):
                return [float(x.strip()) for x in emb_str[1:-1].split(',')]
        except Exception as e:
            logger.warning(f"Failed to parse embedding: {e}")
        return None
