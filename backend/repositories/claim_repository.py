"""
Claim Repository - PostgreSQL and Neo4j operations

Storage strategy:
- PostgreSQL: Claim content, embedding (core.claims table)
- Neo4j: Claim nodes with relationships to entities and phases
- claim_entities join table for PostgreSQL queries
"""
import uuid
import logging
import json
from typing import Optional, List
import asyncpg

from models.claim import Claim
from models.relationships import ClaimEntityLink
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

    async def create(self, claim: Claim) -> Claim:
        """
        Create claim in PostgreSQL

        Neo4j claim nodes are created by event worker when linking to phases

        Args:
            claim: Claim domain model

        Returns:
            Created claim with timestamp
        """
        # Convert embedding to pgvector format
        embedding_str = None
        if claim.embedding:
            embedding_str = '[' + ','.join(str(x) for x in claim.embedding) + ']'

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO core.claims (
                    id, page_id, text, event_time, confidence, modality, metadata, embedding
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::vector)
            """,
                claim.id,
                claim.page_id,
                claim.text,
                claim.event_time,
                claim.confidence,
                claim.modality,
                json.dumps(claim.metadata) if claim.metadata else '{}',
                embedding_str
            )

            # Fetch timestamp
            row = await conn.fetchrow("""
                SELECT created_at FROM core.claims WHERE id = $1
            """, claim.id)

            claim.created_at = row['created_at']

        logger.debug(f"📝 Created claim: {claim.text[:50]}...")
        return claim

    async def link_to_entity(
        self, claim_id: uuid.UUID, entity_id: uuid.UUID, relationship_type: str = 'MENTIONS'
    ) -> ClaimEntityLink:
        """
        Link claim to entity in both PostgreSQL and Neo4j

        Args:
            claim_id: Claim UUID
            entity_id: Entity UUID
            relationship_type: MENTIONS, ACTOR, SUBJECT, LOCATION

        Returns:
            ClaimEntityLink relationship
        """
        # Store in PostgreSQL join table
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO core.claim_entities (claim_id, entity_id, relationship_type)
                VALUES ($1, $2, $3)
                ON CONFLICT (claim_id, entity_id) DO UPDATE SET
                    relationship_type = EXCLUDED.relationship_type
            """, claim_id, entity_id, relationship_type)

        # Neo4j relationship created by event worker via neo4j.link_claim_to_entity()

        logger.debug(f"🔗 Linked claim {claim_id} → {relationship_type} → entity {entity_id}")

        return ClaimEntityLink(
            claim_id=claim_id,
            entity_id=entity_id,
            relationship_type=relationship_type
        )

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

            return Claim(
                id=row['id'],
                page_id=row['page_id'],
                text=row['text'],
                event_time=row['event_time'],
                confidence=row['confidence'],
                modality=row['modality'],
                metadata=row['metadata'] or {},
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

                claims.append(Claim(
                    id=row['id'],
                    page_id=row['page_id'],
                    text=row['text'],
                    event_time=row['event_time'],
                    confidence=row['confidence'],
                    modality=row['modality'],
                    metadata=row['metadata'] or {},
                    embedding=embedding,
                    created_at=row['created_at']
                ))

            return claims

    async def get_entities_for_claim(self, claim_id: uuid.UUID) -> List[tuple]:
        """
        Get all entities linked to a claim

        Args:
            claim_id: Claim UUID

        Returns:
            List of (entity_id, relationship_type) tuples
        """
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT entity_id, relationship_type
                FROM core.claim_entities
                WHERE claim_id = $1
            """, claim_id)

            return [(row['entity_id'], row['relationship_type']) for row in rows]

    def _parse_embedding(self, emb_str: str) -> Optional[List[float]]:
        """Parse embedding from PostgreSQL vector string"""
        try:
            if emb_str.startswith('[') and emb_str.endswith(']'):
                return [float(x.strip()) for x in emb_str[1:-1].split(',')]
        except Exception as e:
            logger.warning(f"Failed to parse embedding: {e}")
        return None
