"""
Claim Repository - Neo4j storage for claims

Storage strategy:
- Neo4j: Claim nodes with text, confidence, event_time
- Neo4j: MENTIONS relationships to Entity nodes
- Neo4j: CONTAINS relationships from Page nodes
- PostgreSQL (optional): Claim embeddings for similarity search

The repository abstracts storage from consumers - they work with Claim domain model.

ID format: cl_xxxxxxxx (11 chars)
"""
import logging
from typing import Optional, List
import asyncpg
from datetime import datetime

from models.claim import Claim
from models.entity import Entity
from services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)


class ClaimRepository:
    """
    Repository for Claim domain model

    Neo4j is the primary store for claims.
    PostgreSQL only used for embeddings (optional).
    """

    def __init__(self, db_pool: asyncpg.Pool, neo4j_service: Neo4jService):
        self.db_pool = db_pool
        self.neo4j = neo4j_service

    # =========================================================================
    # CREATE OPERATIONS
    # =========================================================================

    async def create(
        self,
        claim: Claim,
        entity_ids: List[str] = None
    ) -> Claim:
        """
        Create claim in Neo4j with entity relationships.

        Args:
            claim: Claim domain model
            entity_ids: List of entity IDs (en_xxxxxxxx format) to link via MENTIONS

        Returns:
            Created claim with timestamp
        """
        # Create Claim node in Neo4j
        # Store deterministic_id for deduplication on reprocessing
        deterministic_id = claim.metadata.get('deterministic_id') if claim.metadata else None

        await self.neo4j._execute_write("""
            MERGE (c:Claim {id: $claim_id})
            ON CREATE SET
                c.text = $text,
                c.deterministic_id = $deterministic_id,
                c.event_time = $event_time,
                c.confidence = $confidence,
                c.modality = $modality,
                c.created_at = datetime()
            ON MATCH SET
                c.text = $text,
                c.confidence = $confidence
        """, {
            'claim_id': str(claim.id),
            'deterministic_id': deterministic_id,
            'text': claim.text[:500],  # Truncate for graph storage
            'event_time': claim.event_time.isoformat() if claim.event_time else None,
            'confidence': claim.confidence,
            'modality': claim.modality
        })

        # Create MENTIONS relationships to entities and increment mention_count
        if entity_ids:
            for entity_id in entity_ids:
                await self.neo4j._execute_write("""
                    MATCH (c:Claim {id: $claim_id})
                    MATCH (e:Entity {id: $entity_id})
                    MERGE (c)-[r:MENTIONS]->(e)
                    ON CREATE SET r.created_at = datetime(),
                                  e.mention_count = coalesce(e.mention_count, 0) + 1
                """, {
                    'claim_id': str(claim.id),
                    'entity_id': str(entity_id)
                })

        claim.created_at = datetime.utcnow()
        logger.debug(f"📝 Created claim: {claim.text[:50]}... ({len(entity_ids or [])} entities)")
        return claim

    async def link_to_page(self, claim_id: str, page_id: str) -> None:
        """
        Create CONTAINS relationship from Page to Claim.

        Args:
            claim_id: Claim ID (cl_xxxxxxxx format)
            page_id: Page ID (pg_xxxxxxxx format)
        """
        await self.neo4j._execute_write("""
            MATCH (p:Page {id: $page_id})
            MATCH (c:Claim {id: $claim_id})
            MERGE (p)-[r:CONTAINS]->(c)
            ON CREATE SET r.created_at = datetime()
        """, {
            'page_id': page_id,
            'claim_id': claim_id
        })

    # =========================================================================
    # READ OPERATIONS
    # =========================================================================

    async def get_by_id(self, claim_id: str) -> Optional[Claim]:
        """
        Retrieve claim by ID from Neo4j.

        Args:
            claim_id: Claim ID (cl_xxxxxxxx format)

        Returns:
            Claim model or None
        """
        results = await self.neo4j._execute_read("""
            MATCH (c:Claim {id: $claim_id})
            OPTIONAL MATCH (p:Page)-[:CONTAINS]->(c)
            RETURN c.id as id, c.text as text, c.event_time as event_time,
                   c.confidence as confidence, c.modality as modality,
                   c.created_at as created_at, p.id as page_id
        """, {'claim_id': claim_id})

        if not results:
            return None

        row = results[0]
        # Model handles UUID conversion in __post_init__
        return Claim(
            id=row['id'],
            page_id=row['page_id'],
            text=row['text'],
            event_time=row['event_time'],
            confidence=row['confidence'] or 0.8,
            modality=row['modality'] or 'observation',
            created_at=row['created_at']
        )

    async def get_by_page(self, page_id: str) -> List[Claim]:
        """
        Retrieve all claims for a page.

        Args:
            page_id: Page ID (pg_xxxxxxxx format)

        Returns:
            List of Claim models
        """
        results = await self.neo4j._execute_read("""
            MATCH (p:Page {id: $page_id})-[:CONTAINS]->(c:Claim)
            RETURN c.id as id, c.text as text, c.event_time as event_time,
                   c.confidence as confidence, c.modality as modality,
                   c.created_at as created_at
            ORDER BY c.created_at
        """, {'page_id': page_id})

        claims = []
        for row in results:
            # Model handles UUID conversion in __post_init__
            claims.append(Claim(
                id=row['id'],
                page_id=page_id,
                text=row['text'],
                event_time=row['event_time'],
                confidence=row['confidence'] or 0.8,
                modality=row['modality'] or 'observation',
                created_at=row['created_at']
            ))

        return claims

    async def get_entities_for_claim(self, claim_id: str) -> List[Entity]:
        """
        Get all entities mentioned by a claim.

        Args:
            claim_id: Claim ID (cl_xxxxxxxx format)

        Returns:
            List of Entity models
        """
        results = await self.neo4j._execute_read("""
            MATCH (c:Claim {id: $claim_id})-[:MENTIONS]->(e:Entity)
            RETURN e.id as id, e.canonical_name as canonical_name,
                   e.entity_type as entity_type, e.wikidata_qid as wikidata_qid,
                   e.wikidata_label as wikidata_label,
                   e.wikidata_description as wikidata_description,
                   e.mention_count as mention_count, e.confidence as confidence,
                   e.status as status, e.aliases as aliases
            ORDER BY e.canonical_name
        """, {'claim_id': claim_id})

        entities = []
        for row in results:
            # Model handles UUID conversion in __post_init__
            entities.append(Entity(
                id=row['id'],
                canonical_name=row['canonical_name'],
                entity_type=row['entity_type'],
                wikidata_qid=row.get('wikidata_qid'),
                wikidata_label=row.get('wikidata_label'),
                wikidata_description=row.get('wikidata_description'),
                mention_count=row.get('mention_count', 0),
                confidence=row.get('confidence', 0.0),
                status=row.get('status', 'pending'),
                aliases=row.get('aliases') or []
            ))

        return entities

    async def hydrate_entities(self, claim: Claim) -> Claim:
        """
        Fetch and attach entities from Neo4j.

        Args:
            claim: Claim to hydrate (uses claim.id in cl_xxxxxxxx format)

        Returns:
            Claim with entities populated
        """
        claim.entities = await self.get_entities_for_claim(str(claim.id))
        logger.debug(f"✅ Hydrated {len(claim.entities)} entities for claim {claim.id}")
        return claim

    # =========================================================================
    # EMBEDDING OPERATIONS (PostgreSQL)
    # =========================================================================

    async def store_embedding(
        self,
        claim_id: str,
        embedding: List[float]
    ) -> None:
        """
        Store claim embedding in PostgreSQL for similarity search.

        Args:
            claim_id: Claim ID (cl_xxxxxxxx format)
            embedding: Embedding vector
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO content.claim_embeddings (claim_id, embedding)
                VALUES ($1, $2)
                ON CONFLICT (claim_id) DO UPDATE SET embedding = $2
            """, claim_id, embedding)

    async def get_embedding(self, claim_id: str) -> Optional[List[float]]:
        """
        Get claim embedding from PostgreSQL.

        Args:
            claim_id: Claim ID (cl_xxxxxxxx format)

        Returns:
            Embedding vector or None
        """
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT embedding FROM content.claim_embeddings WHERE claim_id = $1
            """, claim_id)

            if result:
                # Parse vector string to list
                if isinstance(result, str) and result.startswith('['):
                    return [float(x.strip()) for x in result[1:-1].split(',')]
            return None

    # =========================================================================
    # QUERY OPERATIONS
    # =========================================================================

    async def find_similar(
        self,
        embedding: List[float],
        limit: int = 10,
        exclude_claim_ids: List[str] = None
    ) -> List[dict]:
        """
        Find claims similar to given embedding.

        Args:
            embedding: Query embedding vector
            limit: Maximum results
            exclude_claim_ids: Claim IDs (cl_xxxxxxxx format) to exclude

        Returns:
            List of {claim_id, similarity} dicts
        """
        exclude_ids = exclude_claim_ids or []

        async with self.db_pool.acquire() as conn:
            results = await conn.fetch("""
                SELECT claim_id, 1 - (embedding <=> $1) as similarity
                FROM content.claim_embeddings
                WHERE NOT (claim_id = ANY($3::text[]))
                ORDER BY embedding <=> $1
                LIMIT $2
            """, embedding, limit, exclude_ids)

            return [{'claim_id': r['claim_id'], 'similarity': r['similarity']} for r in results]

    async def get_claims_by_entity(self, entity_id: str) -> List[Claim]:
        """
        Get all claims that mention an entity.

        Args:
            entity_id: Entity ID (en_xxxxxxxx format)

        Returns:
            List of Claim models
        """
        results = await self.neo4j._execute_read("""
            MATCH (c:Claim)-[:MENTIONS]->(e:Entity {id: $entity_id})
            OPTIONAL MATCH (p:Page)-[:CONTAINS]->(c)
            RETURN c.id as id, c.text as text, c.event_time as event_time,
                   c.confidence as confidence, c.modality as modality,
                   c.created_at as created_at, p.id as page_id
            ORDER BY c.created_at DESC
        """, {'entity_id': entity_id})

        claims = []
        for row in results:
            # Model handles UUID conversion in __post_init__
            claims.append(Claim(
                id=row['id'],
                page_id=row['page_id'],
                text=row['text'],
                event_time=row['event_time'],
                confidence=row['confidence'] or 0.8,
                modality=row['modality'] or 'observation',
                created_at=row['created_at']
            ))

        return claims
