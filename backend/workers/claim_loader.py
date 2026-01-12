"""
Claim Loader Utility
====================

Utility functions for loading claims with embeddings.
Used by canonical_worker.py.
"""

import logging
from typing import List

from repositories.claim_repository import ClaimRepository
from services.neo4j_service import Neo4jService
from reee.types import Claim

logger = logging.getLogger(__name__)


async def load_claims_with_embeddings(
    claim_repo: ClaimRepository,
    neo4j: Neo4jService,
    limit: int = None
) -> List[Claim]:
    """
    Load claims from Neo4j with entities, then fetch embeddings from PostgreSQL.

    Uses the proper data layer:
    - Neo4j for claim nodes and entity relationships
    - PostgreSQL for embeddings via ClaimRepository
    """
    logger.info("Loading claims from Neo4j...")

    # Build query for claims with entities
    query = '''
        MATCH (c:Claim)
        OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
        WITH c, collect(DISTINCT e.canonical_name) as entities
        OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
        RETURN c.id as id,
               c.text as text,
               c.event_time as event_time,
               c.created_at as created_at,
               p.domain as source,
               p.id as page_id,
               entities
    '''
    if limit:
        query += f' LIMIT {limit}'

    results = await neo4j._execute_read(query)

    claims = []
    for row in results:
        entities = set(e for e in (row['entities'] or []) if e)

        claim = Claim(
            id=row['id'],
            text=row['text'] or "",
            source=row['source'] or "unknown",
            page_id=row['page_id'],
            entities=entities,
            anchor_entities=entities,
            event_time=row['event_time'],
            timestamp=row['created_at']
        )
        claims.append(claim)

    logger.info(f"Loaded {len(claims)} claims")

    # Fetch embeddings from PostgreSQL in batch (much faster than per-claim)
    logger.info("Fetching embeddings from PostgreSQL (batch)...")

    claim_ids = [c.id for c in claims]
    claim_lookup = {c.id: c for c in claims}
    embeddings_found = 0

    # Batch fetch with single register_vector call
    from pgvector.asyncpg import register_vector

    async with claim_repo.db_pool.acquire() as conn:
        await register_vector(conn)

        # Batch query - fetch all embeddings in one query
        results = await conn.fetch("""
            SELECT claim_id, embedding
            FROM core.claim_embeddings
            WHERE claim_id = ANY($1)
        """, claim_ids)

        for row in results:
            claim = claim_lookup.get(row['claim_id'])
            if claim and row['embedding'] is not None:
                emb = row['embedding']
                if hasattr(emb, 'tolist'):
                    claim.embedding = emb.tolist()
                elif isinstance(emb, (list, tuple)):
                    claim.embedding = list(emb)
                embeddings_found += 1

    logger.info(f"Found {embeddings_found}/{len(claims)} embeddings")

    return claims
