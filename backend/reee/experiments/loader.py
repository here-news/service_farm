"""
DB Loader for REEE Experiments
==============================

Loads real claims from Neo4j/Postgres with embeddings and entities.
"""

import os
import asyncio
from typing import Optional
from dataclasses import dataclass

import asyncpg
from pgvector.asyncpg import register_vector
from openai import AsyncOpenAI

from reee import Claim, Parameters, Engine


# Hub locations that should not count as anchor entities
HUB_LOCATIONS = {'Hong Kong', 'China', 'United States', 'UK', 'United Kingdom', 'US'}


@dataclass
class LoadedEvent:
    """An event loaded from the database."""
    id: str
    name: str
    claim_count: int


@dataclass
class ExperimentContext:
    """Shared context for experiments."""
    db_pool: asyncpg.Pool
    neo4j: 'Neo4jService'
    llm: AsyncOpenAI

    async def close(self):
        await self.db_pool.close()
        await self.neo4j.close()


async def create_context() -> ExperimentContext:
    """Create database connections for experiments."""
    from services.neo4j_service import Neo4jService

    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        database=os.getenv('POSTGRES_DB', 'phi_here'),
        user=os.getenv('POSTGRES_USER', 'phi_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'phi_password_dev'),
        min_size=1, max_size=5
    )

    neo4j = Neo4jService(
        uri=os.getenv('NEO4J_URI', 'bolt://neo4j:7687'),
        user=os.getenv('NEO4J_USER', 'neo4j'),
        password=os.getenv('NEO4J_PASSWORD')
    )
    await neo4j.connect()

    llm = AsyncOpenAI()

    return ExperimentContext(db_pool=db_pool, neo4j=neo4j, llm=llm)


async def load_events(ctx: ExperimentContext, min_claims: int = 15, limit: int = 3) -> list[LoadedEvent]:
    """Load events with at least min_claims claims."""
    events = await ctx.neo4j._execute_read('''
        MATCH (e:Event)-[:INTAKES]->(c:Claim)
        WITH e, count(c) as cnt
        WHERE cnt >= $min_claims
        RETURN e.id as id, e.canonical_name as name, cnt
        ORDER BY cnt DESC
        LIMIT $limit
    ''', {'min_claims': min_claims, 'limit': limit})

    return [LoadedEvent(id=e['id'], name=e['name'], claim_count=e['cnt']) for e in events]


async def load_claims_for_event(
    ctx: ExperimentContext,
    event_id: str,
    limit: int = 20
) -> tuple[list[Claim], dict[str, str]]:
    """
    Load claims for a specific event.

    Returns:
        (claims, claim_to_event_name) - claims list and mapping for ground truth
    """
    claims_data = await ctx.neo4j._execute_read('''
        MATCH (e:Event {id: $eid})-[:INTAKES]->(c:Claim)
        WHERE c.text IS NOT NULL
        OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
        OPTIONAL MATCH (c)-[:MENTIONS]->(ent:Entity)
        WITH c, p, e, collect({name: ent.canonical_name, type: ent.entity_type}) as entities
        RETURN c.id as id, c.text as text, p.domain as source, entities, e.canonical_name as event_name
        ORDER BY rand()
        LIMIT $limit
    ''', {'eid': event_id, 'limit': limit})

    claims = []
    claim_to_event = {}

    async with ctx.db_pool.acquire() as conn:
        await register_vector(conn)

        for row in claims_data:
            if not row['text']:
                continue

            # Get embedding
            embedding = await conn.fetchval(
                'SELECT embedding FROM core.claim_embeddings WHERE claim_id = $1',
                row['id']
            )

            # Process entities
            all_entities = set()
            anchor_entities = set()

            for ent in row['entities']:
                ent_name = ent.get('name')
                if ent_name:
                    all_entities.add(ent_name)
                    ent_type = ent.get('type')

                    # Anchor: PERSON, ORG, or non-hub LOCATION
                    if ent_type in ('PERSON', 'ORGANIZATION', 'ORG'):
                        anchor_entities.add(ent_name)
                    elif ent_type == 'LOCATION' and ent_name not in HUB_LOCATIONS:
                        anchor_entities.add(ent_name)

            # Convert embedding
            emb = None
            if embedding is not None:
                try:
                    if len(embedding) > 0:
                        emb = [float(x) for x in embedding]
                except:
                    pass

            claim = Claim(
                id=row['id'],
                text=row['text'],
                source=row['source'] or 'unknown',
                embedding=emb,
                entities=all_entities,
                anchor_entities=anchor_entities
            )
            claims.append(claim)
            claim_to_event[claim.id] = row['event_name'][:25]

    return claims, claim_to_event


async def load_multi_event_claims(
    ctx: ExperimentContext,
    claims_per_event: int = 20,
    num_events: int = 3,
    min_claims: int = 15
) -> tuple[list[Claim], dict[str, str], list[LoadedEvent]]:
    """
    Load claims from multiple events for clustering experiments.

    Returns:
        (all_claims, claim_to_event_name, events)
    """
    events = await load_events(ctx, min_claims=min_claims, limit=num_events)

    all_claims = []
    claim_to_event = {}

    for event in events:
        claims, mapping = await load_claims_for_event(ctx, event.id, limit=claims_per_event)
        all_claims.extend(claims)
        claim_to_event.update(mapping)

    return all_claims, claim_to_event, events


def log(msg: str):
    """Print with immediate flush for progressive output."""
    print(msg, flush=True)
