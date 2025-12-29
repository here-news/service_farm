"""
Diagnose aboutness signals for Wang Fuk Court event.
"""
import asyncio
import os
from openai import AsyncOpenAI
import asyncpg
from pgvector.asyncpg import register_vector
from services.neo4j_service import Neo4jService
from test_eu.core.epistemic_unit import Claim, cosine_similarity


async def diagnose():
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        database=os.getenv('POSTGRES_DB', 'phi_here'),
        user=os.getenv('POSTGRES_USER', 'phi_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'phi_password_dev'),
        min_size=1, max_size=3
    )

    neo4j = Neo4jService(
        uri=os.getenv('NEO4J_URI', 'bolt://neo4j:7687'),
        user=os.getenv('NEO4J_USER', 'neo4j'),
        password=os.getenv('NEO4J_PASSWORD')
    )
    await neo4j.connect()

    # Load Wang Fuk Court claims
    claims_data = await neo4j._execute_read('''
        MATCH (e:Event)-[:INTAKES]->(c:Claim)
        WHERE e.canonical_name = 'Wang Fuk Court Fire' AND c.text IS NOT NULL
        OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
        OPTIONAL MATCH (c)-[:MENTIONS]->(ent:Entity)
        WITH c, p, collect({name: ent.canonical_name, type: ent.entity_type}) as entities
        RETURN c.id as id, c.text as text, p.domain as source, entities
        ORDER BY rand()
        LIMIT 10
    ''', {})

    claims = []
    async with db_pool.acquire() as conn:
        await register_vector(conn)
        for row in claims_data:
            if not row['text']:
                continue

            embedding = await conn.fetchval(
                'SELECT embedding FROM core.claim_embeddings WHERE claim_id = $1',
                row['id']
            )

            all_ent = set()
            anchor_ent = set()
            for ent in row['entities']:
                ent_name = ent.get('name')
                if ent_name:
                    all_ent.add(ent_name)
                    if ent.get('type') in ('PERSON', 'ORGANIZATION', 'ORG'):
                        anchor_ent.add(ent_name)

            emb = None
            if embedding is not None:
                try:
                    if len(embedding) > 0:
                        emb = [float(x) for x in embedding]
                except:
                    pass

            claim = Claim(
                id=row['id'],
                text=row['text'][:80],
                source=row['source'] or 'unknown',
                embedding=emb,
                entities=all_ent,
                anchor_entities=anchor_ent
            )
            claims.append(claim)

    print(f'=== Wang Fuk Court Claims ({len(claims)}) ===')
    for c in claims:
        print(f'\n{c.id[:8]}: {c.text[:60]}...')
        print(f'  Anchors: {c.anchor_entities}')
        print(f'  Entities: {list(c.entities)[:5]}')

    # Compute pairwise signals
    print(f'\n=== Pairwise Analysis (first 5) ===')
    for i, c1 in enumerate(claims[:5]):
        for c2 in claims[i+1:5]:
            sem = cosine_similarity(c1.embedding, c2.embedding) if c1.embedding and c2.embedding else 0
            anchor_overlap = c1.anchor_entities & c2.anchor_entities
            entity_overlap = c1.entities & c2.entities

            print(f'\n{c1.id[:8]} vs {c2.id[:8]}:')
            print(f'  Semantic: {sem:.2f}')
            print(f'  Anchor overlap: {anchor_overlap}')
            print(f'  Entity overlap: {entity_overlap}')

    await db_pool.close()
    await neo4j.close()


if __name__ == '__main__':
    asyncio.run(diagnose())
