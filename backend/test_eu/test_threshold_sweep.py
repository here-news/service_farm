"""
Sweep aboutness threshold to find optimal value.
Runs LLM once then tests different thresholds.
"""
import asyncio
import os
from openai import AsyncOpenAI
import asyncpg
from pgvector.asyncpg import register_vector
from services.neo4j_service import Neo4jService
from test_eu.core.epistemic_unit import EmergenceEngine, Claim, Parameters


HUB_LOCATIONS = {'Hong Kong', 'China', 'United States', 'UK', 'United Kingdom', 'US'}


async def test():
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
    llm = AsyncOpenAI()

    events = await neo4j._execute_read('''
        MATCH (e:Event)-[:INTAKES]->(c:Claim)
        WITH e, count(c) as cnt
        WHERE cnt >= 10
        RETURN e.id as id, e.canonical_name as name, cnt
        ORDER BY cnt DESC
        LIMIT 3
    ''', {})

    print('Events:', [e['name'][:20] for e in events])

    claims_per_event = 7
    all_claims = []
    claim_to_event = {}

    for ev in events:
        claims_data = await neo4j._execute_read('''
            MATCH (e:Event {id: $eid})-[:INTAKES]->(c:Claim)
            WHERE c.text IS NOT NULL
            OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
            OPTIONAL MATCH (c)-[:MENTIONS]->(ent:Entity)
            WITH c, p, collect({name: ent.canonical_name, type: ent.entity_type}) as entities
            RETURN c.id as id, c.text as text, p.domain as source, entities
            ORDER BY rand()
            LIMIT $limit
        ''', {'eid': ev['id'], 'limit': claims_per_event})

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
                        ent_type = ent.get('type')
                        if ent_type in ('PERSON', 'ORGANIZATION', 'ORG'):
                            anchor_ent.add(ent_name)
                        elif ent_type == 'LOCATION' and ent_name not in HUB_LOCATIONS:
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
                all_claims.append(claim)
                claim_to_event[claim.id] = ev['name'][:25]

    print(f'Loaded {len(all_claims)} claims')

    # Run LLM once
    params = Parameters(aboutness_min_signals=1, aboutness_threshold=0.15, hub_max_df=5)
    engine = EmergenceEngine(llm=llm, params=params)

    print('Running LLM identity checks...')
    for claim in all_claims:
        await engine.add_claim(claim)
    print(f'Identity edges: {len(engine.claim_edges)}')

    # Compute surfaces once
    surfaces = engine.compute_surfaces()
    print(f'Surfaces: {len(surfaces)}')

    # Compute aboutness once (gets all edges, then threshold)
    aboutness = engine.compute_surface_aboutness()
    print(f'Aboutness edges: {len(aboutness)}\n')

    # Show all aboutness edges with scores
    print('All aboutness edges:')
    for s1, s2, score, ev in sorted(aboutness, key=lambda x: -x[2]):
        ev1 = list(set(claim_to_event.get(cid, '?') for cid in engine.surfaces[s1].claim_ids))[0][:10]
        ev2 = list(set(claim_to_event.get(cid, '?') for cid in engine.surfaces[s2].claim_ids))[0][:10]
        tag = 'CROSS' if ev1 != ev2 else 'same'
        print(f'  {s1}<->{s2}: {score:.3f} [{ev1}<->{ev2}] {tag}')

    # Sweep thresholds
    print(f'\n{"Threshold":<10} {"Events":<8} {"Frag":<6} {"Pure":<6} {"Cross":<6}')
    print('-' * 45)

    from collections import defaultdict

    for threshold in [0.15, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35]:
        # Build aboutness adjacency at this threshold
        adj = defaultdict(set)
        cross_edges = 0
        for s1, s2, score, _ in aboutness:
            if score >= threshold:
                adj[s1].add(s2)
                adj[s2].add(s1)
                ev1 = list(set(claim_to_event.get(cid, '?') for cid in engine.surfaces[s1].claim_ids))[0]
                ev2 = list(set(claim_to_event.get(cid, '?') for cid in engine.surfaces[s2].claim_ids))[0]
                if ev1 != ev2:
                    cross_edges += 1

        # Find connected components
        visited = set()
        event_groups = []
        for surface_id in engine.surfaces:
            if surface_id in visited:
                continue
            group = set()
            stack = [surface_id]
            while stack:
                curr = stack.pop()
                if curr in visited:
                    continue
                visited.add(curr)
                group.add(curr)
                stack.extend(adj[curr] - visited)
            event_groups.append(group)

        # Count pure events
        event_pure = 0
        for group in event_groups:
            gt_events = set()
            for sid in group:
                s = engine.surfaces[sid]
                for cid in s.claim_ids:
                    gt_events.add(claim_to_event.get(cid, '?'))
            if len(gt_events) == 1:
                event_pure += 1

        n_events = len(event_groups)
        frag = n_events / 3
        pure_pct = event_pure / n_events * 100 if n_events else 0

        print(f'{threshold:<10.2f} {n_events:<8} {frag:<6.1f} {pure_pct:<6.0f}% {cross_edges:<6}')

    await db_pool.close()
    await neo4j.close()


if __name__ == '__main__':
    asyncio.run(test())
