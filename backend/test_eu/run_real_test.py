"""
Test epistemic unit with real claims from database.
"""
import asyncio
import os
from openai import AsyncOpenAI
import asyncpg
from pgvector.asyncpg import register_vector
from services.neo4j_service import Neo4jService
from test_eu.core.epistemic_unit import EmergenceEngine, Claim, Parameters


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

    # Get 3 events
    events = await neo4j._execute_read('''
        MATCH (e:Event)-[:INTAKES]->(c:Claim)
        WITH e, count(c) as cnt
        WHERE cnt >= 10
        RETURN e.id as id, e.canonical_name as name, cnt
        ORDER BY cnt DESC
        LIMIT 3
    ''', {})

    print('=== EVENTS ===')
    for ev in events:
        print(f'  {ev["name"]}: {ev["cnt"]} claims')

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
                all_claims.append(claim)
                claim_to_event[claim.id] = ev['name'][:25]

    print(f'\nLoaded {len(all_claims)} claims')

    # Relaxed parameters
    params = Parameters(
        aboutness_min_signals=1,
        aboutness_threshold=0.30,
        hub_max_df=5
    )

    engine = EmergenceEngine(llm=llm, params=params)

    print('\n=== IDENTITY EDGES ===')
    for claim in all_claims:
        result = await engine.add_claim(claim)
        if result['relations']:
            for r in result['relations']:
                print(f'  {claim.id[:8]} -> {r["other_id"][:8]}: {r["relation"]} ({r["confidence"]:.0%})')

    print(f'Total identity edges: {len(engine.claim_edges)}')

    surfaces = engine.compute_surfaces()
    print(f'\n=== SURFACES: {len(surfaces)} ===')

    pure = 0
    for s in surfaces:
        events_in = set(claim_to_event.get(cid, '?') for cid in s.claim_ids)
        if len(events_in) == 1:
            pure += 1
            purity_tag = 'PURE'
        else:
            purity_tag = 'MIXED'
        ev_short = list(events_in)[0][:12] if len(events_in) == 1 else 'mixed'
        print(f'  {s.id}: {len(s.claim_ids)} claims, anchors={list(s.anchor_entities)[:2]}, event={ev_short} [{purity_tag}]')

    print(f'\nSurface purity: {pure}/{len(surfaces)}')

    aboutness = engine.compute_surface_aboutness()
    print(f'\n=== ABOUTNESS EDGES: {len(aboutness)} ===')
    for s1, s2, score, ev in aboutness:
        ev1 = list(set(claim_to_event.get(cid, '?') for cid in engine.surfaces[s1].claim_ids))[0][:10]
        ev2 = list(set(claim_to_event.get(cid, '?') for cid in engine.surfaces[s2].claim_ids))[0][:10]
        tag = 'CROSS' if ev1 != ev2 else 'same'
        print(f'  {s1}<->{s2}: {score:.2f} sig={ev.get("signals_met",0)} [{ev1}<->{ev2}] {tag}')

    events_out = engine.compute_events()
    print(f'\n=== EVENTS: {len(events_out)} (GT=3) ===')

    event_pure = 0
    for e in events_out:
        gt_events = set()
        for sid in e.surface_ids:
            s = engine.surfaces[sid]
            for cid in s.claim_ids:
                gt_events.add(claim_to_event.get(cid, '?')[:15])
        if len(gt_events) == 1:
            event_pure += 1
            purity_tag = 'PURE'
        else:
            purity_tag = 'MIXED'
        print(f'  {e.id}: {len(e.surface_ids)} surfaces -> {gt_events} [{purity_tag}]')

    print(f'\n=== SUMMARY ===')
    print(f'GT events: 3 | Predicted: {len(events_out)}')
    print(f'Fragmentation: {len(events_out)/3:.1f}x')
    print(f'Surface purity: {pure}/{len(surfaces)}')
    print(f'Event purity: {event_pure}/{len(events_out)}')

    await db_pool.close()
    await neo4j.close()


if __name__ == '__main__':
    asyncio.run(test())
