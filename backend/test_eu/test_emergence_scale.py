"""
Test epistemic unit emergence at larger scale.
"""
import asyncio
import os
import sys
from openai import AsyncOpenAI
import asyncpg
from pgvector.asyncpg import register_vector
from services.neo4j_service import Neo4jService
from test_eu.core.epistemic_unit import EmergenceEngine, Claim, Parameters


def log(msg):
    """Print with immediate flush for progressive output."""
    print(msg, flush=True)


HUB_LOCATIONS = {'Hong Kong', 'China', 'United States', 'UK', 'United Kingdom', 'US'}


async def test(claims_per_event: int = 10):
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

    # Get 3 events
    events = await neo4j._execute_read('''
        MATCH (e:Event)-[:INTAKES]->(c:Claim)
        WITH e, count(c) as cnt
        WHERE cnt >= 15
        RETURN e.id as id, e.canonical_name as name, cnt
        ORDER BY cnt DESC
        LIMIT 3
    ''', {})

    log('=' * 60)
    log(f'EMERGENCE TEST: {claims_per_event} claims/event')
    log('=' * 60)
    log('\nEvents:')
    for ev in events:
        log(f'  {ev["name"]}: {ev["cnt"]} total claims')

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
                    text=row['text'],
                    source=row['source'] or 'unknown',
                    embedding=emb,
                    entities=all_ent,
                    anchor_entities=anchor_ent
                )
                all_claims.append(claim)
                claim_to_event[claim.id] = ev['name'][:25]

    log(f'\nLoaded {len(all_claims)} claims total')

    # Use tuned parameters
    params = Parameters(
        aboutness_min_signals=1,
        aboutness_threshold=0.33,  # Tuned from sweep
        hub_max_df=5
    )

    engine = EmergenceEngine(llm=llm, params=params)

    # Add claims and track progress
    log('\n--- Adding Claims (Identity Detection) ---')
    identity_count = 0
    for i, claim in enumerate(all_claims):
        result = await engine.add_claim(claim)
        if result['relations']:
            identity_count += len(result['relations'])
            for r in result['relations']:
                ev1 = claim_to_event.get(claim.id, '?')[:12]
                ev2 = claim_to_event.get(r['other_id'], '?')[:12]
                cross = ' ⚠️CROSS' if ev1 != ev2 else ''
                log(f'  [{i+1}] {claim.id[:8]}→{r["other_id"][:8]}: {r["relation"]} ({r["confidence"]:.0%}) [{ev1}↔{ev2}]{cross}')

        # Progress indicator
        if (i + 1) % 5 == 0:
            log(f'  ... processed {i+1}/{len(all_claims)} claims, {identity_count} identity edges')

    log(f'\nTotal identity edges: {identity_count}')

    # Compute surfaces
    surfaces = engine.compute_surfaces()
    log(f'\n--- Surfaces (L2): {len(surfaces)} ---')

    # Analyze surface composition
    surface_stats = {'pure': 0, 'mixed': 0}
    for s in surfaces:
        events_in = set(claim_to_event.get(cid, '?') for cid in s.claim_ids)
        if len(events_in) == 1:
            surface_stats['pure'] += 1
        else:
            surface_stats['mixed'] += 1
            log(f'  ⚠️ MIXED {s.id}: {len(s.claim_ids)} claims from {events_in}')

    log(f'\nSurface purity: {surface_stats["pure"]}/{len(surfaces)} ({surface_stats["pure"]/len(surfaces)*100:.0f}%)')

    # Show surface distribution by event
    event_surfaces = {}
    for s in surfaces:
        events_in = list(set(claim_to_event.get(cid, '?') for cid in s.claim_ids))
        ev = events_in[0] if len(events_in) == 1 else 'mixed'
        event_surfaces[ev] = event_surfaces.get(ev, 0) + 1

    log('\nSurfaces per event:')
    for ev, cnt in sorted(event_surfaces.items()):
        log(f'  {ev}: {cnt} surfaces')

    # Compute aboutness
    aboutness = engine.compute_surface_aboutness()
    log(f'\n--- Aboutness Edges: {len(aboutness)} ---')

    # Analyze aboutness edges
    within_event = 0
    cross_event = 0
    for s1, s2, score, _ in aboutness:
        ev1 = list(set(claim_to_event.get(cid, '?') for cid in engine.surfaces[s1].claim_ids))[0]
        ev2 = list(set(claim_to_event.get(cid, '?') for cid in engine.surfaces[s2].claim_ids))[0]
        if ev1 == ev2:
            within_event += 1
        else:
            cross_event += 1
            log(f'  ⚠️ CROSS: {s1}↔{s2} ({score:.2f}) [{ev1[:12]}↔{ev2[:12]}]')

    log(f'\nWithin-event edges: {within_event}')
    log(f'Cross-event edges: {cross_event}')

    # Compute events
    events_out = engine.compute_events()
    log(f'\n--- Events (L3): {len(events_out)} ---')

    # Analyze events
    event_pure = 0
    event_mixed = 0
    for e in events_out:
        gt_events = set()
        total_claims = 0
        for sid in e.surface_ids:
            s = engine.surfaces[sid]
            total_claims += len(s.claim_ids)
            for cid in s.claim_ids:
                gt_events.add(claim_to_event.get(cid, '?'))

        if len(gt_events) == 1:
            event_pure += 1
            status = 'PURE'
        else:
            event_mixed += 1
            status = '⚠️MIXED'

        log(f'  {e.id}: {len(e.surface_ids)} surfaces, {total_claims} claims → {[g[:15] for g in gt_events]} [{status}]')

    # Summary
    log('\n' + '=' * 60)
    log('SUMMARY')
    log('=' * 60)
    log(f'Claims: {len(all_claims)} ({claims_per_event}/event × 3 events)')
    log(f'Identity edges: {identity_count}')
    log(f'Surfaces: {len(surfaces)} (purity: {surface_stats["pure"]}/{len(surfaces)})')
    log(f'Aboutness edges: {len(aboutness)} (within: {within_event}, cross: {cross_event})')
    log(f'Events: {len(events_out)} vs GT=3 (purity: {event_pure}/{len(events_out)})')
    log(f'Fragmentation: {len(events_out)/3:.2f}x')

    # Detect tensions
    meta_claims = engine.detect_tensions()
    log(f'\nMeta-claims: {len(meta_claims)}')
    tension_types = {}
    for mc in meta_claims:
        tension_types[mc.type] = tension_types.get(mc.type, 0) + 1
    for t, cnt in sorted(tension_types.items(), key=lambda x: -x[1]):
        log(f'  {t}: {cnt}')

    await db_pool.close()
    await neo4j.close()


if __name__ == '__main__':
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    asyncio.run(test(n))
