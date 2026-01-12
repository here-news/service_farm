#!/usr/bin/env python3
"""
Bootstrap surfaces from existing claims.
Processes all claims with embeddings and creates surfaces.

Usage:
    docker exec herenews-app python bootstrap_surfaces.py
"""
import asyncio
import asyncpg
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.neo4j_service import Neo4jService
from workers.weaver_worker import WeaverWorker

async def bootstrap():
    print("Connecting to databases...")

    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'db'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        database=os.getenv('POSTGRES_DB', 'phi_here'),
        user=os.getenv('POSTGRES_USER', 'phi_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'phi_password_dev'),
        min_size=2,
        max_size=10
    )

    neo4j = Neo4jService(
        uri=os.getenv('NEO4J_URI', 'bolt://neo4j:7687'),
        user=os.getenv('NEO4J_USER', 'neo4j'),
        password=os.getenv('NEO4J_PASSWORD', 'password')
    )
    await neo4j.connect()

    # Clear existing surfaces
    print("Clearing existing surfaces...")
    await neo4j._execute_write('MATCH (s:Surface) DETACH DELETE s')
    async with db_pool.acquire() as conn:
        await conn.execute('TRUNCATE content.surface_centroids')
        await conn.execute('TRUNCATE content.claim_surfaces')

    # Create worker (no queue needed for direct processing)
    worker = WeaverWorker(db_pool, neo4j, None, worker_id=1)

    # Get all claims with embeddings
    async with db_pool.acquire() as conn:
        claims = await conn.fetch('''
            SELECT claim_id FROM core.claim_embeddings
            ORDER BY created_at ASC
        ''')

    claim_ids = [r['claim_id'] for r in claims]
    total = len(claim_ids)
    print(f"Processing {total} claims...\n")

    # Process all claims
    for i, cid in enumerate(claim_ids):
        try:
            result = await worker.process_claim(cid)

            if result:
                status = 'NEW' if result.is_new_surface else f'LINKED (sim={result.similarity:.3f})'
                print(f'[{i+1:4d}/{total}] {cid} -> {result.surface_id} {status}')
            else:
                print(f'[{i+1:4d}/{total}] {cid} -> SKIPPED')
        except Exception as e:
            print(f'[{i+1:4d}/{total}] {cid} -> ERROR: {e}')

        if (i + 1) % 100 == 0:
            print(f'\n  Progress: {worker.surfaces_created} new, {worker.surfaces_updated} linked\n')

    # Final stats
    result = await neo4j._execute_read('''
        MATCH (s:Surface)
        OPTIONAL MATCH (s)-[:CONTAINS]->(c:Claim)
        WITH s, count(c) as cc
        RETURN count(s) as surface_count,
               sum(cc) as total_claims,
               avg(cc) as avg_claims_per_surface,
               max(cc) as max_claims
    ''')

    r = result[0]
    print(f'\n{"="*60}')
    print(f'FINAL RESULTS')
    print(f'{"="*60}')
    print(f'   Total claims processed: {total}')
    print(f'   Total surfaces: {r["surface_count"]}')
    print(f'   New surfaces: {worker.surfaces_created}')
    print(f'   Claims linked: {worker.surfaces_updated}')
    print(f'   Linking ratio: {worker.surfaces_updated / total * 100:.1f}%')
    print(f'   Avg claims/surface: {r["avg_claims_per_surface"]:.2f}')
    print(f'   Max claims: {r["max_claims"]}')

    # Top surfaces
    multi = await neo4j._execute_read('''
        MATCH (s:Surface)-[:CONTAINS]->(c:Claim)
        WITH s, count(c) as cc
        WHERE cc > 1
        RETURN s.id as id, cc
        ORDER BY cc DESC
        LIMIT 10
    ''')

    if multi:
        print(f'\n   Top surfaces:')
        for m in multi:
            print(f'     {m["id"]}: {m["cc"]} claims')

    await db_pool.close()
    await neo4j.close()
    print("\nDone!")

if __name__ == "__main__":
    asyncio.run(bootstrap())
