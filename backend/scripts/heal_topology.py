#!/usr/bin/env python3
"""
Topology Healing Script
=======================

Heals contaminated L2/L3/L4 topology by reprocessing all claims
with corrected scoped surface keying.

Usage:
    # Dry run (analyze only)
    python scripts/heal_topology.py --dry-run

    # Full heal (clear and requeue all claims)
    python scripts/heal_topology.py --heal

    # Heal but preserve Cases
    python scripts/heal_topology.py --heal --preserve-cases

Why healing is needed:
- Old surface keying used global question_key (e.g., "policy_announcement")
- This caused cross-event contamination (unrelated incidents sharing surfaces)
- New keying uses (scope_id, question_key) where scope_id comes from anchors
- Existing surfaces need to be rebuilt with correct scoping
"""

import asyncio
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncpg
from services.neo4j_service import Neo4jService

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


async def analyze_contamination(neo4j: Neo4jService):
    """Analyze current topology for contamination."""
    logger.info("=== Topology Contamination Analysis ===")

    # Count surfaces
    result = await neo4j._execute_read("""
        MATCH (s:Surface)
        RETURN
            count(s) as total,
            count(CASE WHEN s.scope_id IS NOT NULL THEN 1 END) as scoped
    """)
    total = result[0]['total']
    scoped = result[0]['scoped']
    unscoped = total - scoped

    logger.info(f"Surfaces: {total} total, {scoped} scoped, {unscoped} unscoped")

    if unscoped > 0:
        logger.warning(f"  CONTAMINATION: {unscoped} surfaces lack scope_id (created with old keying)")

    # Count incidents
    result = await neo4j._execute_read("MATCH (i:Incident) RETURN count(i) as count")
    incidents = result[0]['count']
    logger.info(f"Incidents: {incidents}")

    # Count cases
    result = await neo4j._execute_read("MATCH (c:Case) RETURN count(c) as count")
    cases = result[0]['count']
    logger.info(f"Cases: {cases}")

    # Count claims
    result = await neo4j._execute_read("MATCH (c:Claim) RETURN count(c) as count")
    claims = result[0]['count']
    logger.info(f"Claims: {claims}")

    # Check for mega-surfaces (potential contamination)
    result = await neo4j._execute_read("""
        MATCH (s:Surface)-[:CONTAINS_CLAIM]->(c:Claim)
        WITH s, count(c) as claim_count
        WHERE claim_count > 10
        RETURN s.id, s.question_key, claim_count
        ORDER BY claim_count DESC
        LIMIT 10
    """)

    if result:
        logger.warning("Potential mega-surfaces (>10 claims):")
        for r in result:
            logger.warning(f"  {r['s.id'][:12]}: qk={r['s.question_key']}, claims={r['claim_count']}")

    return {
        'total_surfaces': total,
        'scoped_surfaces': scoped,
        'unscoped_surfaces': unscoped,
        'incidents': incidents,
        'cases': cases,
        'claims': claims,
        'needs_healing': unscoped > 0
    }


async def heal_topology(
    db_pool: asyncpg.Pool,
    neo4j: Neo4jService,
    preserve_cases: bool = False
):
    """
    Clear L2/L3/L4 topology for reprocessing.

    The principled_weaver in poll mode will automatically pick up claims
    that are no longer linked to surfaces and reprocess them.
    """
    logger.info("=== Starting Topology Healing ===")

    # 1. Clear existing topology
    logger.info("🧹 Clearing L2 Surfaces...")
    await neo4j._execute_write("MATCH (s:Surface) DETACH DELETE s")

    logger.info("🧹 Clearing L3 Incidents...")
    await neo4j._execute_write("MATCH (i:Incident) DETACH DELETE i")

    logger.info("🧹 Clearing MetaClaims...")
    await neo4j._execute_write("MATCH (m:MetaClaim) DETACH DELETE m")

    if not preserve_cases:
        logger.info("🧹 Clearing L4 Cases...")
        await neo4j._execute_write("MATCH (c:Case) DETACH DELETE c")

    # 2. Clear PostgreSQL surface data
    logger.info("🧹 Clearing PostgreSQL surface tables...")
    async with db_pool.acquire() as conn:
        await conn.execute("TRUNCATE content.surface_centroids")
        await conn.execute("TRUNCATE content.claim_surfaces")

    # 3. Count claims that will be reprocessed
    result = await neo4j._execute_read("MATCH (c:Claim) RETURN count(c) as count")
    total_claims = result[0]['count'] if result else 0

    logger.info("=== Healing Complete ===")
    logger.info(f"📚 {total_claims} claims ready for reprocessing")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Start principled_weaver in poll mode:")
    logger.info("     docker exec herenews-workers python workers/principled_weaver.py --poll")
    logger.info("  2. After L3 is stable, run canonical_worker to rebuild Cases")


async def main():
    parser = argparse.ArgumentParser(description="Heal contaminated topology")
    parser.add_argument('--dry-run', action='store_true', help="Analyze only, don't heal")
    parser.add_argument('--heal', action='store_true', help="Clear and requeue all claims")
    parser.add_argument('--preserve-cases', action='store_true', help="Keep existing Cases")
    args = parser.parse_args()

    if not args.dry_run and not args.heal:
        parser.print_help()
        print("\nPlease specify --dry-run or --heal")
        return

    # Connect to databases
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        database=os.getenv('POSTGRES_DB', 'phi_here'),
        user=os.getenv('POSTGRES_USER', 'phi_user'),
        password=os.getenv('POSTGRES_PASSWORD', ''),
    )

    neo4j = Neo4jService()
    await neo4j.connect()

    try:
        # Always analyze first
        analysis = await analyze_contamination(neo4j)

        if args.dry_run:
            if analysis['needs_healing']:
                logger.info("\nRecommendation: Run with --heal to fix contamination")
            else:
                logger.info("\nTopology appears healthy (all surfaces scoped)")
            return

        if args.heal:
            if not analysis['needs_healing'] and analysis['total_surfaces'] > 0:
                logger.info("\nNo healing needed - all surfaces already scoped")
                logger.info("(Use --heal with --force to heal anyway)")
                return

            await heal_topology(db_pool, neo4j, args.preserve_cases)

    finally:
        await db_pool.close()
        await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
