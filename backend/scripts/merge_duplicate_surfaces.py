#!/usr/bin/env python3
"""
Merge Duplicate Surfaces Script
================================

Merges L2 surfaces that have duplicate (scope_id, question_key) pairs.

This is a healing script to fix the L2 invariant violation where:
- Surface key = (scope_id, question_key) should be UNIQUE
- Duplicates were created during poll-mode processing race conditions

The merge strategy:
1. Find all (scope_id, question_key) groups with >1 surface
2. For each group, pick the OLDEST surface as the survivor
3. Merge claims from all duplicate surfaces into the survivor
4. Update incident membership to point to survivor
5. Delete the duplicate surfaces

Usage:
    # Dry run (analyze only)
    python scripts/merge_duplicate_surfaces.py --dry-run

    # Execute merge
    python scripts/merge_duplicate_surfaces.py --merge
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


async def find_duplicates(neo4j: Neo4jService) -> list:
    """Find all duplicate (scope_id, question_key) groups."""
    result = await neo4j._execute_read("""
        MATCH (s:Surface)
        WHERE s.scope_id IS NOT NULL AND s.question_key IS NOT NULL
        WITH s.scope_id as scope, s.question_key as qk, collect(s) as surfaces, count(s) as cnt
        WHERE cnt > 1
        RETURN scope, qk,
               [s IN surfaces | {id: s.id, created_at: s.created_at}] as surface_data,
               cnt
        ORDER BY cnt DESC
    """)
    return result


async def merge_surfaces(neo4j: Neo4jService, db_pool: asyncpg.Pool, dry_run: bool = True):
    """Merge duplicate surfaces."""
    logger.info("=== Duplicate Surface Merge ===")

    duplicates = await find_duplicates(neo4j)

    if not duplicates:
        logger.info("✓ No duplicate surfaces found - L2 invariant is intact")
        return

    logger.info(f"Found {len(duplicates)} duplicate (scope_id, question_key) groups")

    total_merged = 0
    total_deleted = 0

    for dup in duplicates:
        scope = dup['scope']
        qk = dup['qk']
        surfaces = dup['surface_data']

        # Sort by created_at (oldest first) to pick survivor
        # If created_at is None, sort by ID for determinism
        surfaces_sorted = sorted(
            surfaces,
            key=lambda s: (s.get('created_at') or '9999', s['id'])
        )

        survivor_id = surfaces_sorted[0]['id']
        duplicate_ids = [s['id'] for s in surfaces_sorted[1:]]

        logger.info(f"\n{scope}::{qk}")
        logger.info(f"  Survivor: {survivor_id}")
        logger.info(f"  Duplicates to merge: {duplicate_ids}")

        if dry_run:
            # Just show what would happen
            for dup_id in duplicate_ids:
                claim_count = await neo4j._execute_read("""
                    MATCH (s:Surface {id: $id})-[:CONTAINS]->(c:Claim)
                    RETURN count(c) as cnt
                """, {'id': dup_id})
                cnt = claim_count[0]['cnt'] if claim_count else 0
                logger.info(f"    {dup_id}: {cnt} claims would be merged")
        else:
            # Actually merge
            for dup_id in duplicate_ids:
                # 1. Move claims from duplicate to survivor (Surface CONTAINS Claims)
                await neo4j._execute_write("""
                    MATCH (dup:Surface {id: $dup_id})-[r:CONTAINS]->(c:Claim)
                    MATCH (survivor:Surface {id: $survivor_id})
                    MERGE (survivor)-[:CONTAINS]->(c)
                    DELETE r
                """, {'dup_id': dup_id, 'survivor_id': survivor_id})

                # 2. Update incident membership
                await neo4j._execute_write("""
                    MATCH (i:Incident)-[r:CONTAINS]->(dup:Surface {id: $dup_id})
                    MATCH (survivor:Surface {id: $survivor_id})
                    MERGE (i)-[:CONTAINS]->(survivor)
                    DELETE r
                """, {'dup_id': dup_id, 'survivor_id': survivor_id})

                # 3. Merge entities
                await neo4j._execute_write("""
                    MATCH (dup:Surface {id: $dup_id})
                    MATCH (survivor:Surface {id: $survivor_id})
                    SET survivor.entities = CASE
                        WHEN survivor.entities IS NULL THEN dup.entities
                        WHEN dup.entities IS NULL THEN survivor.entities
                        ELSE survivor.entities + [e IN dup.entities WHERE NOT e IN survivor.entities]
                    END,
                    survivor.anchor_entities = CASE
                        WHEN survivor.anchor_entities IS NULL THEN dup.anchor_entities
                        WHEN dup.anchor_entities IS NULL THEN survivor.anchor_entities
                        ELSE survivor.anchor_entities + [e IN dup.anchor_entities WHERE NOT e IN survivor.anchor_entities]
                    END
                """, {'dup_id': dup_id, 'survivor_id': survivor_id})

                # 4. Delete duplicate surface
                await neo4j._execute_write("""
                    MATCH (s:Surface {id: $id})
                    DETACH DELETE s
                """, {'id': dup_id})

                total_deleted += 1
                logger.info(f"    ✓ Merged {dup_id} into {survivor_id}")

            total_merged += 1

    # Also clean up PostgreSQL if we merged
    if not dry_run and total_deleted > 0:
        logger.info("\n🧹 Cleaning up PostgreSQL claim_surfaces...")
        async with db_pool.acquire() as conn:
            # Remove orphaned claim_surfaces entries
            await conn.execute("""
                DELETE FROM content.claim_surfaces
                WHERE surface_id NOT IN (
                    SELECT id FROM (
                        SELECT unnest($1::text[]) as id
                    ) t
                )
            """, [[dup['surface_data'][0]['id'] for dup in duplicates]])

    logger.info(f"\n=== Summary ===")
    logger.info(f"Duplicate groups: {len(duplicates)}")
    if dry_run:
        logger.info(f"Surfaces that would be deleted: {sum(len(d['surface_data'])-1 for d in duplicates)}")
        logger.info("\nRun with --merge to execute")
    else:
        logger.info(f"Surfaces merged: {total_merged}")
        logger.info(f"Duplicate surfaces deleted: {total_deleted}")


async def add_uniqueness_constraint(neo4j: Neo4jService, dry_run: bool = True):
    """Add uniqueness constraint for (scope_id, question_key)."""
    logger.info("\n=== Uniqueness Constraint ===")

    # Check if constraint already exists
    existing = await neo4j._execute_read("""
        SHOW CONSTRAINTS
        YIELD name, labelsOrTypes, properties
        WHERE 'Surface' IN labelsOrTypes
        RETURN name, properties
    """)

    has_constraint = any(
        'scope_id' in (c.get('properties') or []) and 'question_key' in (c.get('properties') or [])
        for c in existing
    )

    if has_constraint:
        logger.info("✓ Uniqueness constraint already exists")
        return

    if dry_run:
        logger.info("Would create constraint: Surface(scope_id, question_key) UNIQUE")
    else:
        # Create composite uniqueness constraint
        try:
            await neo4j._execute_write("""
                CREATE CONSTRAINT surface_scoped_key IF NOT EXISTS
                FOR (s:Surface)
                REQUIRE (s.scope_id, s.question_key) IS UNIQUE
            """)
            logger.info("✓ Created uniqueness constraint: Surface(scope_id, question_key)")
        except Exception as e:
            logger.warning(f"Could not create constraint (may need Enterprise): {e}")
            logger.info("The merge process will still work, constraint just prevents future duplicates")


async def main():
    parser = argparse.ArgumentParser(description="Merge duplicate L2 surfaces")
    parser.add_argument('--dry-run', action='store_true', help="Analyze only, don't merge")
    parser.add_argument('--merge', action='store_true', help="Execute merge")
    args = parser.parse_args()

    if not args.dry_run and not args.merge:
        parser.print_help()
        print("\nPlease specify --dry-run or --merge")
        return

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
        await merge_surfaces(neo4j, db_pool, dry_run=args.dry_run)
        await add_uniqueness_constraint(neo4j, dry_run=args.dry_run)
    finally:
        await db_pool.close()
        await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
