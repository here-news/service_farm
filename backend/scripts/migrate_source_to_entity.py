#!/usr/bin/env python3
"""
Migration script: Convert Source nodes to Entity nodes with is_publisher=true

Source nodes are being deprecated in favor of Entity nodes with:
- entity_type = 'ORGANIZATION'
- is_publisher = true
- domain property for lookup
- dedup_key = 'publisher_{domain}'

This script:
1. Creates Entity nodes from Source nodes
2. Migrates PUBLISHED_BY relationships from Source to Entity
3. Deletes the old Source nodes

Run this script ONCE after deploying the new code.

Usage:
    python scripts/migrate_source_to_entity.py [--dry-run]
"""
import asyncio
import argparse
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j import AsyncGraphDatabase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def migrate_sources(dry_run: bool = False):
    """Migrate Source nodes to Entity nodes with is_publisher=true"""

    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://neo4j:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD', 'herenews_neo4j_pass')

    driver = AsyncGraphDatabase.driver(
        neo4j_uri,
        auth=(neo4j_user, neo4j_password)
    )

    try:
        await driver.verify_connectivity()
        logger.info("Connected to Neo4j")

        async with driver.session() as session:
            # Count Source nodes
            result = await session.run("MATCH (s:Source) RETURN count(s) as count")
            record = await result.single()
            source_count = record['count'] if record else 0

            logger.info(f"Found {source_count} Source nodes to migrate")

            if source_count == 0:
                logger.info("No Source nodes to migrate")
                return

            if dry_run:
                # Show what would be migrated
                result = await session.run("""
                    MATCH (s:Source)
                    RETURN s.id as id, s.canonical_name as name, s.domain as domain
                """)
                async for record in result:
                    logger.info(f"  Would migrate: {record['name']} ({record['domain']})")
                logger.info("[DRY RUN] No changes made")
                return

            # Migrate Source nodes to Entity nodes
            logger.info("Step 1: Creating Entity nodes from Source nodes...")
            result = await session.run("""
                MATCH (s:Source)
                WITH s
                CREATE (e:Entity {
                    id: replace(s.id, 'sr_', 'en_'),
                    canonical_name: s.canonical_name,
                    entity_type: 'ORGANIZATION',
                    domain: s.domain,
                    is_publisher: true,
                    dedup_key: 'publisher_' + toLower(s.domain),
                    mention_count: coalesce(s.mention_count, 1),
                    status: 'pending',
                    created_at: coalesce(s.created_at, datetime()),
                    migrated_from: s.id
                })
                RETURN count(e) as created
            """)
            record = await result.single()
            created = record['created'] if record else 0
            logger.info(f"  Created {created} Entity nodes")

            # Migrate PUBLISHED_BY relationships
            logger.info("Step 2: Migrating PUBLISHED_BY relationships...")
            result = await session.run("""
                MATCH (p:Page)-[r:PUBLISHED_BY]->(s:Source)
                MATCH (e:Entity {migrated_from: s.id})
                MERGE (p)-[r2:PUBLISHED_BY]->(e)
                ON CREATE SET r2.created_at = coalesce(r.created_at, datetime())
                WITH r
                DELETE r
                RETURN count(*) as migrated
            """)
            record = await result.single()
            migrated = record['migrated'] if record else 0
            logger.info(f"  Migrated {migrated} PUBLISHED_BY relationships")

            # Delete Source nodes
            logger.info("Step 3: Deleting Source nodes...")
            result = await session.run("""
                MATCH (s:Source)
                DETACH DELETE s
                RETURN count(*) as deleted
            """)
            record = await result.single()
            deleted = record['deleted'] if record else 0
            logger.info(f"  Deleted {deleted} Source nodes")

            # Remove migrated_from property
            logger.info("Step 4: Cleaning up migration markers...")
            await session.run("""
                MATCH (e:Entity)
                WHERE e.migrated_from IS NOT NULL
                REMOVE e.migrated_from
            """)

            logger.info("Migration complete!")

            # Verify
            result = await session.run("MATCH (s:Source) RETURN count(s) as count")
            record = await result.single()
            remaining = record['count'] if record else 0

            result = await session.run("""
                MATCH (e:Entity {is_publisher: true})
                RETURN count(e) as count
            """)
            record = await result.single()
            publishers = record['count'] if record else 0

            logger.info(f"Verification:")
            logger.info(f"  Remaining Source nodes: {remaining}")
            logger.info(f"  Publisher Entity nodes: {publishers}")

    finally:
        await driver.close()


async def main():
    parser = argparse.ArgumentParser(description='Migrate Source nodes to Entity nodes')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without applying')
    args = parser.parse_args()

    await migrate_sources(dry_run=args.dry_run)


if __name__ == '__main__':
    asyncio.run(main())
