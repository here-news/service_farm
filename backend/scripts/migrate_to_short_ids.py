#!/usr/bin/env python3
"""
Migration script: UUID to Short ID format

Migrates all entity IDs from UUID (36 chars) to short prefixed format (11 chars):
- pg_xxxxxxxx - page
- cl_xxxxxxxx - claim
- en_xxxxxxxx - entity
- ev_xxxxxxxx - event
- sr_xxxxxxxx - source

Run this script ONCE after deploying the new code.

Usage:
    python scripts/migrate_to_short_ids.py [--dry-run]
"""
import asyncio
import argparse
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncpg
from neo4j import AsyncGraphDatabase

from utils.id_generator import uuid_to_short_id, is_uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ShortIdMigration:
    """Migrate UUIDs to short prefixed IDs"""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.db_pool = None
        self.neo4j_driver = None
        self.stats = {
            'pages': 0,
            'claims': 0,
            'entities': 0,
            'events': 0,
            'sources': 0,
            'relationships': 0
        }

    async def connect(self):
        """Connect to databases"""
        # PostgreSQL - use individual env vars like the rest of the app
        pg_host = os.getenv('POSTGRES_HOST', 'postgres')
        pg_port = os.getenv('POSTGRES_PORT', '5432')
        pg_db = os.getenv('POSTGRES_DB', 'herenews')
        pg_user = os.getenv('POSTGRES_USER', 'herenews_user')
        pg_password = os.getenv('POSTGRES_PASSWORD', 'herenews_pass')

        self.db_pool = await asyncpg.create_pool(
            host=pg_host,
            port=int(pg_port),
            database=pg_db,
            user=pg_user,
            password=pg_password
        )
        logger.info("✅ Connected to PostgreSQL")

        # Neo4j
        neo4j_uri = os.getenv('NEO4J_URI', 'bolt://neo4j:7687')
        neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
        neo4j_password = os.getenv('NEO4J_PASSWORD', 'herenews_neo4j_pass')

        self.neo4j_driver = AsyncGraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password)
        )
        await self.neo4j_driver.verify_connectivity()
        logger.info("✅ Connected to Neo4j")

    async def close(self):
        """Close connections"""
        if self.db_pool:
            await self.db_pool.close()
        if self.neo4j_driver:
            await self.neo4j_driver.close()

    async def migrate_postgresql(self):
        """Migrate PostgreSQL tables to use short IDs"""
        logger.info("📦 Starting PostgreSQL migration...")

        async with self.db_pool.acquire() as conn:
            # Check if migration already done (id column is text)
            col_type = await conn.fetchval("""
                SELECT data_type FROM information_schema.columns
                WHERE table_schema = 'core' AND table_name = 'pages' AND column_name = 'id'
            """)

            if col_type == 'text':
                logger.info("✅ PostgreSQL already migrated (id columns are TEXT)")
                return

            if self.dry_run:
                # Count records that would be migrated
                pages_count = await conn.fetchval("SELECT COUNT(*) FROM core.pages")
                claims_count = await conn.fetchval("SELECT COUNT(*) FROM core.claims") or 0
                logger.info(f"[DRY RUN] Would migrate PostgreSQL tables:")
                logger.info(f"  - core.pages: {pages_count} rows (id column: uuid → text)")
                logger.info(f"  - core.claims: {claims_count} rows (id, page_id columns)")
                self.stats['pages'] = pages_count
                self.stats['claims'] = claims_count
                return

            # Step 1: Drop views
            logger.info("  Step 1: Dropping dependent views...")
            await conn.execute("DROP VIEW IF EXISTS bridge.artifacts_for_users CASCADE")

            # Step 2: Drop ALL foreign key constraints in core and bridge schemas
            logger.info("  Step 2: Dropping all foreign key constraints...")
            fk_constraints = await conn.fetch("""
                SELECT conname, conrelid::regclass as table_name
                FROM pg_constraint c
                JOIN pg_namespace n ON c.connamespace = n.oid
                WHERE c.contype = 'f'
                AND n.nspname IN ('core', 'bridge')
            """)
            for fk in fk_constraints:
                logger.info(f"    Dropping {fk['conname']} on {fk['table_name']}")
                await conn.execute(f"ALTER TABLE {fk['table_name']} DROP CONSTRAINT IF EXISTS {fk['conname']}")

            # Step 3: Convert all columns to TEXT and update values
            logger.info("  Step 3: Converting columns...")

            # Pages (parent)
            await self._migrate_table(conn, 'core.pages', 'page', ['id'])

            # Claims
            await self._migrate_table(conn, 'core.claims', 'claim', ['id', 'page_id'])

            # Event embeddings
            await self._migrate_table(conn, 'core.event_embeddings', 'event', ['event_id'])

            # Other tables with various ID columns
            await self._migrate_table(conn, 'core.page_entities', 'page', ['page_id', 'entity_id'])
            await self._migrate_table(conn, 'core.page_events', 'page', ['page_id', 'event_id'])
            await self._migrate_table(conn, 'core.rogue_extraction_tasks', 'page', ['page_id'])
            await self._migrate_table(conn, 'core.claim_entities', 'claim', ['claim_id', 'entity_id'])
            await self._migrate_table(conn, 'bridge.artifact_metadata', 'page', ['artifact_id'])

            # Step 4: Recreate foreign key constraints
            logger.info("  Step 4: Recreating foreign key constraints...")
            fk_definitions = [
                ("core.claims", "claims_page_id_fkey", "page_id", "core.pages", "id"),
                ("core.page_entities", "page_entities_page_id_fkey", "page_id", "core.pages", "id"),
                ("core.page_events", "page_events_page_id_fkey", "page_id", "core.pages", "id"),
                ("core.rogue_extraction_tasks", "rogue_extraction_tasks_page_id_fkey", "page_id", "core.pages", "id"),
                ("bridge.artifact_metadata", "artifact_metadata_artifact_id_fkey", "artifact_id", "core.pages", "id"),
            ]
            for table, name, col, ref_table, ref_col in fk_definitions:
                try:
                    await conn.execute(f"""
                        ALTER TABLE {table}
                        ADD CONSTRAINT {name}
                        FOREIGN KEY ({col}) REFERENCES {ref_table}({ref_col})
                    """)
                    logger.info(f"    Recreated {name}")
                except Exception as e:
                    logger.warning(f"    Could not recreate {name}: {e}")

            # Step 5: Recreate views
            logger.info("  Step 5: Recreating views...")
            await conn.execute("""
                CREATE VIEW bridge.artifacts_for_users AS
                SELECT p.id,
                    p.canonical_url,
                    p.title,
                    p.status,
                    p.domain,
                    p.language,
                    p.word_count,
                    p.pub_time,
                    p.created_at,
                    am.submitted_by_id,
                    am.submission_source,
                    am.submitted_at,
                    am.user_metadata
                FROM core.pages p
                LEFT JOIN bridge.artifact_metadata am ON p.id = am.artifact_id
            """)

            logger.info(f"✅ PostgreSQL migration complete: {self.stats}")

    async def _migrate_table(self, conn, table: str, entity_type: str, id_columns: list):
        """Migrate a single table - convert UUID columns to TEXT with short IDs"""
        # Check if table exists
        schema, tbl = table.split('.') if '.' in table else ('public', table)
        exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = $1 AND table_name = $2
            )
        """, schema, tbl)

        if not exists:
            logger.info(f"  Skipping {table} (table does not exist)")
            return

        logger.info(f"  Migrating {table}...")

        for col in id_columns:
            # Determine the entity type for this column
            if col in ('page_id', 'artifact_id'):
                col_entity_type = 'page'
            elif col == 'event_id':
                col_entity_type = 'event'
            elif col == 'entity_id':
                col_entity_type = 'entity'
            elif col == 'claim_id':
                col_entity_type = 'claim'
            else:
                col_entity_type = entity_type

            # Check current column type
            col_type = await conn.fetchval("""
                SELECT data_type FROM information_schema.columns
                WHERE table_schema = $1 AND table_name = $2 AND column_name = $3
            """, schema, tbl, col)

            if col_type is None:
                logger.info(f"    {col} column does not exist, skipping")
                continue

            if col_type == 'text':
                logger.info(f"    {col} already TEXT, skipping")
                continue

            # Convert UUIDs to short IDs in-place using UPDATE
            # First, alter column type to TEXT (this preserves the UUID string representation)
            await conn.execute(f"""
                ALTER TABLE {table} ALTER COLUMN {col} TYPE TEXT USING {col}::TEXT
            """)
            logger.info(f"    Converted {col} column to TEXT")

            # Now update the values to short format
            rows = await conn.fetch(f"""
                SELECT {col} as old_id FROM {table}
                WHERE {col} ~ '^[0-9a-f]{{8}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{12}}$'
            """)

            count = 0
            for row in rows:
                old_id = row['old_id']
                if is_uuid(old_id):
                    new_id = uuid_to_short_id(old_id, col_entity_type)
                    await conn.execute(f"""
                        UPDATE {table} SET {col} = $1 WHERE {col} = $2
                    """, new_id, old_id)
                    count += 1

            self.stats[f"{col_entity_type}s"] = self.stats.get(f"{col_entity_type}s", 0) + count
            logger.info(f"    Converted {count} {col} values to short IDs")

    async def migrate_neo4j(self):
        """Migrate Neo4j nodes to use short IDs"""
        logger.info("📊 Starting Neo4j migration...")

        async with self.neo4j_driver.session() as session:
            # Check if migration already done
            result = await session.run("""
                MATCH (n) WHERE n.id IS NOT NULL
                WITH n LIMIT 1
                RETURN n.id as sample_id
            """)
            record = await result.single()

            if record:
                sample_id = record['sample_id']
                if sample_id and not is_uuid(sample_id):
                    logger.info("✅ Neo4j already migrated (IDs are short format)")
                    return

            if self.dry_run:
                # Count nodes that would be migrated
                for label, entity_type in [('Page', 'page'), ('Claim', 'claim'),
                                           ('Entity', 'entity'), ('Event', 'event'), ('Source', 'source')]:
                    result = await session.run(f"""
                        MATCH (n:{label})
                        WHERE n.id IS NOT NULL AND n.id =~ '.*-.*-.*-.*-.*'
                        RETURN count(n) as count
                    """)
                    record = await result.single()
                    count = record['count'] if record else 0
                    self.stats[f"{entity_type}s"] = count
                    if count > 0:
                        logger.info(f"  - {label} nodes: {count}")

                logger.info("[DRY RUN] Would migrate Neo4j nodes")
                return

            # Migrate each node type
            await self._migrate_neo4j_nodes(session, 'Page', 'page')
            await self._migrate_neo4j_nodes(session, 'Claim', 'claim')
            await self._migrate_neo4j_nodes(session, 'Entity', 'entity')
            await self._migrate_neo4j_nodes(session, 'Event', 'event')
            await self._migrate_neo4j_nodes(session, 'Source', 'source')

            logger.info(f"✅ Neo4j migration complete: {self.stats}")

    async def _migrate_neo4j_nodes(self, session, label: str, entity_type: str):
        """Migrate nodes of a specific label"""
        logger.info(f"  Migrating {label} nodes...")

        # Get all nodes with UUID format IDs
        result = await session.run(f"""
            MATCH (n:{label})
            WHERE n.id IS NOT NULL AND n.id =~ '.*-.*-.*-.*-.*'
            RETURN n.id as old_id
        """)

        count = 0
        async for record in result:
            old_id = record['old_id']
            if is_uuid(old_id):
                new_id = uuid_to_short_id(old_id, entity_type)

                # Update the node ID
                await session.run(f"""
                    MATCH (n:{label} {{id: $old_id}})
                    SET n.id = $new_id, n.old_uuid = $old_id
                """, {'old_id': old_id, 'new_id': new_id})

                count += 1

        self.stats[f"{entity_type}s"] = count
        logger.info(f"    Converted {count} {label} nodes")

    async def verify_migration(self):
        """Verify migration completed successfully"""
        logger.info("🔍 Verifying migration...")

        errors = []

        # Check PostgreSQL
        async with self.db_pool.acquire() as conn:
            # Check pages table
            sample = await conn.fetchval("SELECT id FROM core.pages LIMIT 1")
            if sample and is_uuid(sample):
                errors.append(f"PostgreSQL pages still has UUID: {sample}")

            # Check claims table
            sample = await conn.fetchval("SELECT id FROM core.claims LIMIT 1")
            if sample and is_uuid(sample):
                errors.append(f"PostgreSQL claims still has UUID: {sample}")

        # Check Neo4j
        async with self.neo4j_driver.session() as session:
            result = await session.run("""
                MATCH (n) WHERE n.id IS NOT NULL AND n.id =~ '.*-.*-.*-.*-.*'
                RETURN count(n) as uuid_count
            """)
            record = await result.single()
            uuid_count = record['uuid_count'] if record else 0

            if uuid_count > 0:
                errors.append(f"Neo4j still has {uuid_count} nodes with UUID format")

        if errors:
            logger.error("❌ Migration verification failed:")
            for err in errors:
                logger.error(f"  - {err}")
            return False

        logger.info("✅ Migration verification passed")
        return True

    async def run(self):
        """Run the full migration"""
        logger.info("=" * 60)
        logger.info("Short ID Migration")
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        logger.info("=" * 60)

        try:
            await self.connect()

            # Run migrations
            await self.migrate_postgresql()
            await self.migrate_neo4j()

            # Verify
            if not self.dry_run:
                await self.verify_migration()

            logger.info("=" * 60)
            logger.info("Migration Summary:")
            for key, count in self.stats.items():
                logger.info(f"  {key}: {count}")
            logger.info("=" * 60)

        finally:
            await self.close()


async def main():
    parser = argparse.ArgumentParser(description='Migrate UUIDs to short IDs')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without applying')
    args = parser.parse_args()

    migration = ShortIdMigration(dry_run=args.dry_run)
    await migration.run()


if __name__ == '__main__':
    asyncio.run(main())
