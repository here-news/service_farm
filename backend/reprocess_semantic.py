#!/usr/bin/env python3
"""
Reprocess pages through semantic analysis.

Usage:
    python reprocess_semantic.py --all          # Reprocess all pages
    python reprocess_semantic.py --status X    # Reprocess pages with status X
"""
import asyncio
import asyncpg
import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.neo4j_service import Neo4jService
from services.job_queue import JobQueue

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def clear_graph():
    """Clear claims and events from Neo4j (keep entities)"""
    neo4j = Neo4jService()
    await neo4j.connect()

    logger.info("🗑️ Clearing Neo4j graph (Claims, Events)...")

    # Delete events and their relationships
    await neo4j._execute_write("""
        MATCH (e:Event)
        DETACH DELETE e
    """, {})
    logger.info("   Deleted Event nodes")

    # Delete claims and their relationships
    await neo4j._execute_write("""
        MATCH (c:Claim)
        DETACH DELETE c
    """, {})
    logger.info("   Deleted Claim nodes")

    await neo4j.close()
    logger.info("✅ Graph cleared")


async def queue_pages_for_reprocessing(status_filter: str = None):
    """Queue pages for semantic reprocessing"""
    pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=5432,
        user=os.getenv('POSTGRES_USER', 'herenews_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
        database=os.getenv('POSTGRES_DB', 'herenews')
    )

    job_queue = JobQueue(os.getenv('REDIS_URL', 'redis://redis:6379'))
    await job_queue.connect()

    # Find pages to reprocess
    if status_filter:
        pages = await pool.fetch("""
            SELECT id, url FROM core.pages WHERE status = $1
        """, status_filter)
    else:
        pages = await pool.fetch("""
            SELECT id, url FROM core.pages
            WHERE status IN ('semantic_complete', 'event_complete', 'semantic_failed')
        """)

    logger.info(f"📄 Found {len(pages)} pages to reprocess")

    # Delete existing claims in PostgreSQL for these pages
    page_ids = [p['id'] for p in pages]
    if page_ids:
        deleted = await pool.execute("""
            DELETE FROM core.claims WHERE page_id = ANY($1)
        """, page_ids)
        logger.info(f"🗑️ Deleted PostgreSQL claims for {len(page_ids)} pages")

    # Reset page status and queue for semantic processing
    for page in pages:
        await pool.execute("""
            UPDATE core.pages
            SET status = 'extracted',
                current_stage = 'extraction',
                updated_at = NOW()
            WHERE id = $1
        """, page['id'])

        await job_queue.enqueue('queue:semantic:high', {
            'page_id': str(page['id']),
            'url': page['url']
        })
        logger.info(f"📤 Queued: {page['url']}")

    await pool.close()
    await job_queue.close()

    logger.info(f"✅ Queued {len(pages)} pages for semantic reprocessing")


async def main():
    parser = argparse.ArgumentParser(description='Reprocess pages through semantic analysis')
    parser.add_argument('--all', action='store_true', help='Reprocess all completed pages')
    parser.add_argument('--status', type=str, help='Reprocess pages with specific status')
    parser.add_argument('--no-clear', action='store_true', help='Skip clearing Neo4j graph')
    args = parser.parse_args()

    if not args.all and not args.status:
        parser.print_help()
        return

    # Clear graph first (unless --no-clear)
    if not args.no_clear:
        await clear_graph()

    # Queue pages for reprocessing
    status_filter = args.status if args.status else None
    await queue_pages_for_reprocessing(status_filter)


if __name__ == '__main__':
    asyncio.run(main())
